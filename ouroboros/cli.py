"""Offline-first CLI. Heavy training and upstream execution always require explicit selection."""

import argparse
import importlib.util
import json
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .config import ModelConfig, RecursiveCycleConfig, TuningConfig
from .core.checkpoints import verify_checkpoint
from .core.code_mutator import CodeMutator, NumericObjective
from .core.pareto_pruner import ParetoCheckpointPruner
from .core.weight_trials import (
    WeightEditPlan,
    apply_weight_patch,
    run_weight_edit_demo,
    run_weight_edit_trial,
    verify_weight_patch,
)
from .data import DatasetBundle, load_samples, unique_samples
from .pipeline.self_loop import RecursiveTuningLoop
from .repositories import load_sources, sync_sources, verify_sources
from .storage import audit_storage
from .tensor_inspection import inspect_safetensors


def emit(value):
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def load_run_config(path):
    """Resolve paths relative to the config file, not the caller's working directory."""
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(data) - {"model", "tuning", "cycle", "data"}
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    if not isinstance(data.get("data"), dict) or not {"train", "validation"} <= set(data["data"]):
        raise ValueError("Config requires data.train and data.validation")
    if set(data["data"]) - {"train", "validation", "anchors", "test"}:
        raise ValueError("Unknown data configuration keys")

    def resolve(value):
        p = Path(value).expanduser()
        return str((path.parent / p).resolve()) if not p.is_absolute() else str(p)

    model_values = data.get("model", {})
    if model_values.get("model_name_or_path"):
        model_values["model_name_or_path"] = resolve(model_values["model_name_or_path"])
    cycle_values = data.get("cycle", {})
    for key in ("checkpoint_dir", "output_dir"):
        cycle_values[key] = resolve(cycle_values.get(key, f"../{key.removesuffix('_dir')}s"))
    bundle = DatasetBundle(
        load_samples(resolve(data["data"]["train"])),
        load_samples(resolve(data["data"]["validation"])),
        load_samples(resolve(data["data"]["anchors"]), anchors=True)
        if data["data"].get("anchors")
        else [],
        load_samples(resolve(data["data"]["test"])) if data["data"].get("test") else [],
    )
    return (
        ModelConfig(**model_values),
        TuningConfig(**data.get("tuning", {})),
        RecursiveCycleConfig(**cycle_values),
        bundle,
    )


def load_weight_trial_config(path):
    """A data-only edit specification; no Python modules or generated code are imported."""
    import numpy as np

    path = Path(path).resolve()
    values = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model",
        "edit",
        "validation",
        "direction_file",
        "output_dir",
        "license",
        "provenance",
    }
    if not isinstance(values, dict) or set(values) != required:
        raise ValueError("Weight-trial config must contain exactly: " + ", ".join(sorted(required)))

    def resolve(value):
        return str((path.parent / Path(value).expanduser()).resolve())

    model_values = dict(values["model"])
    ModelConfig(**model_values)
    if not model_values.get("model_name_or_path"):
        raise ValueError("An explicit local model path is required")
    model_values["model_name_or_path"] = resolve(model_values["model_name_or_path"])
    model = ModelConfig(**model_values)
    if model.use_qlora:
        raise ValueError("Weight trials do not support quantized models")
    for key in ("license", "provenance"):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError("Explicit model license and provenance are required")
    direction_path = Path(resolve(values["direction_file"]))
    if direction_path.stat().st_size > 1024 * 1024:
        raise ValueError("Direction JSON exceeds 1 MiB")
    if direction_path.suffix != ".json":
        raise ValueError("Direction must be a JSON vector, not executable or pickle data")
    raw_direction = np.asarray(json.loads(direction_path.read_text(encoding="utf-8")))
    if raw_direction.dtype.kind not in "fiu":
        raise ValueError("Direction values must be numbers, not strings or booleans")
    direction = raw_direction.astype(np.float64)
    if (
        direction.ndim not in (1, 2)
        or not direction.size
        or not np.isfinite(direction).all()
        or not np.isfinite(np.linalg.norm(direction))
        or not np.linalg.norm(direction)
    ):
        raise ValueError("Direction must be a finite nonzero vector")
    if direction.ndim == 2 and direction.shape[1] != 1:
        raise ValueError("Direction must be a vector or a single column")
    validation = unique_samples(load_samples(resolve(values["validation"])))
    if not validation:
        raise ValueError("Nonempty validation data are required")
    return {
        "model": model,
        "plan": WeightEditPlan(**values["edit"]),
        "direction": direction,
        "validation": validation,
        "output_dir": resolve(values["output_dir"]),
        "provenance": values["provenance"],
        "license_name": values["license"],
    }


def doctor():
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "numpy",
            "yaml",
            "safetensors",
            "torch",
            "transformers",
            "peft",
            "trl",
            "easyeditor",
            "mcp",
            "huggingface_hub",
            "gguf",
            "math_verify",
            "lmformatenforcer",
        )
    }
    lfs = shutil.which("git-lfs") is not None
    return {
        "version": __version__,
        "python": sys.version.split()[0],
        "packages": packages,
        "tools": {
            "git": shutil.which("git") is not None,
            "git_lfs": lfs,
            "mergekit_yaml": shutil.which("mergekit-yaml") is not None,
            "docker": shutil.which("docker") is not None,
        },
        "default_mode": "measured tiny CPU classifier; NOT an LLM",
        "network": "Only explicit models fetch/fetch-smoke and repos sync access the network; inference is local-only.",
        "ready": {
            "core": all(packages[name] for name in ("numpy", "yaml", "safetensors")),
            "self_edit": all(packages[name] for name in ("torch", "transformers", "safetensors")),
            "mcp": packages["mcp"],
            "research_lab": all(
                packages[name] for name in ("torch", "transformers", "safetensors")
            ),
            "boxed_math_grader": packages["math_verify"],
            "structured_research_commands": packages["lmformatenforcer"],
        },
        "install_hints": {
            "self_edit": "python -m pip install -e '.[llm]' -c constraints/ml-tested.txt",
            "mcp": "python -m pip install -e '.[mcp]'",
            "structured_commands": "python -m pip install -e '.[structured]'",
            "github_smoke_model": "python -m pip install -e '.[llm,smoke]' -c constraints/ml-tested.txt",
        },
    }


def parser():
    root = argparse.ArgumentParser(description="Ouroboros: auditable local adaptation experiments")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Train/evaluate a tiny NumPy LoRA classifier (not an LLM)"
    )
    demo.add_argument("--generations", type=int, default=3)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--output-dir", default="outputs")
    demo.add_argument("--checkpoint-dir", default="checkpoints")
    run = commands.add_parser("run", help="Real LoRA SFT on a pre-downloaded local causal LM")
    run.add_argument("--config", required=True)
    run.add_argument(
        "--check-only", action="store_true", help="Validate config/data without loading a model"
    )
    evaluate = commands.add_parser(
        "evaluate", help="Verify hashes and re-evaluate an exported checkpoint"
    )
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument(
        "--model-path", help="Relocated local base model directory (LLM checkpoints only)"
    )
    commands.add_parser("doctor", help="Inspect dependencies without loading models")
    research = commands.add_parser(
        "research", help="Bounded training-research foundation; independent gates and SQLite memory"
    )
    research_ops = research.add_subparsers(dest="research_operation", required=True)
    research_ops.add_parser("tools", help="Versioned research commands and authority boundaries")
    research_ops.add_parser(
        "deepseek",
        help="Source-linked DeepSeek features: adopted, partial, deferred, not applicable",
    )
    dataset = research_ops.add_parser(
        "data", help="Build or verify the owned multi-domain research dataset (no downloads)"
    )
    dataset.add_argument("--directory", required=True)
    dataset.add_argument("--model-path", help="Local model directory for the generated config")
    dataset.add_argument("--seed", type=int, default=17)
    dataset.add_argument("--verify", action="store_true", help="Verify an existing dataset instead")
    selection = research_ops.add_parser(
        "compare-selection", help="Predeclared offline comparison of selection policies"
    )
    selection.add_argument("--database")
    selection.add_argument("--fixture", action="store_true", help="Synthetic machinery check only")
    selection.add_argument("--min-episodes", type=int, default=20)
    selection.add_argument("--alpha", type=float, default=0.05)
    selection.add_argument("--output", help="Directory for the frozen selection design")
    foundation_run = research_ops.add_parser(
        "run", help="One frozen research loop; never overwrite a baseline"
    )
    foundation_run.add_argument("--config", required=True)
    foundation_run.add_argument("--check-only", action="store_true")
    campaign = research_ops.add_parser(
        "campaign", help="Explicit finite cycles, fresh hidden data, one memory ledger"
    )
    campaign.add_argument("--config", action="append", required=True)
    campaign.add_argument("--max-cycles", type=int, required=True)
    history = research_ops.add_parser(
        "history", help="Read-only bounded authoritative result retrieval"
    )
    history.add_argument("--database", required=True)
    history.add_argument("--query", default="")
    history.add_argument("--limit", type=int, default=20)
    recover = research_ops.add_parser(
        "recover", help="Mark a dead unfinished run interrupted; never reopen a holdout"
    )
    recover.add_argument("--database", required=True)
    recover.add_argument("--experiment-id", required=True)
    inspect_run = research_ops.add_parser(
        "verify-run", help="Verify a foundation artifact tree without model loading"
    )
    inspect_run.add_argument("--report", required=True)
    inspect_run.add_argument(
        "--database", help="Explicit relocated authoritative database, otherwise use recorded path"
    )
    foundation_init = research_ops.add_parser(
        "init", help="Create explicitly synthetic foundation fixtures/config, no model download"
    )
    foundation_init.add_argument("directory")
    foundation_init.add_argument("--model-path", required=True)
    init = commands.add_parser(
        "init", help="Create a small self-edit project; never overwrite files"
    )
    init.add_argument("directory", nargs="?", default="ouroboros-lab")
    init.add_argument(
        "--discovery",
        action="store_true",
        help="Also copy the broader five-role discovery evaluation, license and strict config",
    )
    self_edit = commands.add_parser(
        "self-edit", help="Same local model selects bounded edits to its own weights"
    )
    self_edit.add_argument("--config", required=True)
    self_edit.add_argument(
        "--check-only",
        action="store_true",
        help="Check schema/data without a model load or download",
    )
    lab = commands.add_parser(
        "lab", help="Frozen-editor research sessions, instrumented copies and multi-tensor programs"
    )
    lab_ops = lab.add_subparsers(dest="lab_operation", required=True)
    lab_tools = lab_ops.add_parser(
        "tools", help="JSON catalogue of data-only research actions (no model load)"
    )
    lab_tools.add_argument(
        "--discovery",
        action="store_true",
        help="Hypothesis/audit/gradient-access/retention profile",
    )
    lab_run = lab_ops.add_parser(
        "run", help="Run configured model-authored or labelled scripted sessions"
    )
    lab_run.add_argument("--config", required=True)
    lab_run.add_argument("--check-only", action="store_true")
    lab_experiment = lab_ops.add_parser(
        "experiment",
        help="Plan-first discovery controls/ablations on a receipt-verified local checkpoint",
    )
    from .lab.experiment import add_arguments

    add_arguments(lab_experiment)
    lab_diagnose = lab_ops.add_parser(
        "diagnose",
        help="Measure practice readiness only; never generate reserved/final answers or edits",
    )
    lab_diagnose.add_argument("--config", required=True)
    lab_diagnose.add_argument("--check-only", action="store_true")
    lab_verify = lab_ops.add_parser(
        "verify-bundle", help="Verify an accepted full-state-bound patch without Torch"
    )
    lab_verify.add_argument("--bundle", required=True)
    lab_eval = lab_ops.add_parser(
        "evaluate", help="Explicit independent local reevaluation; always undo patch"
    )
    lab_eval.add_argument("--config", required=True)
    lab_eval.add_argument("--bundle", required=True)
    lab_eval.add_argument("--split", choices=["validation", "test"], default="test")
    lab_eval.add_argument("--seed", type=int)
    lab_import = lab_ops.add_parser(
        "import-data", help="Convert explicit local math/GSM8K/DeepScaleR rows; never download"
    )
    lab_import.add_argument("--input", required=True)
    lab_import.add_argument("--output-dir", required=True)
    lab_import.add_argument(
        "--format", choices=["sample", "math", "gsm8k", "deepscaler"], required=True
    )
    lab_import.add_argument("--source", required=True)
    lab_import.add_argument("--license", required=True)
    lab_import.add_argument("--domain", default="math")
    lab_bench = lab_ops.add_parser(
        "benchmark",
        help="Plan an offline lm-eval call; execution requires --execute and separate installation",
    )
    lab_bench.add_argument("--model-dir", required=True)
    lab_bench.add_argument("--output-dir", required=True)
    lab_bench.add_argument("--task", action="append", required=True)
    lab_bench.add_argument("--limit", type=int, default=100)
    lab_bench.add_argument("--max-new-tokens", type=int, default=4096)
    lab_bench.add_argument("--device", default="cpu")
    lab_bench.add_argument("--timeout-seconds", type=int, default=1800)
    lab_bench.add_argument("--execute", action="store_true")
    models = commands.add_parser(
        "models", help="Explicit model provisioning or offline checksum verification"
    )
    model_ops = models.add_subparsers(dest="model_operation", required=True)
    fetch = model_ops.add_parser(
        "fetch", help="Explicit Hub download: safetensors/config/tokenizer only, pinned receipt"
    )
    fetch.add_argument("--repo", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    fetch.add_argument("--directory", required=True)
    fetch.add_argument("--revision", default="main")
    fetch.add_argument("--max-bytes", type=int, default=1024**3)
    smoke = model_ops.add_parser(
        "fetch-smoke",
        help="Explicit pinned GitHub pretrained GGUF -> float32 safetensors conversion",
    )
    smoke.add_argument("--directory", required=True)
    model_verify = model_ops.add_parser(
        "verify", help="Verify local files against their download receipt (no network)"
    )
    model_verify.add_argument("--directory", required=True)
    mcp = commands.add_parser(
        "mcp", help="Local stdio MCP server; read-only unless experiments explicitly enabled"
    )
    mcp.add_argument("--workspace", default=".")
    mcp.add_argument("--allow-experiments", action="store_true")
    mcp.add_argument(
        "--config",
        action="append",
        default=[],
        help="Allowlist one self-edit config; repeat to approve more",
    )
    mcp.add_argument(
        "--lab-config",
        action="append",
        default=[],
        help="Allowlist a research config AND its input data/action bytes",
    )
    mcp.add_argument("--timeout-seconds", type=int, default=600)
    repos = commands.add_parser(
        "repos", help="List, verify, or explicitly fetch locked upstream sources"
    )
    repos.add_argument("operation", nargs="?", choices=["list", "verify", "sync"], default="list")
    repos.add_argument("--root", default=".")
    repos.add_argument("--name", action="append", default=[])
    storage = commands.add_parser(
        "storage", help="Inventory files or preview/apply run-scoped checkpoint pruning"
    )
    storage.add_argument("operation", choices=["audit", "prune"])
    storage.add_argument("--root", default=".")
    storage.add_argument("--run-id")
    storage.add_argument("--keep", type=int, default=2)
    storage.add_argument("--protect", action="append", default=[])
    storage.add_argument("--apply", action="store_true")
    objective = commands.add_parser(
        "objective", help="Interpret a bounded scalar arithmetic objective (no Python execution)"
    )
    objective.add_argument("expression")
    objective.add_argument("--loss", type=float, default=1.0)
    objective.add_argument("--beta", type=float, default=0.1)
    objective.add_argument("--step", type=float, default=0)
    mutation = commands.add_parser(
        "propose", help="Export a source mutation as a patch; never execute or apply it"
    )
    mutation.add_argument("--source", required=True)
    mutation.add_argument(
        "--type", choices=["dpo_beta", "learning_rate", "lora_dropout"], required=True
    )
    mutation.add_argument("--value", type=float, required=True)
    mutation.add_argument("--output-dir", required=True)
    weights = commands.add_parser(
        "weights", help="Inspect safetensors or explicitly evaluate reversible weight patches"
    )
    operations = weights.add_subparsers(dest="weight_operation", required=True)
    inspect = operations.add_parser(
        "inspect", help="Bounded metadata/numeric inspection; no model loading"
    )
    inspect.add_argument("--file", required=True)
    inspect.add_argument("--tensor")
    inspect.add_argument("--limit", type=int, default=100)
    inspect.add_argument("--values", type=int, default=0)
    inspect.add_argument(
        "--checksum", action="store_true", help="Explicitly hash the entire file (may be large)"
    )
    edit_demo = operations.add_parser(
        "demo", help="Real 2x2 Torch edit fixture, NOT an LLM benchmark"
    )
    edit_demo.add_argument("--output-dir", default="outputs/weight-trials")
    verify = operations.add_parser(
        "verify-patch", help="Check patch integrity without Torch or a base model"
    )
    verify.add_argument("--patch", required=True)
    trial = operations.add_parser(
        "trial", help="Measure a local model edit; always restore the model"
    )
    trial.add_argument("--config", required=True)
    trial.add_argument("--check-only", action="store_true")
    trial.add_argument(
        "--list-parameters",
        action="store_true",
        help="Load the model and list matrix names instead of editing",
    )
    evaluate_patch = operations.add_parser(
        "evaluate-patch",
        help="Apply a verified patch for scoring, then undo it; never save a full model",
    )
    evaluate_patch.add_argument("--config", required=True)
    evaluate_patch.add_argument("--patch", required=True)
    evaluate_patch.add_argument("--data", required=True)
    return root


def main(argv=None):
    args_list = list(sys.argv[1:] if argv is None else argv)
    # Keep the original invocation usable, but replace its fabricated simulation with a labeled demo.
    if args_list and args_list[0] in {"--generations", "--simulate", "--domain", "--lora-r"}:
        print(
            "Legacy flags now run the measured demo, not a language-model benchmark.",
            file=sys.stderr,
        )
        filtered, i = [], 0
        while i < len(args_list):
            if args_list[i] == "--simulate":
                i += 1
            elif args_list[i] in {"--domain", "--lora-r"}:
                i += 2
            else:
                filtered.append(args_list[i])
                i += 1
        args_list = ["demo", *filtered]
    args = parser().parse_args(args_list)
    try:
        if args.command == "doctor":
            emit(doctor())
        elif args.command == "research":
            if args.research_operation == "tools":
                from .research.contracts import commands as research_commands

                emit(research_commands())
            elif args.research_operation == "deepseek":
                from .research.applicability import report as deepseek_report

                result = deepseek_report(Path(__file__).resolve().parents[1])
                emit(result)
                if result["missing_local_evidence"]:
                    return 2
            elif args.research_operation == "data":
                from .research.dataset import verify_dataset, write_dataset

                result = (
                    verify_dataset(args.directory)
                    if args.verify
                    else write_dataset(args.directory, seed=args.seed, model_path=args.model_path)
                )
                emit(result)
            elif args.research_operation == "compare-selection":
                from .research.memory import ExperimentMemory
                from .research.selection import analyse, compare_fixture

                if args.fixture:
                    result = compare_fixture(min_episodes=args.min_episodes)
                else:
                    if not args.database:
                        raise ValueError("compare-selection needs --database or --fixture")
                    memory = ExperimentMemory(args.database, readonly=True)
                    try:
                        result = analyse(
                            memory,
                            alpha=args.alpha,
                            min_episodes=args.min_episodes,
                            directory=args.output,
                        )
                    finally:
                        memory.close()
                emit(result)
                return 3 if result["status"] == "insufficient_data" else 0
            elif args.research_operation == "run":
                from .research.controller import run_research

                result = run_research(args.config, check_only=args.check_only)
                emit(result)
                if result.get("state") in ("failed", "interrupted"):
                    return 2
            elif args.research_operation == "campaign":
                from .research.controller import run_campaign

                result = run_campaign(args.config, max_cycles=args.max_cycles)
                emit(result)
                if any(row["state"] in ("failed", "interrupted") for row in result["reports"]):
                    return 2
            elif args.research_operation in ("history", "recover"):
                from .research.memory import ExperimentMemory

                memory = ExperimentMemory(
                    args.database, readonly=args.research_operation == "history"
                )
                try:
                    if args.research_operation == "history":
                        emit(
                            {
                                "results": memory.search(args.query, limit=args.limit),
                                "integrity_verified": True,
                            }
                        )
                    else:
                        memory.recover_interrupted(args.experiment_id)
                        emit(
                            {
                                "state": memory.state(args.experiment_id),
                                "automatically_resumed": False,
                            }
                        )
                finally:
                    memory.close()
            elif args.research_operation == "verify-run":
                from .research.project import verify_run

                emit(verify_run(args.report, database=args.database))
            else:
                from .research.project import init_foundation

                emit(init_foundation(args.directory, args.model_path))
        elif args.command == "init":
            from .project import init_project

            emit(init_project(args.directory, discovery=args.discovery))
        elif args.command == "lab":
            if args.lab_operation == "tools":
                from .lab.tools import capabilities

                emit(capabilities(discovery=args.discovery))
            elif args.lab_operation == "experiment":
                from .lab.experiment import run_experiment

                emit(run_experiment(args))
            elif args.lab_operation == "verify-bundle":
                from .lab.patches import verify_bundle

                emit(verify_bundle(args.bundle))
            elif args.lab_operation == "import-data":
                from .lab.benchmarks import import_local_data

                emit(
                    import_local_data(
                        args.input,
                        args.output_dir,
                        format=args.format,
                        source=args.source,
                        license_name=args.license,
                        domain=args.domain,
                    )
                )
            elif args.lab_operation == "benchmark":
                from .lab.benchmarks import execute_benchmark

                emit(
                    execute_benchmark(
                        args.model_dir,
                        args.output_dir,
                        args.task,
                        allow_execution=args.execute,
                        timeout=args.timeout_seconds,
                        limit=args.limit,
                        device=args.device,
                        max_new_tokens=args.max_new_tokens,
                    )
                )
            else:
                from .lab.config import lab_config_summary, load_lab_config

                config = load_lab_config(args.config)
                if args.lab_operation in ("run", "diagnose") and args.check_only:
                    emit(lab_config_summary(config))
                else:
                    from .lab.runner import (
                        evaluate_configured_bundle,
                        run_configured_diagnosis,
                        run_configured_lab,
                    )

                    emit(
                        run_configured_lab(config)
                        if args.lab_operation == "run"
                        else run_configured_diagnosis(config)
                        if args.lab_operation == "diagnose"
                        else evaluate_configured_bundle(
                            config, args.bundle, split=args.split, seed=args.seed
                        )
                    )
        elif args.command == "models":
            from .model_files import fetch_model, fetch_smoke_model, verify_model_files

            if args.model_operation == "fetch":
                result = fetch_model(
                    args.repo, args.directory, revision=args.revision, max_bytes=args.max_bytes
                )
            elif args.model_operation == "fetch-smoke":
                result = fetch_smoke_model(args.directory)
            else:
                result = verify_model_files(args.directory)
            emit(result)
        elif args.command == "self-edit":
            from .self_edit import (
                load_self_edit_config,
                run_configured_self_edit,
                self_edit_config_summary,
            )

            config = load_self_edit_config(args.config)
            if args.check_only:
                emit(self_edit_config_summary(config))
            else:
                emit(run_configured_self_edit(config))
        elif args.command == "mcp":
            from .mcp_server import create_server

            create_server(
                args.workspace,
                allow_experiments=args.allow_experiments,
                approved_configs=args.config,
                approved_lab_configs=args.lab_config,
                timeout_seconds=args.timeout_seconds,
            ).run(transport="stdio")
        elif args.command == "demo":
            loop = RecursiveTuningLoop(
                RecursiveCycleConfig(
                    max_generations=args.generations,
                    seed=args.seed,
                    output_dir=args.output_dir,
                    checkpoint_dir=args.checkpoint_dir,
                ),
                TuningConfig(
                    lora_r=4, lora_alpha=4, learning_rate=0.5, num_train_epochs=30, batch_size=16
                ),
            )
            loop.run_cycle()
            return 1 if loop.report["status"] in {"failed", "final_test_failed"} else 0
        elif args.command == "run":
            model, tuning, cycle, dataset = load_run_config(args.config)
            if args.check_only:
                emit(
                    {
                        "valid": True,
                        "dataset_sha256": dataset.fingerprint(),
                        "counts": {
                            k: len(getattr(dataset, k))
                            for k in ("train", "validation", "anchors", "test")
                        },
                        "model_loaded": False,
                    }
                )
            else:
                from .backends.transformers_backend import TransformersLoRABackend

                backend = TransformersLoRABackend(model, tuning, cycle.seed)
                loop = RecursiveTuningLoop(cycle, tuning, backend=backend, dataset=dataset)
                loop.run_cycle()
                return 1 if loop.report["status"] in {"failed", "final_test_failed"} else 0
        elif args.command == "evaluate":
            from .core.evaluator import evaluate_samples

            metadata = verify_checkpoint(args.checkpoint)
            if metadata["backend"]["backend"] == "numpy-lora-classifier":
                from .backends.numpy_backend import NumpyLoRABackend

                backend = NumpyLoRABackend.load(args.checkpoint)
            elif metadata["backend"]["backend"] == "transformers-peft-lora":
                from .backends.transformers_backend import TransformersLoRABackend

                values = dict(metadata["backend"]["model_config"])
                if args.model_path:
                    values["model_name_or_path"] = args.model_path
                backend = TransformersLoRABackend(
                    ModelConfig(**values),
                    TuningConfig(**metadata["backend"]["tuning_config"]),
                    adapter_path=args.checkpoint,
                )
            else:
                raise ValueError("Unsupported checkpoint backend")
            samples = load_samples(args.data)
            emit({**evaluate_samples(backend.predict, samples), "checksum_verified": True})
        elif args.command == "repos":
            if args.operation == "list":
                emit(load_sources(args.root))
            else:
                result = (
                    sync_sources(args.root, args.name)
                    if args.operation == "sync"
                    else verify_sources(args.root)
                )
                emit(result)
                return 0 if all(row["status"] == "verified" for row in result) else 1
        elif args.command == "storage":
            if args.operation == "audit":
                if args.apply:
                    raise ValueError(
                        "Audit never deletes files; --apply is valid only for checkpoint pruning"
                    )
                emit(audit_storage(args.root))
            else:
                if not args.run_id:
                    raise ValueError("Checkpoint pruning requires --run-id")
                pruner = ParetoCheckpointPruner(args.keep)
                # Baseline is always protected; caller should also protect the run.json latest_checkpoint.
                protected = {"generation_0000", *args.protect}
                plan = pruner.plan(args.root, run_id=args.run_id, protected=protected)
                if args.apply:
                    plan["deleted"] = pruner.prune_checkpoints(
                        args.root, run_id=args.run_id, protected=protected, apply=True
                    )
                emit(plan)
        elif args.command == "weights":
            if args.weight_operation == "inspect":
                emit(
                    inspect_safetensors(
                        args.file,
                        tensor=args.tensor,
                        limit=args.limit,
                        values=args.values,
                        checksum=args.checksum,
                    )
                )
            elif args.weight_operation == "demo":
                emit(run_weight_edit_demo(args.output_dir))
            elif args.weight_operation == "verify-patch":
                emit(verify_weight_patch(args.patch))
            else:
                config = load_weight_trial_config(args.config)
                if args.weight_operation == "trial" and args.check_only:
                    emit(
                        {
                            "valid": True,
                            "model_loaded": False,
                            "plan": asdict(config["plan"]),
                            "direction_shape": list(config["direction"].shape),
                            "validation_examples": len(config["validation"]),
                            "note": "Target tensor existence/shape are checked only when a model is loaded.",
                        }
                    )
                    return 0
                if args.weight_operation == "evaluate-patch":
                    verify_weight_patch(args.patch)
                from .backends.local_model import LocalCausalLM

                backend = LocalCausalLM(config["model"])
                if args.weight_operation == "trial":
                    if args.list_parameters:
                        emit(
                            {
                                "model_loaded": True,
                                "parameters": [
                                    {"name": name, "shape": list(p.shape), "dtype": str(p.dtype)}
                                    for name, p in backend.model.named_parameters()
                                    if p.ndim == 2
                                ],
                                "edited": False,
                            }
                        )
                        return 0
                    result = run_weight_edit_trial(
                        backend.model,
                        backend.predict,
                        config["validation"],
                        config["direction"],
                        config["plan"],
                        config["output_dir"],
                        provenance=config["provenance"],
                        license_name=config["license_name"],
                        model_info=backend.describe(),
                    )
                    emit(result)
                    # A measured rejection is a completed experiment, not a successful model edit.
                    return 0
                else:
                    from .core.evaluator import evaluate_samples

                    samples = load_samples(args.data)
                    apply_weight_patch(backend.model, args.patch)
                    try:
                        result = evaluate_samples(backend.predict, samples)
                    finally:
                        apply_weight_patch(backend.model, args.patch, reverse=True)
                    emit({**result, "patch_reverted": True, "model_saved": False})
        elif args.command == "objective":
            emit(
                {
                    "value": NumericObjective(args.expression)(
                        loss=args.loss, beta=args.beta, step=args.step
                    ),
                    "arbitrary_python_executed": False,
                }
            )
        elif args.command == "propose":
            source = Path(args.source)
            code = source.read_text(encoding="utf-8")
            mutator = CodeMutator()
            proposed = mutator.propose_loss_mutation(code, args.type, args.value)
            emit(
                {
                    "patch": mutator.export_proposal(
                        code, proposed, args.output_dir, relative_path=source.name
                    ),
                    "applied": False,
                }
            )
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        ImportError,
        sqlite3.Error,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ouroboros: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
