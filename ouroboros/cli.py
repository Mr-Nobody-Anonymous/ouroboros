"""Offline-first CLI. Heavy training and upstream execution always require explicit selection."""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .config import ModelConfig, RecursiveCycleConfig, TuningConfig
from .core.checkpoints import verify_checkpoint
from .core.code_mutator import CodeMutator, NumericObjective
from .core.pareto_pruner import ParetoCheckpointPruner
from .data import DatasetBundle, load_samples
from .pipeline.self_loop import RecursiveTuningLoop
from .repositories import load_sources, sync_sources, verify_sources
from .storage import audit_storage


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
    if set(data["data"]) - {"train", "validation", "anchors"}:
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
    )
    return (
        ModelConfig(**model_values),
        TuningConfig(**data.get("tuning", {})),
        RecursiveCycleConfig(**cycle_values),
        bundle,
    )


def doctor():
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ("numpy", "yaml", "torch", "transformers", "peft", "trl", "easyeditor")
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
        "network": "Core commands never download a model. repos sync explicitly uses GitHub.",
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
            history = loop.run_cycle()
            return 1 if any(row.get("error") for row in history) else 0
        elif args.command == "run":
            model, tuning, cycle, dataset = load_run_config(args.config)
            if args.check_only:
                emit(
                    {
                        "valid": True,
                        "dataset_sha256": dataset.fingerprint(),
                        "counts": {
                            k: len(getattr(dataset, k)) for k in ("train", "validation", "anchors")
                        },
                        "model_loaded": False,
                    }
                )
            else:
                from .backends.transformers_backend import TransformersLoRABackend

                backend = TransformersLoRABackend(model, tuning, cycle.seed)
                loop = RecursiveTuningLoop(cycle, tuning, backend=backend, dataset=dataset)
                history = loop.run_cycle()
                return 1 if any(row.get("error") for row in history) else 0
        elif args.command == "evaluate":
            from .core.evaluator import SelfEvaluator

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
            suite = {}
            for sample, prediction in zip(
                samples, backend.predict([s.instruction for s in samples])
            ):
                suite.setdefault(sample.domain, []).append(
                    {"prediction": prediction, "target": sample.target}
                )
            emit({"scores": SelfEvaluator().evaluate_generation(suite), "checksum_verified": True})
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
        subprocess.SubprocessError,
    ) as exc:
        print(f"ouroboros: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
