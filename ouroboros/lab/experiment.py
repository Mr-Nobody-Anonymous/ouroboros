"""Plan-first, serial closed-loop controls and ablations on one local checkpoint.

Run the SAME fixed protocol for each checkpoint. No accuracy/gain assertion makes CI pass;
only protocol integrity is asserted. Failed scientific hypotheses are first-class results.
"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from importlib import metadata, resources
from pathlib import Path

from ..config import ModelConfig
from ..core.checkpoints import atomic_json, file_sha256
from ..data import DatasetBundle, load_samples
from ..model_files import verify_model_files
from ..self_edit import model_fingerprint, preflight_model
from .config import LabPolicy, Sampling
from .controls import HostControl, NoEditControl, random_references
from .discovery_config import discovery_summary, validate_discovery
from .journal import SUCCESS_LEVELS

CONDITIONS = (
    "no_edit",
    "human_directed",
    "random_valid",
    "unconstrained",
    "schema",
    "model_directed_gradient",
    "schema_no_feedback",
    "schema_full_context",
    "schema_sparse_ops",
    "schema_staged",
    "schema_staged_gradient",
)
MODEL_CONDITIONS = CONDITIONS[3:]
SEEDS = (17, 42)
FAMILY_MODELS = 2
FAMILY_ALPHA = 0.05
EFFECTIVE_ALPHA = FAMILY_ALPHA / (len(CONDITIONS) * len(SEEDS) * FAMILY_MODELS)


def add_arguments(parser):
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--workdir", required=True, help="New directory; never overwrite an experiment"
    )
    parser.add_argument(
        "--profile",
        choices=("135m", "360m"),
        default="135m",
        help="Explicit parameter/byte resource profile, NOT different acceptance rules",
    )
    parser.add_argument(
        "--data-dir", default=str(resources.files("ouroboros.resources").joinpath("discovery"))
    )
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help="Comma-separated subset; family correction remains fixed at the full registered family",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(map(str, SEEDS)),
        help="Subset of the two registered editor/control seeds, not an adaptive seed search",
    )
    parser.add_argument("--check-only", action="store_true")


def policy_for(profile):
    return LabPolicy(
        max_sessions=1,
        max_turns=24,
        max_seconds=600,
        max_invalid_actions=4,
        max_repeated_actions=3,
        max_forward_calls=20,
        max_edit_actions=8,
        max_model_parameters=400_000_000 if profile == "360m" else 300_000_000,
        max_model_bytes=2 * 1024**3 if profile == "360m" else 1024**3,
        max_edit_tensors=2,
        max_changed_entries=8192,
        max_read_tokens=512,
        max_history_chars=8000,
        diagnosis_problems=32,
        max_problems_per_call=32,
        require_significance=True,
        forgetting_threshold=0,
        improvement_threshold=0,
        alpha=EFFECTIVE_ALPHA,
        permutation_draws=10_000,
        cpu_threads=2,
    )


def load_data(directory):
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    if manifest_path.stat().st_size > 1024**2:
        raise ValueError("Discovery data manifest exceeds 1 MiB")
    manifest = json.loads(manifest_path.read_text())
    if set(manifest.get("files", {})) != {
        f"{key}.json" for key in ("practice", "development", "reserved", "final", "retention")
    }:
        raise ValueError("Expected a five-role, hash-bound discovery data manifest")
    for name, sha in manifest["files"].items():
        if (directory / name).is_symlink() or file_sha256(directory / name) != sha:
            raise ValueError("Prepared discovery evaluation bytes changed")
    rows = {
        key: load_samples(directory / f"{key}.json")
        for key in ("practice", "development", "reserved", "final", "retention")
    }
    dataset = DatasetBundle(rows["practice"], rows["reserved"], test=rows["final"])
    return dataset, rows, manifest


def code_bindings():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "ouroboros").rglob("*.py"))
    wrapper = root / "scripts/run_discovery_experiment.py"
    if wrapper.exists():
        paths.append(wrapper)
    return {str(path.relative_to(root)): file_sha256(path) for path in paths}


def _first_prompt(report):
    for event in report["sessions"][0]["events"]:
        path = Path(report["report_path"]).parent / event["artifact"]
        value = json.loads(path.read_text())
        if value["editor_prompt"] is not None:
            return hashlib.sha256(value["editor_prompt"].encode()).hexdigest()
    return None


def summarize_run(report, condition, seed):
    session = report["sessions"][0]
    journal = json.loads(Path(session["edit_journal"]["path"]).read_text())
    edits = [
        row
        for row in journal["experiments"]
        if row["operation"].get("action") in ("edit", "restore")
    ]
    nonempty = [
        row for row in edits if row["actual_nonempty_change"] and row["validation_result"]["passed"]
    ]
    accepted = session["status"] == "accepted"
    generalized = report["final_test"]["generalization_assessment"]["strict_independent_gain"]
    retained = report["retention"].get("retention_of_validated_useful_edit")
    return {
        "condition": condition,
        "seed": seed,
        "status": session["status"],
        "model_authored_actions": report["model_authored_actions"],
        "initial_prompt_sha256": _first_prompt(report),
        "editor_generations": session["editor_generations"],
        "parsed_commands": sum(row["parsed_json"] for row in session["events"]),
        "valid_commands": sum(row["status"] == "ok" for row in session["events"]),
        "invalid_commands": session["counters"]["invalid_actions"],
        "valid_action_names": [row["action"] for row in session["events"] if row["status"] == "ok"],
        "edit_attempts": len(edits),
        "valid_nonempty_edits": len(nonempty),
        "changed_entries_by_edit": [row["changes"]["changed_entries"] for row in nonempty],
        "practice_outcomes": [row.get("outcome") for row in edits],
        "baseline": session["practice_baseline"],
        "adaptive_summary": journal["adaptive_summary"],
        "development_check": session.get("development_check"),
        "reserved_check": session.get("reserved_check"),
        "accepted": accepted,
        "final": report["final_test"]["paired"],
        "terminal_diagnostic_final": report.get("terminal_candidate_test"),
        "retention": {
            key: value
            for key, value in report["retention"].items()
            if key not in ("artifact", "artifact_sha256")
        },
        "milestones": {
            "command_validity": any(row["status"] == "ok" for row in session["events"]),
            "edit_validity": bool(nonempty),
            "edit_usefulness": accepted,
            "unseen_generalization": generalized if accepted else None,
            "retention": retained,
            "validated_autonomous_useful_edit": bool(
                report["model_authored_actions"]
                and report["model"].get("is_llm", False)
                and ((report.get("source") or {}).get("download_receipt") or {}).get(
                    "verified", False
                )
                and nonempty
                and accepted
                and generalized
                and retained
            ),
            "learned_future_self_improvement": False,
        },
        "base_restored": report["base_restored"],
        "source_files_unchanged": report["source_files_unchanged"],
        "report_path": report["report_path"],
        "report_sha256": file_sha256(report["report_path"]),
        "journal_path": session["edit_journal"]["path"],
        "journal_sha256": session["edit_journal"]["sha256"],
        "measurement_cache_hits": journal["measurement_cache_hits"],
        "actual_session_generations": session["counters"]["generated_sequences"],
    }


def github_evidence(summary, run):
    """Small independently readable check annotations; full journals stay in artifacts."""

    def paired(value):
        if value is None:
            return None
        return {
            key: value[key]
            for key in (
                "baseline",
                "candidate",
                "accuracy_delta",
                "completion_nll_delta",
                "problem_sign_flip",
                "wrong_to_right",
                "right_to_wrong",
            )
            if key in value
        }

    evidence = {
        "kind": "ouroboros-discovery-run-evidence",
        "profile": summary["profile"],
        "parameters": summary["model"]["parameters"],
        "state_sha256": summary["state_sha256"],
        "dataset_sha256": summary["dataset_sha256"],
        "plan_sha256": summary["plan_sha256"],
        **{
            key: run[key]
            for key in (
                "condition",
                "seed",
                "status",
                "model_authored_actions",
                "initial_prompt_sha256",
                "editor_generations",
                "parsed_commands",
                "valid_commands",
                "invalid_commands",
                "valid_action_names",
                "edit_attempts",
                "valid_nonempty_edits",
                "changed_entries_by_edit",
                "practice_outcomes",
                "baseline",
                "adaptive_summary",
                "accepted",
                "milestones",
                "base_restored",
                "source_files_unchanged",
                "journal_sha256",
                "measurement_cache_hits",
                "actual_session_generations",
            )
        },
        "development_check": run["development_check"],
        "reserved_check": run["reserved_check"],
        "final": paired(run["final"]),
        "terminal_diagnostic_final": paired(run["terminal_diagnostic_final"]["paired"])
        if run["terminal_diagnostic_final"]
        else None,
        "retention": {
            key: run["retention"][key]
            for key in (
                "status",
                "candidate_kind",
                "category_retention",
                "completion_nll_retention",
                "retention_of_validated_useful_edit",
                "base_restored",
                "actual_generated_sequences",
                "actual_loss_evaluated_problems",
                "measurement_cache_hits",
            )
            if key in run["retention"]
        },
        "retention_paired": {
            key: paired(run["retention"][key])
            for key in (
                "baseline_vs_candidate",
                "candidate_vs_after",
                "subsequent_only_vs_combined",
            )
            if key in run["retention"]
        },
    }
    if os.environ.get("GITHUB_ACTIONS") == "true":
        message = json.dumps(evidence, separators=(",", ":"), allow_nan=False)
        if len(message.encode()) > 60_000:
            raise ValueError("Discovery annotation exceeds its explicit bounded evidence size")
        print(
            "::notice title=Ouroboros discovery evidence::"
            + message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A"),
            flush=True,
        )
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(
                    f"\n- **{summary['profile']} / {run['seed']} / {run['condition']}**: {run['status']}; {run['valid_commands']} valid commands, {run['valid_nonempty_edits']} real edits; accepted={run['accepted']}, autonomous validated milestone={run['milestones']['validated_autonomous_useful_edit']}. Practice baseline {run['baseline']['accuracy']:.4f}, NLL {run['baseline']['loss']['problem_mean_nll']:.6f}. Full failed trials and exact values are in the artifact, not substituted by test counts.\n"
                )
    return evidence


def _check_data_binding(directory, manifest, manifest_sha256):
    directory = Path(directory)
    if file_sha256(directory / "manifest.json") != manifest_sha256 or any(
        file_sha256(directory / name) != sha for name, sha in manifest["files"].items()
    ):
        raise RuntimeError(
            "Evaluation inputs changed after preregistration; preserve this study and start a new one"
        )


def diagnose_results(runs):
    rows = []
    for seed in sorted({run["seed"] for run in runs}):
        arms = {run["condition"]: run for run in runs if run["seed"] == seed}
        human = arms.get("human_directed")
        rows.append(
            {
                "seed": seed,
                "question": "Can the permitted mechanism help a host-directed practice control?",
                "observed": human["practice_outcomes"] if human else None,
                "interpretation": "A measured NLL decrease establishes local loss usefulness only. If accuracy/development/reserved do not pass, the full useful-edit mechanism is NOT positively controlled; do not attribute autonomous failure solely to reasoning.",
            }
        )
        for a, b, factor in (
            ("unconstrained", "schema", "structural command decoding"),
            ("schema", "model_directed_gradient", "practice sensitivity access"),
            ("schema", "schema_no_feedback", "within-context edited-practice feedback"),
            ("schema", "schema_full_context", "prompt verbosity/context load"),
            ("schema", "schema_sparse_ops", "available numerical operators"),
            (
                "schema",
                "schema_staged",
                "host workflow-phase scaffolding, still model-authored numerical edits",
            ),
            (
                "schema_staged",
                "schema_staged_gradient",
                "practice sensitivity access with matched staged workflow",
            ),
        ):
            left, right = arms.get(a), arms.get(b)
            rows.append(
                {
                    "seed": seed,
                    "factor": factor,
                    "arms": [a, b],
                    "observed": [
                        {
                            key: value[key]
                            for key in (
                                "status",
                                "valid_commands",
                                "valid_nonempty_edits",
                                "practice_outcomes",
                                "accepted",
                            )
                        }
                        for value in (left, right)
                    ]
                    if left and right
                    else None,
                    "interpretation": "Matched fixed protocol/seed comparison, descriptive with very few editor trajectories. No edits/no use of the differing capability cannot identify an error-to-parameter mapping, causal parameter-selection deficit, or whether a larger/trained model would solve it. Host stalls are not model abstentions.",
                }
            )
    return rows


def run_experiment(args):
    conditions = [value.strip() for value in args.conditions.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    if (
        not conditions
        or len(set(conditions)) != len(conditions)
        or not set(conditions) <= set(CONDITIONS)
    ):
        raise ValueError("Choose a nonempty unique subset of registered conditions")
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(SEEDS):
        raise ValueError("Choose a unique subset of the two registered seeds")
    # Execution ordering is fixed regardless of flag ordering; random control depends only on
    # the corresponding host practice trial, never another arm's held-out result.
    conditions = [name for name in CONDITIONS if name in conditions]
    if "random_valid" in conditions and "human_directed" not in conditions:
        raise ValueError(
            "Matched random control requires the corresponding host control in this study"
        )
    policy = policy_for(args.profile)
    dataset, splits, data_manifest = load_data(args.data_dir)
    model = ModelConfig(
        model_name_or_path=str(Path(args.model_dir).resolve()),
        device="cpu",
        torch_dtype="float32",
        max_new_tokens=128,
        prompt_format="chat",
    )
    discovery_base = {
        "development": splits["development"],
        "retention": splits["retention"],
        "followup_operations": [
            {"op": "add", "name": "pending-final-block-name", "indices": [0], "value": 0.001}
        ],
    }
    validate_discovery(discovery_base, dataset, policy)
    if args.check_only:
        return {
            "valid": True,
            "model_loaded": False,
            "network_accessed": False,
            "conditions": conditions,
            "seeds": seeds,
            "counts": data_manifest["counts"],
            "policy": asdict(policy),
            "model": asdict(model),
            "evidence_levels": SUCCESS_LEVELS,
        }
    root = Path(args.workdir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    path = root / "discovery_summary.json"
    summary = {
        "kind": "ouroboros-closed-loop-discovery",
        "schema_version": 1,
        "status": "planning",
        "profile": args.profile,
        "conditions": conditions,
        "seeds": seeds,
        "runs": [],
        "validated_autonomous_useful_edit": False,
        "learned_future_self_improvement": False,
    }
    atomic_json(path, summary)
    try:
        receipt = verify_model_files(model.model_name_or_path)
        source = preflight_model({"model": model, "policy": policy})
        if args.profile == "360m" and source["stored_tensor_elements"] <= 135_000_000:
            raise ValueError("The 360m profile requires a genuinely larger-than-135M checkpoint")
        config_path = Path(model.model_name_or_path) / "config.json"
        architecture = json.loads(config_path.read_text())
        layers = architecture.get("num_hidden_layers")
        if type(layers) is not int or not 1 <= layers <= 128:
            raise ValueError(
                "This registered host control requires the bounded Llama-compatible block layout"
            )
        target = f"model.layers.{layers - 1}.post_attention_layernorm.weight"
        discovery_base["followup_operations"][0]["name"] = target
        bindings = code_bindings()
        manifest_sha256 = file_sha256(Path(args.data_dir) / "manifest.json")
        plan = {
            "data_manifest_sha256": manifest_sha256,
            "kind": "ouroboros-discovery-preregistration",
            "schema_version": 1,
            "model": asdict(model),
            "receipt": receipt,
            "source": source,
            "data_manifest": data_manifest,
            "dataset_sha256": dataset.fingerprint(),
            "conditions": conditions,
            "seeds": seeds,
            "policy": asdict(policy),
            "sampling": asdict(Sampling()),
            "editor": {
                "max_new_tokens": 512,
                "sampling": asdict(Sampling(temperature=0.3, top_p=0.95, top_k=20)),
            },
            "host_control": {
                "target": target,
                "attempts": 2,
                "relative_doses": [0.01, 0.03],
                "top_gradient_entries": 256,
                "practice_ids": list(range(len(dataset.train))),
                "policy": "Host-directed NLL gradient descent, rollback harmful trials; recompute gradient for second dose. Matched random uses corresponding host trial support and actual norm with independent direction, always rolling back its first dose. No-edit ends without a proposed edit. All use the same hypothesis/edit host validation.",
            },
            "discovery": discovery_summary(validate_discovery(discovery_base, dataset, policy)),
            "family": {
                "models": FAMILY_MODELS,
                "seeds": list(SEEDS),
                "arms": list(CONDITIONS),
                "alpha": FAMILY_ALPHA,
                "per_gate_alpha": EFFECTIVE_ALPHA,
                "method": f"Conservative Bonferroni over {FAMILY_MODELS} checkpoints x {len(SEEDS)} seeds x {len(CONDITIONS)} arms ({FAMILY_MODELS * len(SEEDS) * len(CONDITIONS)} possible submissions); both sealed development and reserved gates require the same corrected alpha. Subsetting does not loosen it; no result-based repeated attempts.",
            },
            "code_sha256": bindings,
            "evidence_levels": SUCCESS_LEVELS,
            "cache": "Exact within-process, per-backend source/state/config/sample/seed measurement cache. Cache hits are reported separately from actual generation/NLL counts; no synthetic answers, external callback cache or cross-seed replication claim.",
            "final_policy": "Freeze selected and last valid terminal diagnostic overlays before one final comparison per frozen overlay. Diagnostic final never reselects or rescues a failed gate. Then fixed four-cell retention test; no result returns to the editor. No accepted edit means no retention-of-useful-edit claim.",
            "limitations": "Bounded small selected evaluation, not an upstream benchmark reproduction or power guarantee. Exact prompt separation does not ensure pretraining decontamination; synthetic templates can correlate problems. A frozen checkpoint with context feedback is not a learned improvement policy. Multiple checkpoint sizes/precision/training sources can confound attribution to size alone.",
            "github": {
                key: os.environ.get(key)
                for key in ("GITHUB_RUN_ID", "GITHUB_SHA", "GITHUB_REPOSITORY")
            },
        }
        atomic_json(root / "discovery_plan.json", plan)
        summary.update(
            status="running",
            plan_sha256=file_sha256(root / "discovery_plan.json"),
            receipt_verified=True,
        )
        atomic_json(path, summary)
        from .action_schema import require_structured_decoder
        from .backend import ResearchCausalLM
        from .runner import run_lab

        if any(
            condition not in ("no_edit", "human_directed", "random_valid", "unconstrained")
            for condition in conditions
        ):
            require_structured_decoder()
        backend = ResearchCausalLM(model)
        if backend.base_model_files != receipt["files"]:
            raise RuntimeError(
                "Loaded model/tokenizer source differs from the preregistered verified receipt"
            )
        before = model_fingerprint(backend.model)
        atomic_json(
            root / "runtime_binding.json",
            {
                "state_sha256": before,
                "model": backend.describe(),
                "packages": {
                    key: metadata.version(key) for key in ("torch", "transformers", "safetensors")
                },
                "plan_sha256": summary["plan_sha256"],
            },
        )
        summary.update(
            model=backend.describe(), state_sha256=before, dataset_sha256=dataset.fingerprint()
        )
        cache, references = {}, {}
        for seed in seeds:
            for condition in conditions:
                _check_data_binding(args.data_dir, data_manifest, manifest_sha256)
                if code_bindings() != bindings:
                    raise RuntimeError(
                        "Source changed after preregistration; preserve this study and start a new one"
                    )
                print(f"discovery: seed={seed} condition={condition}", flush=True)
                is_model = condition in MODEL_CONDITIONS
                discovery = validate_discovery(
                    {
                        **discovery_base,
                        "gradient_access": condition
                        in ("human_directed", "model_directed_gradient", "schema_staged_gradient"),
                        "feedback": condition != "schema_no_feedback",
                        "prompt_style": "full" if condition == "schema_full_context" else "compact",
                        "operation_scope": "sparse" if condition == "schema_sparse_ops" else "all",
                        "workflow": "staged"
                        if condition in ("schema_staged", "schema_staged_gradient")
                        else "free",
                    },
                    dataset,
                    policy,
                )
                controller = (
                    None
                    if is_model
                    else NoEditControl()
                    if condition == "no_edit"
                    else HostControl(
                        target,
                        random_seed=seed + 700001 if condition == "random_valid" else None,
                        references=references.get(seed),
                    )
                )
                report = run_lab(
                    backend,
                    dataset,
                    replace(policy, seed=seed),
                    root / f"seed_{seed}" / condition,
                    sampling=Sampling(),
                    grader="exact",
                    editor={
                        "mode": "model" if is_model else "scripted",
                        "decoding": "json_schema"
                        if is_model and condition != "unconstrained"
                        else "unconstrained",
                        "max_new_tokens": 512,
                        "sampling": asdict(Sampling(temperature=0.3, top_p=0.95, top_k=20)),
                    },
                    actions=None,
                    controller=controller,
                    discovery=discovery,
                    measurement_cache=cache,
                    source=source,
                    provenance="Plan-bound closed-loop selected arithmetic/copy/reasoning/GSM8K study. Model actions never have a host-written fallback; host controls explicitly labelled. Receipt preserves precision/source differences.",
                    license_name=receipt["license_metadata"],
                )
                run = summarize_run(report, condition, seed)
                summary["runs"].append(run)
                if condition == "human_directed":
                    journal = json.loads(Path(run["journal_path"]).read_text())
                    references[seed] = random_references(journal, target)
                    atomic_json(root / f"seed_{seed}" / "random_reference.json", references[seed])
                if not report["base_restored"] or not report["source_files_unchanged"]:
                    raise RuntimeError("Restoration/source integrity did not pass")
                paired = {row["condition"]: row for row in summary["runs"] if row["seed"] == seed}
                if (
                    "unconstrained" in paired
                    and "schema" in paired
                    and paired["unconstrained"]["initial_prompt_sha256"]
                    != paired["schema"]["initial_prompt_sha256"]
                ):
                    raise RuntimeError(
                        "Unconstrained/schema initial prompt mismatch; not a matched syntax control"
                    )
                atomic_json(path, summary)
                github_evidence(summary, run)
                print(
                    json.dumps(
                        {
                            key: run[key]
                            for key in (
                                "condition",
                                "seed",
                                "status",
                                "valid_commands",
                                "valid_nonempty_edits",
                                "accepted",
                                "milestones",
                            )
                        }
                    ),
                    flush=True,
                )
        if code_bindings() != bindings or model_fingerprint(backend.model) != before:
            raise RuntimeError("End-of-study code/model binding mismatch")
        verify_model_files(model.model_name_or_path)
        _check_data_binding(args.data_dir, data_manifest, manifest_sha256)
        summary.update(
            status="completed",
            base_restored=True,
            source_files_unchanged=True,
            diagnosis=diagnose_results(summary["runs"]),
            cache_entries=len(cache),
            validated_autonomous_useful_edit=any(
                run["milestones"]["validated_autonomous_useful_edit"] for run in summary["runs"]
            ),
        )
    except BaseException as exc:
        summary.update(
            status="failed" if isinstance(exc, Exception) else "interrupted",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        atomic_json(path, summary)
    return {**summary, "summary_path": str(path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    result = run_experiment(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
