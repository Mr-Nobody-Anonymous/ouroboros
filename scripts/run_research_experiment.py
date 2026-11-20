"""Fixed small-model command-interface experiment, not a capability benchmark.

Requires an existing receipt-verified local model; no download and no host-authored action
fallback. Optionally compares unconstrained/schema decoding with identical model, prompts,
answer/editor budgets, seeds, data, sampling and host policy. Dynamic wall-clock values never enter editor context; the initial prompt hashes must match. Both conditions and their order
are written before either runs. Final results do not choose a condition or another edit.
"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ouroboros.core.checkpoints import atomic_json, file_sha256
from ouroboros.lab.action_schema import ACTION_SCHEMA_VERSION, require_structured_decoder
from ouroboros.lab.backend import ResearchCausalLM
from ouroboros.lab.config import load_lab_config
from ouroboros.lab.runner import run_lab
from ouroboros.lab.tools import TOOLS
from ouroboros.model_files import verify_model_files
from ouroboros.project import init_project
from ouroboros.self_edit import preflight_model


def condition_summary(report):
    """Aggregate measurements only: never copy prompts, replies, held-out rows or host paths."""
    sessions = report["sessions"]
    events = [event for session in sessions for event in session["events"]]
    staged_actions = changed = 0
    for event in events:
        if event["action"] == "edit" and event["status"] == "ok":
            artifact = Path(report["report_path"]).parent / event["artifact"]
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            staged_actions += 1
            changed = max(changed, payload["observation"]["changed_entries"])
    return {
        "decoding": report["editor_decoding"],
        "model_authored_actions": report["model_authored_actions"],
        "editor_generations": sum(session["editor_generations"] for session in sessions),
        "parsed_json_replies": sum(event["parsed_json"] for event in events),
        "valid_tool_actions": sum(event["status"] == "ok" for event in events),
        "invalid_actions": sum(session["counters"]["invalid_actions"] for session in sessions),
        "valid_action_names": sorted(
            {
                event["action"]
                for event in events
                if event["status"] == "ok" and event["action"] in TOOLS
            }
        ),
        "staged_edit_actions": staged_actions,
        "max_staged_changed_entries": changed,
        "practice_test_actions": sum(
            event["status"] == "ok" and event["action"] in ("test", "control", "sweep")
            for event in events
        ),
        "status": sessions[-1]["status"],
        "accepted": report["selected_session"] is not None,
        "practice_baseline_accuracy": sessions[0]["practice_baseline"]["accuracy"],
        "practice_cut_share": sessions[0]["practice_baseline"]["cut_share"],
        "practice_signal": sessions[0]["practice_readiness"]["signal"],
        "base_restored": report["base_restored"],
        "source_files_unchanged": report["source_files_unchanged"],
        "final_test_used_for_promotion": report["final_test"]["used_for_promotion"],
    }


def emit_evidence(summary):
    # Deliberate allowlist. Do not serialize a full report or an arbitrary summary to CI notices.
    safe = {
        key: summary[key]
        for key in (
            "kind",
            "parameters",
            "editor_state_sha256",
            "dataset_sha256",
            "conditions",
            "base_restored",
            "source_files_unchanged",
            "capability_gain_claimed",
        )
    }
    for key in ("initial_editor_prompt_sha256", "matched_initial_editor_prompt"):
        if key in summary:
            safe[key] = summary[key]
    fields = {
        "decoding",
        "model_authored_actions",
        "editor_generations",
        "parsed_json_replies",
        "valid_tool_actions",
        "invalid_actions",
        "valid_action_names",
        "staged_edit_actions",
        "max_staged_changed_entries",
        "practice_test_actions",
        "status",
        "accepted",
        "practice_baseline_accuracy",
        "practice_cut_share",
        "practice_signal",
        "base_restored",
        "source_files_unchanged",
        "final_test_used_for_promotion",
    }
    safe["conditions"] = [
        {key: row[key] for key in sorted(fields)} for row in summary["conditions"]
    ]
    for row in safe["conditions"]:
        if row["decoding"] not in ("unconstrained", "json_schema") or row["status"] not in {
            "accepted",
            "rejected",
            "withdrawn",
            "budget_exhausted",
            "stalled",
        }:
            raise ValueError("Unexpected experiment evidence condition")
        if any(name not in TOOLS for name in row["valid_action_names"]):
            raise ValueError("Unexpected experiment action name")
        if row["practice_signal"] not in (
            "all_correct",
            "all_incorrect",
            "mixed_correct_and_incorrect",
        ):
            raise ValueError("Unexpected practice signal")
        if any(type(value) not in (str, int, float, bool, list) for value in row.values()):
            raise ValueError("Unexpected experiment evidence type")
    text = json.dumps(safe, separators=(",", ":"), allow_nan=False)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print("::notice title=Ouroboros structured editor evidence::" + text)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
                stream.write(
                    "\n### Structured research editor (fixed arithmetic/copy fixture, NOT a benchmark)\n\n```json\n"
                    + json.dumps(safe, indent=2)
                    + "\n```\n\nGrammar constrains syntax, not edit quality. No host-written action fallback. Strict gates unchanged.\n"
                )
    return safe


def implementation_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [
        *root.glob("ouroboros/lab/*.py"),
        *root.glob("ouroboros/backends/*.py"),
        *root.glob("ouroboros/resources/*.json"),
        root / "ouroboros/core/weight_trials.py",
        root / "ouroboros/self_edit.py",
        root / "ouroboros/project.py",
        root / "ouroboros/model_files.py",
        root / "ouroboros/config.py",
        root / "ouroboros/data.py",
        Path(__file__).resolve(),
    ]
    return {str(path.relative_to(root)): file_sha256(path) for path in sorted(set(paths))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, help="New directory only")
    parser.add_argument("--model-dir", required=True, help="Existing receipt-verified model")
    parser.add_argument("--compare-unconstrained", action="store_true")
    parser.add_argument("--turns", type=int, default=16)
    parser.add_argument("--max-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    if not 1 <= args.turns <= 160 or not 1 <= args.max_seconds <= 3600:
        parser.error("turns must be 1-160; max-seconds must be 1-3600 per condition")
    require_structured_decoder()
    model_path, root = Path(args.model_dir).resolve(), Path(args.workdir).resolve()
    receipt = verify_model_files(model_path)
    init_project(root)
    path = root / "research_lab.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    values["model"]["model_name_or_path"] = str(model_path)
    values["policy"].update(max_turns=args.turns, max_seconds=args.max_seconds)
    atomic_json(path, values)
    config = load_lab_config(path)
    source = preflight_model(config)
    conditions = (["unconstrained"] if args.compare_unconstrained else []) + ["json_schema"]
    plan = {
        "kind": "ouroboros-fixed-interface-comparison-plan",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": config["dataset"].fingerprint(),
        "config_sha256": file_sha256(path),
        "conditions_in_order": conditions,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "implementation_files_sha256": implementation_hashes(),
        "only_condition_difference": "editor.decoding; identical initial editor prompt verified by hash, with dynamic wall-clock values excluded. Later histories may differ as model actions/observations differ. Grammar filters tokens, not host-written edit choices.",
        "answer_tokens": config["model"].max_new_tokens,
        "editor": config["editor"],
        "sampling": asdict(config["sampling"]),
        "policy": asdict(config["policy"]),
        "selection": "No condition chosen from final results; no cross-condition feedback, action substitution or target/grader/budget changes. Final data are not returned to either editor. Repeated public fixtures are engineering evidence only.",
        "capability_gain_claimed": False,
    }
    plan_path = root / "interface_plan.json"
    atomic_json(plan_path, plan)
    backend = ResearchCausalLM(config["model"])
    reports, first_prompts = [], []
    for decoding in conditions:
        editor = {**config["editor"], "decoding": decoding}
        report = run_lab(
            backend,
            config["dataset"],
            config["policy"],
            root / "outputs" / decoding,
            sampling=config["sampling"],
            editor=editor,
            grader=config["grader"],
            provenance=config["provenance"],
            license_name=config["license_name"],
            source=source,
        )
        reports.append(report)
        events = report["sessions"][0]["events"]
        if events:
            artifact = Path(report["report_path"]).parent / events[0]["artifact"]
            prompt = json.loads(artifact.read_text(encoding="utf-8"))["editor_prompt"]
            first_prompts.append(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        else:
            first_prompts.append(None)
        atomic_json(
            root / (decoding + "_report_location.json"), {"report_path": report["report_path"]}
        )
    if not all(report["base_restored"] and report["source_files_unchanged"] for report in reports):
        raise RuntimeError("Experiment did not confirm full restoration/source integrity")
    if any(
        report["sessions"][0]["practice_baseline"] != reports[0]["sessions"][0]["practice_baseline"]
        or report["editor_state_sha256"] != reports[0]["editor_state_sha256"]
        for report in reports
    ):
        raise RuntimeError("Fixed-condition model/baseline equality failed")
    if len(reports) > 1 and (first_prompts[0] is None or len(set(first_prompts)) != 1):
        raise RuntimeError(
            "Comparison initial editor prompts did not match exactly; do not attribute results only to decoding"
        )
    if implementation_hashes() != plan["implementation_files_sha256"]:
        raise RuntimeError("Experiment implementation/resource files changed during the run")
    summary = {
        "kind": "ouroboros-research-interface-experiment",
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "parameters": backend.describe()["parameters"],
        "editor_state_sha256": reports[0]["editor_state_sha256"],
        "dataset_sha256": config["dataset"].fingerprint(),
        "plan_sha256": file_sha256(plan_path),
        "initial_editor_prompt_sha256": first_prompts[0],
        "matched_initial_editor_prompt": len(reports) > 1 and len(set(first_prompts)) == 1,
        "model_receipt_verified": receipt["verified"],
        "conditions": [condition_summary(report) for report in reports],
        "base_restored": True,
        "source_files_unchanged": True,
        "capability_gain_claimed": False,
    }
    atomic_json(root / "experiment_summary.json", summary)
    emit_evidence(summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
