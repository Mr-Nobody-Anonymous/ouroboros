"""Explicit pretrained research smoke: actual model-authored JSON attempts + labelled tool script.

Requires an existing receipt-verified local model. Never downloads or executes generated Python.
The synthetic contrast texts/public few-example tasks are plumbing fixtures, not a benchmark.
"""

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from ouroboros.config import ModelConfig
from ouroboros.core.checkpoints import atomic_json
from ouroboros.lab.backend import ResearchCausalLM
from ouroboros.lab.config import LabPolicy, Sampling, load_lab_config
from ouroboros.lab.runner import run_lab
from ouroboros.model_files import verify_model_files
from ouroboros.project import init_project
from ouroboros.self_edit import preflight_model


def github_evidence(summary):
    """Whitelisted compact measurements; never emit paths, prompts, predictions or raw actions."""
    safe = {
        key: summary[key]
        for key in (
            "parameters",
            "editor_state_sha256",
            "model_authored",
            "scripted_tools",
            "base_restored",
            "source_files_unchanged",
            "capability_gain_claimed",
        )
    }
    for section, fields in {
        "model_authored": (
            "editor_generations",
            "valid_tool_actions",
            "invalid_actions",
            "status",
            "accepted",
        ),
        "scripted_tools": (
            "actions",
            "forward_requests",
            "changed_entries",
            "edited_tensors",
            "status",
            "accepted",
            "reserved_baseline_accuracy",
            "reserved_candidate_accuracy",
            "max_control_norm_rounding_error",
        ),
    }.items():
        safe[section] = {key: summary[section][key] for key in fields}
        if safe[section]["status"] not in {
            "accepted",
            "rejected",
            "withdrawn",
            "budget_exhausted",
            "script_exhausted",
        }:
            raise ValueError("Unexpected research evidence status")
        if any(type(value) not in (int, float, bool, str) for value in safe[section].values()):
            raise ValueError("Research evidence must contain scalar measurements")
    message = json.dumps(safe, separators=(",", ":"), allow_nan=False)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print("::notice title=Ouroboros research evidence::" + message)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
                stream.write(
                    "\n### Pretrained research-tool smoke\n\n```json\n"
                    + json.dumps(safe, indent=2)
                    + "\n```\n\nModel-authored attempts and scripted integration are distinct. No benchmark gain is claimed.\n"
                )
    return safe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, help="New directory only")
    parser.add_argument(
        "--model-dir", required=True, help="Existing receipt-verified safetensors model"
    )
    args = parser.parse_args(argv)
    root, model_path = Path(args.workdir).resolve(), Path(args.model_dir).resolve()
    receipt = verify_model_files(model_path)
    init_project(root)
    config = load_lab_config(root / "research_lab.json")
    model = ModelConfig(model_name_or_path=str(model_path), prompt_format="chat", max_new_tokens=12)
    policy = LabPolicy(
        max_turns=24,
        max_forward_calls=12,
        max_edit_actions=8,
        max_seconds=600,
        max_edit_tensors=4,
        max_read_tokens=96,
        cpu_threads=2,
    )
    source = preflight_model({"model": model, "policy": policy})
    backend = ResearchCausalLM(model)
    candidates = [
        (name, tensor)
        for name, tensor in backend.model.named_parameters()
        if name.endswith("mlp.down_proj.weight")
        and tensor.ndim == 2
        and tensor.numel() <= policy.max_matrix_elements
    ]
    if len(candidates) < 2:
        raise ValueError("Research smoke needs at least two supported MLP output matrices")
    names = [name for name, _ in candidates[-2:]]
    layer_module = names[0].rsplit(".mlp.", 1)[0]
    module = names[0].removesuffix(".weight")
    provenance = "Explicit pretrained research smoke on receipt-verified local data. Model-authored JSON attempts are separate from scripted integration. Synthetic text contrast and public few-example exact-match fixtures, NOT benchmark evidence."
    actual_policy = replace(
        policy, max_turns=2, max_invalid_actions=2, max_forward_calls=2, max_seconds=240
    )
    actual = run_lab(
        backend,
        config["dataset"],
        actual_policy,
        root / "outputs/model-authored",
        sampling=Sampling(),
        editor={"mode": "model", "max_new_tokens": 96},
        provenance=provenance,
        license_name="Apache-2.0; see model receipt",
        source=source,
    )

    def projections(strength):
        return [
            {
                "op": "project",
                "name": name,
                "direction": "d0",
                "strength": strength,
                "top_columns": 1,
            }
            for name in names
        ]

    actions = [
        {"action": "inventory", "limit": 4},
        {"action": "weights", "names": names, "values": 4},
        {
            "action": "read",
            "kind": "hidden",
            "texts": ["One plus one is two.", "One plus one is one."],
            "modules": [layer_module],
            "last": 0,
        },
        {
            "action": "read",
            "kind": "mlp_input",
            "texts": ["One plus one is two."],
            "modules": [module],
            "last": 0,
        },
        {"action": "read", "kind": "logprobs", "texts": ["One plus one is two."], "last": 0},
        {
            "action": "contrast",
            "module": layer_module,
            "failing": ["The answer is one. One one one one one."],
            "successful": ["The answer is two."],
            "last": 0,
        },
        {"action": "writers", "name": names[0], "direction": "d0", "limit": 4},
        {"action": "edit", "operations": projections(0.05)},
        {"action": "save", "name": "small"},
        {"action": "test", "ids": [0, 1], "saved": ["small"]},
        {"action": "control", "kind": "random_direction", "ids": [0, 1], "seed": 43},
        {"action": "control", "kind": "no_op", "ids": [0, 1]},
        {"action": "control", "kind": "layer_isolation", "names": [names[0]], "ids": [0, 1]},
        {
            "action": "sweep",
            "ids": [0, 1],
            "variants": [
                {"label": "weak", "operations": projections(0.01)},
                {"label": "stronger", "operations": projections(0.1)},
            ],
        },
        {"action": "submit"},
        {"action": "confirm"},
    ]
    atomic_json(root / "tool_actions.json", actions)
    raw_config = json.loads((root / "research_lab.json").read_text())
    raw_config.update(
        model=asdict(model),
        policy=asdict(policy),
        editor={"mode": "scripted", "actions": "tool_actions.json", "max_new_tokens": 96},
        output_dir="outputs/scripted-tools",
        provenance=provenance,
    )
    atomic_json(root / "scripted_lab.json", raw_config)
    raw_config.update(
        policy=asdict(actual_policy),
        editor={"mode": "model", "max_new_tokens": 96},
        output_dir="outputs/model-authored",
        provenance=provenance,
    )
    atomic_json(root / "model_lab.json", raw_config)
    scripted = run_lab(
        backend,
        config["dataset"],
        policy,
        root / "outputs/scripted-tools",
        sampling=Sampling(),
        editor={"mode": "scripted", "max_new_tokens": 96},
        actions=actions,
        provenance=provenance,
        license_name="Apache-2.0; see model receipt",
        source=source,
    )
    record = scripted["sessions"][0]
    if len(record["events"]) != len(actions) or any(
        event["status"] != "ok" for event in record["events"]
    ):
        raise RuntimeError("A scripted integration tool failed; inspect archived observations")
    run_dir = Path(scripted["report_path"]).parent
    observations = [
        json.loads((run_dir / event["artifact"]).read_text()) for event in record["events"]
    ]
    changed = next(
        row["observation"]["changed_entries"]
        for row in observations
        if row["action"]["action"] == "edit"
    )
    matches = next(
        row["observation"]["control"]["norm_matches"]
        for row in observations
        if row["action"].get("kind") == "random_direction"
    )
    if not changed or any(item["relative_rounding_error"] > 0.001 for item in matches):
        raise RuntimeError("Nonzero edit/norm-matched control contract failed")
    if not all(
        report["base_restored"] and report["source_files_unchanged"]
        for report in (actual, scripted)
    ):
        raise RuntimeError("Research smoke did not confirm restoration/source integrity")
    authored = actual["sessions"][0]
    summary = {
        "kind": "ouroboros-pretrained-research-smoke",
        "parameters": backend.describe()["parameters"],
        "editor_state_sha256": actual["editor_state_sha256"],
        "model_authored": {
            "editor_generations": authored["editor_generations"],
            "valid_tool_actions": sum(event["status"] == "ok" for event in authored["events"]),
            "invalid_actions": authored["counters"]["invalid_actions"],
            "status": authored["status"],
            "accepted": authored["status"] == "accepted",
        },
        "scripted_tools": {
            "actions": len(record["events"]),
            "forward_requests": record["counters"]["forward_calls"],
            "changed_entries": changed,
            "edited_tensors": len(names),
            "status": record["status"],
            "accepted": record["status"] == "accepted",
            "reserved_baseline_accuracy": record["reserved_check"]["paired"]["baseline"][
                "accuracy"
            ],
            "reserved_candidate_accuracy": record["reserved_check"]["paired"]["candidate"][
                "accuracy"
            ],
            "max_control_norm_rounding_error": max(
                item["relative_rounding_error"] for item in matches
            ),
        },
        "base_restored": True,
        "source_files_unchanged": True,
        "capability_gain_claimed": False,
        "actual_report": actual["report_path"],
        "scripted_report": scripted["report_path"],
        "model_receipt_verified": receipt["verified"],
    }
    atomic_json(root / "research_summary.json", summary)
    github_evidence(summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
