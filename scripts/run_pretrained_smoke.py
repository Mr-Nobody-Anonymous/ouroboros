"""Explicit end-to-end pretrained smoke. Tests mechanics, not a required capability gain.

Uses the pinned GitHub Q4_1 distribution, converted to float32. Ordinary pytest stays offline.
Use a NEW workdir. Without --fetch an already-provisioned --model-dir is required.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def cli(*args):
    completed = subprocess.run(
        [sys.executable, "-m", "ouroboros.cli", *map(str, args)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    )
    return json.loads(completed.stdout)


def github_evidence(summary, report):
    """Expose small measured results via GitHub annotations/API when artifact hosts are blocked.

    No predictions, paths or environment secrets are included. Stdout notices and the job summary
    are emitted only in Actions, after main has verified the actual pretrained/restoration contract.
    """
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    keys = (
        "model_parameters",
        "base_restored",
        "source_files_unchanged",
        "baseline_macro_accuracy",
        "candidate_macro_accuracies",
        "decisions",
        "changed_entries",
        "accepted_patches",
    )
    evidence = {key: summary[key] for key in keys}
    evidence.update(
        kind="pretrained-mechanics-smoke",
        base_state_sha256=report["base_state_sha256"],
        restored_state_sha256=report["restored_state_sha256"],
        final_test_used_for_promotion=report["final_test"]["used_for_promotion"],
    )
    encoded = json.dumps(evidence, sort_keys=True, allow_nan=False)
    print("::notice title=Ouroboros pretrained smoke::" + encoded)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
            stream.write(
                "## Ouroboros pretrained mechanics smoke\n\n```json\n" + encoded + "\n```\n\n"
                "A rejection is a valid measured result, not a capability gain. Full reports are in the JSON artifact; no model weights are uploaded.\n"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default="outputs/pretrained-smoke")
    parser.add_argument("--model-dir", help="Reuse a model provisioned by models fetch-smoke")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Explicitly allow the pinned GitHub model-data download",
    )
    args = parser.parse_args(argv)
    if not args.model_dir and not args.fetch:
        parser.error("Supply an existing --model-dir, or explicitly enable --fetch")
    root = Path(args.workdir).expanduser().resolve()
    cli("init", root)  # Refuses overwrite, even for a previous failed smoke run.
    model = (
        Path(args.model_dir).expanduser().resolve() if args.model_dir else root / "models/smollm2"
    )
    if not model.exists():
        if not args.fetch:
            parser.error("Model does not exist; no download without --fetch")
        receipt = cli("models", "fetch-smoke", "--directory", model)
    else:
        receipt = cli("models", "verify", "--directory", model)
    if (
        receipt.get("source", {}).get("sha256")
        != "b179c9523d0e6a0f98a330c7562b682750a6f8c8c15e5bc70ea373728110db53"
    ):
        raise ValueError("This smoke script requires the pinned Q4_1-derived model receipt")
    config_path = root / "self_edit.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["model_name_or_path"] = str(model)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    cli("self-edit", "--config", config_path, "--check-only")
    report = cli("self-edit", "--config", config_path)
    if (
        report.get("status") != "completed"
        or not report.get("base_restored")
        or not report.get("source_files_unchanged")
        or not report.get("model", {}).get("is_llm")
        or report.get("model", {}).get("parameters") != 134_515_008
        or len(report.get("rounds", [])) != 2
        or report.get("final_test", {}).get("status") != "completed"
        or report["final_test"].get("used_for_promotion") is not False
        or report["base_state_sha256"] != report["restored_state_sha256"]
    ):
        raise RuntimeError(
            "Pretrained smoke control-flow/restoration contract failed; inspect the run report"
        )
    for row in report["rounds"]:
        trial = row["trial"]
        if (
            not row["proposal"].get("proposer_is_target_model")
            or not trial.get("model_restored")
            or not trial.get("edit", {}).get("edited_weights", 0)
        ):
            raise RuntimeError("A same-model, nonzero, restored edit trial was not observed")
    summary = {
        "kind": "pretrained-mechanics-smoke",
        "model_parameters": report["model"]["parameters"],
        "model_receipt": receipt,
        "report_path": report["report_path"],
        "base_restored": report["base_restored"],
        "source_files_unchanged": report["source_files_unchanged"],
        "baseline_macro_accuracy": report["baseline"]["macro_accuracy"],
        "candidate_macro_accuracies": [
            r["trial"]["candidate"]["macro_accuracy"] if r["trial"]["candidate"] else None
            for r in report["rounds"]
        ],
        "decisions": [r["trial"]["status"] for r in report["rounds"]],
        "changed_entries": [r["trial"]["edit"]["edited_weights"] for r in report["rounds"]],
        "accepted_patches": len(report["accepted_chain"]),
        "final_test": report["final_test"],
        "scope": "Pretrained weight-edit mechanics, NOT a benchmark reproduction or required capability gain.",
    }
    (root / "pretrained_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    github_evidence(summary, report)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
