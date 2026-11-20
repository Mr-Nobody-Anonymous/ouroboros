#!/usr/bin/env python3
"""Read-only, portable discovery artifact review; no ML, inference or candidate selection.

Use --study-dir for local artifacts, or explicitly --github-run for a completed GitHub
run. Optional --publish-checks exports reviewed aggregates/numerical audits through the
Checks API, not oversized Actions annotations. Raw hidden prompts/answers are NOT exported.
"""

import argparse
import base64
import gzip
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import zipfile
from pathlib import Path

MAX_FILE_BYTES = 32 * 1024**2
MAX_ARCHIVE_BYTES = 128 * 1024**2
PART_CHARACTERS = 30_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON: {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def read_json(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Unsafe or oversized review input")
    return strict_json(path.read_bytes())


def bound_json(root, path, expected):
    root, path = Path(root).resolve(), Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Artifact escapes its study directory")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("A complete SHA256 artifact binding is required")
    value = read_json(path)
    if digest(path.read_bytes()) != expected:
        raise ValueError(f"Artifact hash changed: {path.name}")
    return path, value


def numeric_audit(experiment, state_sha256):
    if (
        not experiment["roundtrip_replay"]["passed"]
        or not experiment["restored_to_resident_base"]
        or experiment["original_checkpoint_sha256"] != state_sha256
    ):
        raise ValueError("Original runner did not confirm exact replay/resident restoration")
    count, squared = 0, []
    for tensor in experiment["changes"]["tensors"]:
        indices, before, after = (
            tensor["indices"],
            tensor["before_values"],
            tensor["after_values"],
        )
        if (
            not tensor["values_complete"]
            or not len(indices) == len(before) == len(after) == tensor["changed_entries"]
            or any(type(i) is not int or i < 0 for i in indices)
            or sorted(set(indices)) != indices
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in before + after)
        ):
            raise ValueError("Incomplete or invalid actual numerical edit ledger")
        norm = math.sqrt(math.fsum((b - a) ** 2 for a, b in zip(before, after)))
        if not math.isclose(norm, tensor["delta_norm"], rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("Numerical delta norm disagrees with complete audited values")
        count += len(indices)
        squared.append(norm**2)
    if count != experiment["changes"]["changed_entries"] or not math.isclose(
        math.sqrt(math.fsum(squared)),
        experiment["changes"]["delta_norm"],
        rel_tol=1e-12,
        abs_tol=1e-14,
    ):
        raise ValueError("Combined numerical edit count/norm is inconsistent")


def followup_audit(retention, state_sha256):
    """Check both fixed-intervention ledgers, including any available refused-probe ledger."""
    if (
        retention["status"] not in ("completed", "refused")
        or not retention["base_restored"]
        or retention["original_checkpoint_sha256"] != state_sha256
    ):
        raise ValueError("Fixed follow-up integrity/start-state binding failed")
    for changes, replay in (
        ("subsequent_only_changes", "subsequent_only_roundtrip_replay"),
        ("changes", "roundtrip_replay"),
    ):
        if changes not in retention and retention["status"] == "refused":
            continue
        if changes not in retention or replay not in retention:
            raise ValueError("Completed fixed follow-up is missing its numerical ledger/replay")
        numeric_audit(
            {
                "changes": retention[changes],
                "roundtrip_replay": retention[replay],
                "restored_to_resident_base": retention["base_restored"],
                "original_checkpoint_sha256": retention["original_checkpoint_sha256"],
            },
            state_sha256,
        )


def review_study(directory, *, allow_partial=False):
    """Verify original hashes and numbers; never replay weights or re-evaluate hidden items."""
    root = Path(directory).resolve()
    summary = read_json(root / "discovery_summary.json")
    _, plan = bound_json(root, root / "discovery_plan.json", summary["plan_sha256"])
    if summary["kind"] != "ouroboros-closed-loop-discovery":
        raise ValueError("Expected a discovery experiment, not an arbitrary report")
    if not allow_partial and summary["status"] != "completed":
        raise ValueError("Study is incomplete; --allow-partial must be explicit")
    if (
        plan["family"]["per_gate_alpha"] != 0.05 / 44
        or plan["conditions"] != summary["conditions"]
        or plan["seeds"] != summary["seeds"]
    ):
        raise ValueError("Preregistered family/condition/seed binding disagrees")
    if summary["status"] == "completed" and len(summary["runs"]) != len(
        summary["conditions"]
    ) * len(summary["seeds"]):
        raise ValueError("Completed study is missing registered arm reports")
    rows, audits, references, seen = [], [], {}, set()
    for run in summary["runs"]:
        seed, condition = run["seed"], run["condition"]
        if (
            seed not in plan["seeds"]
            or condition not in plan["conditions"]
            or not re.fullmatch(r"[a-z_]+", condition)
            or (seed, condition) in seen
        ):
            raise ValueError("Unexpected/duplicate arm in the registered study")
        seen.add((seed, condition))
        # Original artifact JSON keeps its original absolute paths. Resolve by the fixed
        # protocol layout instead of rewriting hash-bound files when an archive is moved.
        original_report = Path(run["report_path"])
        run_id = original_report.parent.name
        if original_report.name != "lab.json" or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("Unexpected report layout")
        report_path, report = bound_json(
            root,
            root / f"seed_{seed}" / condition / run_id / "lab.json",
            run["report_sha256"],
        )
        journal_path, journal = bound_json(
            root, report_path.parent / "session_01/edit_journal.json", run["journal_sha256"]
        )
        if (
            report["status"] != "completed"
            or not report["base_restored"]
            or not report["source_files_unchanged"]
            or report["editor_state_sha256"] != summary["state_sha256"]
            or report["starting_target_state_sha256"] != summary["state_sha256"]
            or report["model_authored_actions"] != run["model_authored_actions"]
            or report["sessions"][0]["edit_journal"]["sha256"] != run["journal_sha256"]
        ):
            raise ValueError("Completed arm integrity/start-state/authorship binding failed")
        session = report["sessions"][0]
        row = {k: v for k, v in run.items() if k not in ("report_path", "journal_path")}
        row.update(
            report_artifact=str(report_path.relative_to(root)),
            journal_artifact=str(journal_path.relative_to(root)),
            elapsed_seconds=report["elapsed_seconds"],
            frozen_candidates_before_final=report["frozen_candidates_before_final"],
            final_generated_sequences=report["final_test"]["generated_sequences"],
            practice_loss_evaluated_problems=journal["loss_evaluated_problems"],
            practice_gradient_evaluated_problems=journal["gradient_evaluated_problems"],
            stop_reason=session.get("reason"),
            editor_trajectory=[],
        )
        for event in session["events"]:
            _, artifact = bound_json(
                root, report_path.parent / event["artifact"], event["artifact_sha256"]
            )
            generation = artifact.get("editor_generation")
            if generation is None:
                continue
            prompt = strict_json(artifact["editor_prompt"])
            row["editor_trajectory"].append(
                {
                    "turn": event["turn"],
                    "status": event["status"],
                    "action": (artifact.get("action") or {}).get("action"),
                    "prompt_tokens": generation["prompt_tokens"],
                    "reply_tokens": generation["new_tokens"],
                    "reply_cut": generation["cut"],
                    "visible_practice_examples": len(prompt["starting_diagnosis"]["practice"]),
                    "workflow": prompt["workflow"],
                    "gradient_access": prompt["gradient_access"],
                    "feedback_access": prompt["feedback_access"],
                    "raw_reply_sha256": digest(artifact["raw_editor_reply"].encode()),
                    "error": artifact["observation"].get("error"),
                }
            )
        trials = []
        for experiment in journal["experiments"]:
            numeric_audit(experiment, summary["state_sha256"])
            trials.append(
                {k: v for k, v in experiment.items() if k != "practice_evaluation"}
                | {"paired_practice": (experiment.get("practice_evaluation") or {}).get("paired")}
            )
        actual_edits = [
            trial
            for trial in trials
            if trial["operation"]["action"] in ("edit", "restore")
            and trial["actual_nonempty_change"]
            and trial["validation_result"]["passed"]
        ]
        if len(actual_edits) != run["valid_nonempty_edits"]:
            raise ValueError("Nonempty numerical edit count disagrees with study summary")
        if condition == "human_directed":
            references[seed] = actual_edits
        elif condition == "random_valid":
            reference = references.get(seed, [])
            if len(reference) != len(actual_edits):
                raise ValueError("A corresponding host trial is required for every random edit")
            row["matched_random_audit"] = []
            for human, random in zip(reference, actual_edits):
                a, b = human["changes"]["tensors"][0], random["changes"]["tensors"][0]
                row["matched_random_audit"].append(
                    {
                        "host_experiment": human["id"],
                        "random_experiment": random["id"],
                        "same_tensor": a["name"] == b["name"],
                        "same_actual_support": a["indices"] == b["indices"],
                        "host_actual_delta_norm": a["delta_norm"],
                        "random_actual_delta_norm": b["delta_norm"],
                        "relative_norm_discrepancy": abs(a["delta_norm"] - b["delta_norm"])
                        / a["delta_norm"],
                    }
                )
        retention_path, retention = bound_json(
            root,
            report_path.parent / "retention_measurement.json",
            report["retention"]["artifact_sha256"],
        )
        followup_audit(retention, summary["state_sha256"])
        rows.append(row)
        audits.append(
            {
                "condition": condition,
                "seed": seed,
                "journal_sha256": run["journal_sha256"],
                "hypotheses": journal["hypotheses"],
                "analyses": journal["analyses"],
                "sensitivities": journal["sensitivities"],
                "experiments": trials,
                "fixed_subsequent_edit": {k: v for k, v in retention.items() if k != "measurement"},
                "retention_artifact": str(retention_path.relative_to(root)),
                "retention_sha256": report["retention"]["artifact_sha256"],
            }
        )
    result = {
        "kind": "ouroboros-reviewed-closed-loop-discovery-study",
        "study_status": summary["status"],
        "profile": summary["profile"],
        "planned_arms": summary["conditions"],
        "planned_seeds": summary["seeds"],
        "completed_arm_reports": len(rows),
        "model": {
            k: summary["model"][k]
            for k in ("backend", "is_llm", "parameters", "quantized", "generation_tokens")
        },
        "source_receipt": {k: v for k, v in plan["receipt"].items() if k != "directory"},
        "state_sha256": summary["state_sha256"],
        "dataset_sha256": summary["dataset_sha256"],
        "plan_sha256": summary["plan_sha256"],
        "raw_summary_sha256": digest((root / "discovery_summary.json").read_bytes()),
        "data_manifest_sha256": plan["data_manifest_sha256"],
        "runtime_code_sha256": plan["code_sha256"],
        "family": plan["family"],
        "runs": rows,
        "diagnosis": summary.get("diagnosis"),
        "integrity": {
            "completed_reports_journals_and_events_hash_verified": True,
            "starting_state_equality_verified": True,
            "actual_stored_dtype_numerical_deltas_complete": True,
            "numerical_norms_independently_recomputed": True,
            "fixed_followup_available_numerical_norms_independently_recomputed": True,
            "original_runner_confirmed_tensor_replay_and_resident_restoration": True,
            "reviewer_reexecuted_model_or_tensor_replay": False,
            "whole_study_complete": summary["status"] == "completed",
        },
        "validated_autonomous_useful_edit": summary.get("validated_autonomous_useful_edit", False),
        "learned_future_self_improvement": False,
        "limitations": "Only completed arms reviewed. No inference/new evaluation/selection. A running/failed study is not complete. Strict exact accuracy/significance and zero-forgetting rules unchanged; practice/raw development gains do not pass them. HOST controls are not autonomous. Last-proposed diagnostic may have been rolled back; it never rescues failed gates. Greedy answers at repeated editor seeds are not independent benchmark replications. Small template-correlated evaluation, source/size/precision confounds and in-process boundaries remain. Numerical/hash review does not independently rerun a model or reconstruct unedited tensor entries.",
    }
    complete_audits = {
        "kind": "ouroboros-reviewed-complete-numerical-edit-audits",
        "profile": summary["profile"],
        "study_status": summary["status"],
        "plan_sha256": summary["plan_sha256"],
        "state_sha256": summary["state_sha256"],
        "source_receipt": result["source_receipt"],
        "scope": "Completed-arm attempted numerical interventions, actual before/after values, paired practice aggregates and fixed follow-up audits. No raw hidden prompts/answers, no substituted actions, no new model inference. Full original artifacts remain separate.",
        "runs": audits,
    }
    return result, complete_audits


def gh_json(arguments, *, payload=None):
    command = ["gh", "api", *arguments]
    if payload is not None:
        command += ["--input", "-"]
    completed = subprocess.run(
        command,
        input=json.dumps(payload) if payload is not None else None,
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )
    return strict_json(completed.stdout)


def download_study(repo, run_id, expected_head, seed, profile, workdir):
    source = gh_json([f"repos/{repo}/actions/runs/{run_id}"])
    if (
        source["head_sha"] != expected_head
        or source["status"] != "completed"
        or source["conclusion"] != "success"
    ):
        raise ValueError("Expected source run/head is not successfully complete")
    inventory = gh_json([f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100"])
    name = f"discovery-{profile}-evidence-{seed}"
    selected = [a for a in inventory["artifacts"] if a["name"] == name and not a["expired"]]
    if len(selected) != 1:
        raise ValueError("Exactly one nonexpired named source artifact is required")
    artifact = selected[0]
    if not 1 <= artifact["size_in_bytes"] <= MAX_ARCHIVE_BYTES or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", artifact.get("digest") or ""
    ):
        raise ValueError("Expected a bounded SHA256-bound artifact")
    archive_path = workdir / "source-artifact.zip"
    with archive_path.open("xb") as stream:
        subprocess.run(
            ["gh", "api", f"repos/{repo}/actions/artifacts/{artifact['id']}/zip"],
            stdout=stream,
            check=True,
            timeout=180,
        )
    if archive_path.stat().st_size != artifact["size_in_bytes"] or (
        "sha256:" + digest(archive_path.read_bytes()) != artifact["digest"]
    ):
        raise ValueError("Downloaded archive size/digest differs from GitHub's inventory")
    extract = workdir / "original-artifact"
    extract.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > 10_000 or sum(m.file_size for m in members) > MAX_ARCHIVE_BYTES:
            raise ValueError("Artifact expansion exceeds the fixed review budget")
        seen = set()
        for member in members:
            path = Path(member.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or path in seen
                or stat.S_ISLNK(member.external_attr >> 16)
                or member.file_size > MAX_FILE_BYTES
                or (not member.is_dir() and path.suffix != ".json")
                or member.flag_bits & 1
            ):
                raise ValueError("Unsafe/duplicate/non-JSON source archive member")
            seen.add(path)
            target = extract / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.is_dir():
                target.mkdir(exist_ok=True)
            else:
                target.write_bytes(archive.read(member))
    roots = list(extract.rglob("discovery_summary.json"))
    if len(roots) != 1:
        raise ValueError("Expected one study summary in the named artifact")
    return roots[0].parent, {
        "source_run_id": source["id"],
        "source_head_sha": source["head_sha"],
        "source_run_url": source["html_url"],
        "source_run_conclusion": source["conclusion"],
        "artifact_id": artifact["id"],
        "artifact_name": name,
        "artifact_bytes": artifact["size_in_bytes"],
        "artifact_sha256": artifact["digest"],
        "archive_payload_downloaded_and_digest_verified": True,
        "archive_safely_extracted_json_only": True,
        "model_rerun": False,
    }


def encoded_parts(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    encoded = base64.b64encode(gzip.compress(payload, mtime=0)).decode("ascii")
    return [
        encoded[i : i + PART_CHARACTERS] for i in range(0, len(encoded), PART_CHARACTERS)
    ], digest(payload)


def publish_checks(repo, head, reviewed, audits, transport):
    """Explicit trusted-branch output publication, not an Actions notice or model execution."""
    parts, sha = encoded_parts(
        {"summary": reviewed, "audits": audits, "github_transport": transport}
    )
    seed = reviewed["planned_seeds"]
    if len(seed) != 1:
        raise ValueError("Publish one fixed-seed archive per check set")
    baseline = reviewed["runs"][0]["baseline"]
    text = (
        "**Evidence-integrity review, not a capability-success check.** No model was rerun, "
        "no new item evaluated, no edit selected and no threshold changed.\n\n"
        f"Source: {transport['source_run_url']}; checkpoint parameters: {reviewed['model']['parameters']}; "
        f"completed arms: {reviewed['completed_arm_reports']}; seed: {seed[0]}. "
        f"Practice exact accuracy: {baseline['accuracy']}; problem NLL: {baseline['loss']['problem_mean_nll']}. "
        f"Validated autonomous useful edit: **{reviewed['validated_autonomous_useful_edit']}**.\n\n"
        "`output.text` holds a bounded part of gzip/base64 JSON containing the hash-verified "
        "aggregate results and complete numerical edit ledgers, **not raw hidden prompts/answers**. "
        "Concatenate parts in numeric order, base64-decode, gzip-decompress, then verify the "
        f"uncompressed SHA256 `{sha}` before reading it. Original source artifacts stay unchanged."
    )
    outputs = []
    for index, part in enumerate(parts):
        envelope = {
            "kind": "ouroboros-discovery-evidence-part",
            "encoding": "gzip+base64+utf8-json",
            "part_index": index,
            "part_count": len(parts),
            "uncompressed_sha256": sha,
            "source_run_id": transport["source_run_id"],
            "artifact_id": transport["artifact_id"],
            "seed": seed[0],
            "data": part,
        }
        payload = {
            "name": f"Ouroboros evidence {reviewed['profile']} seed {seed[0]} ({index + 1}/{len(parts)})",
            "head_sha": head,
            "details_url": transport["source_run_url"],
            "status": "completed",
            "conclusion": "success",
            "external_id": f"ouroboros-evidence:{transport['artifact_id']}:{index}",
            "output": {
                "title": "Artifact/numerical integrity only; inspect the scientific milestone separately",
                "summary": text,
                "text": json.dumps(envelope, sort_keys=True, separators=(",", ":")),
            },
        }
        if len(payload["output"]["text"].encode()) > 60_000:
            raise ValueError("Bounded Checks API export exceeded its explicit size")
        result = gh_json(["--method", "POST", f"repos/{repo}/check-runs"], payload=payload)
        outputs.append({k: result[k] for k in ("id", "name", "html_url")})
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--study-dir")
    source.add_argument("--github-run", type=int)
    parser.add_argument("--expected-head")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--seed", type=int, choices=(17, 42))
    parser.add_argument("--profile", choices=("135m", "360m"), default="360m")
    parser.add_argument("--workdir", required=True, help="NEW review output directory")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--publish-checks", action="store_true")
    parser.add_argument("--head-sha", default=os.environ.get("GITHUB_SHA"))
    args = parser.parse_args(argv)
    if args.publish_checks and (not args.github_run or not args.head_sha or args.allow_partial):
        raise ValueError("Publishing requires a complete explicitly bound GitHub source and head")
    if args.github_run and (
        not args.repo
        or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo)
        or not re.fullmatch(r"[0-9a-f]{40}", args.expected_head or "")
        or args.seed is None
    ):
        raise ValueError("GitHub review needs --repo, --seed and an exact --expected-head")
    if args.publish_checks and not re.fullmatch(r"[0-9a-f]{40}", args.head_sha):
        raise ValueError("Publishing needs an exact target commit")
    workdir = Path(args.workdir).resolve()
    if args.study_dir and workdir.is_relative_to(Path(args.study_dir).resolve()):
        raise ValueError("Review output must not be written inside the registered study")
    workdir.mkdir(parents=True, exist_ok=False)
    transport = None
    directory = args.study_dir
    if args.github_run:
        directory, transport = download_study(
            args.repo,
            args.github_run,
            args.expected_head,
            args.seed,
            args.profile,
            workdir,
        )
    reviewed, audits = review_study(directory, allow_partial=args.allow_partial)
    if args.github_run and (
        reviewed["profile"] != args.profile or reviewed["planned_seeds"] != [args.seed]
    ):
        raise ValueError("Downloaded study profile/seed differs from its named artifact")
    for name, value in (("reviewed_summary.json", reviewed), ("numerical_audits.json", audits)):
        (workdir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    result = {
        "study_status": reviewed["study_status"],
        "completed_arm_reports": reviewed["completed_arm_reports"],
        "workdir": str(workdir),
        "validated_autonomous_useful_edit": reviewed["validated_autonomous_useful_edit"],
        "model_rerun": False,
        "github_transport": transport,
    }
    if args.publish_checks:
        result["checks"] = publish_checks(args.repo, args.head_sha, reviewed, audits, transport)
    (workdir / "review_receipt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
