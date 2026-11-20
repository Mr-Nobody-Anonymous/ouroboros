"""Dependency-free artifact/evidence review, never model inference or network in tests."""

import base64
import gzip
import hashlib
import io
import json
import os
import zipfile

import pytest

from scripts import review_discovery_experiment as review


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def portable_study(tmp_path):
    root = tmp_path / "relocated-study"
    run_id = "ab" * 16
    directory = root / "seed_17/no_edit" / run_id
    event = save(
        directory / "session_01/turn_001.json",
        {"editor_generation": None, "raw_editor_reply": "PRIVATE HIDDEN MARKER"},
    )
    journal = save(
        directory / "session_01/edit_journal.json",
        {
            "experiments": [],
            "hypotheses": [],
            "analyses": [],
            "sensitivities": [],
            "loss_evaluated_problems": 1,
            "gradient_evaluated_problems": 0,
        },
    )
    retention = save(
        directory / "retention_measurement.json",
        {"status": "completed", "measurement": {"raw_hidden_prompt": "PRIVATE HIDDEN MARKER"}},
    )
    report = save(
        directory / "lab.json",
        {
            "status": "completed",
            "base_restored": True,
            "source_files_unchanged": True,
            "editor_state_sha256": "a" * 64,
            "starting_target_state_sha256": "a" * 64,
            "model_authored_actions": False,
            "sessions": [
                {
                    "edit_journal": {"sha256": journal},
                    "events": [
                        {
                            "artifact": "session_01/turn_001.json",
                            "artifact_sha256": event,
                            "turn": 1,
                            "status": "ok",
                        }
                    ],
                }
            ],
            "elapsed_seconds": 1,
            "frozen_candidates_before_final": {},
            "final_test": {"generated_sequences": 1},
            "retention": {"artifact_sha256": retention},
        },
    )
    plan = save(
        root / "discovery_plan.json",
        {
            "family": {"per_gate_alpha": 0.05 / 44},
            "conditions": ["no_edit"],
            "seeds": [17],
            "receipt": {"repo_id": "test-owned-placeholder-not-a-real-model", "verified": False},
            "data_manifest_sha256": "b" * 64,
            "code_sha256": {},
        },
    )
    save(
        root / "discovery_summary.json",
        {
            "kind": "ouroboros-closed-loop-discovery",
            "status": "completed",
            "profile": "135m",
            "plan_sha256": plan,
            "conditions": ["no_edit"],
            "seeds": [17],
            "state_sha256": "a" * 64,
            "dataset_sha256": "c" * 64,
            "model": {
                "backend": "test-owned-report-fixture-no-neural-model",
                "is_llm": False,
                "parameters": 4,
                "quantized": False,
                "generation_tokens": {},
            },
            "runs": [
                {
                    "seed": 17,
                    "condition": "no_edit",
                    "report_path": f"/nonexistent/original-machine/seed_17/no_edit/{run_id}/lab.json",
                    "report_sha256": report,
                    "journal_sha256": journal,
                    "model_authored_actions": False,
                    "valid_nonempty_edits": 0,
                }
            ],
            "validated_autonomous_useful_edit": False,
        },
    )
    return root


def test_portable_review_checks_original_hashes_without_paths_or_raw_hidden_exports(portable_study):
    summary, audits = review.review_study(portable_study)
    assert summary["completed_arm_reports"] == 1 and summary["integrity"]["whole_study_complete"]
    assert not summary["validated_autonomous_useful_edit"] and not summary["model"]["is_llm"]
    assert "PRIVATE HIDDEN MARKER" not in json.dumps({"summary": summary, "audits": audits})
    assert not summary["integrity"]["reviewer_reexecuted_model_or_tensor_replay"]
    event = next(portable_study.glob("seed_17/no_edit/*/session_01/turn_001.json"))
    event.write_text(event.read_text() + "\n")
    with pytest.raises(ValueError, match="hash changed"):
        review.review_study(portable_study)


def test_partial_review_is_explicit_and_never_writes_inside_a_registered_study(portable_study):
    path = portable_study / "discovery_summary.json"
    value = json.loads(path.read_text())
    value["status"] = "running"
    save(path, value)
    with pytest.raises(ValueError, match="incomplete"):
        review.review_study(portable_study)
    assert review.review_study(portable_study, allow_partial=True)[0]["study_status"] == "running"
    with pytest.raises(ValueError, match="inside the registered study"):
        review.main(["--study-dir", str(portable_study), "--workdir", str(portable_study / "bad")])
    assert not (portable_study / "bad").exists()


def test_numeric_review_refuses_incomplete_or_inconsistent_values():
    trial = {
        "roundtrip_replay": {"passed": True},
        "restored_to_resident_base": True,
        "original_checkpoint_sha256": "a" * 64,
        "changes": {
            "changed_entries": 1,
            "delta_norm": 0.25,
            "tensors": [
                {
                    "indices": [2],
                    "before_values": [0.25],
                    "after_values": [0.5],
                    "changed_entries": 1,
                    "delta_norm": 0.25,
                    "values_complete": True,
                }
            ],
        },
    }
    review.numeric_audit(trial, "a" * 64)
    trial["changes"]["tensors"][0]["after_values"] = [0.75]
    with pytest.raises(ValueError, match="norm disagrees"):
        review.numeric_audit(trial, "a" * 64)
    trial["changes"]["tensors"][0]["values_complete"] = False
    with pytest.raises(ValueError, match="Incomplete"):
        review.numeric_audit(trial, "a" * 64)


def test_strict_json_and_lossless_bounded_check_parts():
    for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
        with pytest.raises(ValueError):
            review.strict_json(text)
    value = {"test_owned_random_bytes_not_model_data": os.urandom(90_000).hex()}
    parts, sha = review.encoded_parts(value)
    assert len(parts) > 1 and max(map(len, parts)) <= review.PART_CHARACTERS
    payload = gzip.decompress(base64.b64decode("".join(parts)))
    assert hashlib.sha256(payload).hexdigest() == sha and json.loads(payload) == value


@pytest.mark.parametrize("name", ["../outside.json", "/absolute.json", "execute.py"])
def test_archive_download_checks_digest_and_refuses_unsafe_members(tmp_path, monkeypatch, name):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(name, "{}")
    payload = archive.getvalue()
    artifact = {
        "id": 123,
        "name": "discovery-360m-evidence-17",
        "expired": False,
        "size_in_bytes": len(payload),
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }

    def api(arguments, **kwargs):
        if "artifacts?" in arguments[0]:
            return {"artifacts": [artifact]}
        return {
            "id": 456,
            "head_sha": "d" * 40,
            "status": "completed",
            "conclusion": "success",
            "html_url": "https://github.com/test-owned/nonexistent/actions/runs/456",
        }

    def download(command, *, stdout, **kwargs):
        stdout.write(payload)

    monkeypatch.setattr(review, "gh_json", api)
    monkeypatch.setattr(review.subprocess, "run", download)
    with pytest.raises(ValueError, match="Unsafe"):
        review.download_study("test-owned/nonexistent", 456, "d" * 40, 17, "360m", tmp_path)
    assert not (tmp_path.parent / "outside.json").exists()


def test_check_publication_is_explicit_small_and_reconstructable(monkeypatch):
    payloads = []

    def api(arguments, *, payload):
        payloads.append(payload)
        return {
            "id": len(payloads),
            "name": payload["name"],
            "html_url": "https://github.com/test-owned/nonexistent",
        }

    monkeypatch.setattr(review, "gh_json", api)
    summary = {
        "planned_seeds": [17],
        "profile": "360m",
        "completed_arm_reports": 11,
        "model": {"parameters": 361821120},
        "runs": [{"baseline": {"accuracy": 0.0, "loss": {"problem_mean_nll": 1.0}}}],
        "validated_autonomous_useful_edit": False,
    }
    audits = {"explicit_test_fixture_data": os.urandom(70_000).hex()}
    transport = {
        "source_run_url": "https://github.com/test-owned/nonexistent",
        "artifact_id": 123,
        "source_run_id": 456,
    }
    result = review.publish_checks("test-owned/nonexistent", "d" * 40, summary, audits, transport)
    assert len(result) > 1 and all(p["head_sha"] == "d" * 40 for p in payloads)
    assert all(len(p["output"]["text"].encode()) < 60_000 for p in payloads)
    assert all("not a capability-success" in p["output"]["summary"] for p in payloads)
    envelopes = [json.loads(p["output"]["text"]) for p in payloads]
    decoded = gzip.decompress(base64.b64decode("".join(e["data"] for e in envelopes)))
    assert hashlib.sha256(decoded).hexdigest() == envelopes[0]["uncompressed_sha256"]
    assert json.loads(decoded) == {
        "summary": summary,
        "audits": audits,
        "github_transport": transport,
    }
