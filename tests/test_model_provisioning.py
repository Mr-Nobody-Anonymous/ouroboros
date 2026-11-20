"""Mocked explicit network boundary; CI tests never download models unless the smoke job opts in."""

import sys
from types import SimpleNamespace

import pytest

from ouroboros.model_files import fetch_model, verify_model_files


def fake_hub(monkeypatch, *, fail=False):
    payload = {
        "config.json": b"{}",
        "tokenizer.json": b"{}",
        "chat_template.jinja": b"template",
        "model.safetensors": b"not loaded in this receipt unit test",
        "pytorch_model.bin": b"never downloaded",
        "custom_model.py": b"never executed",
        "nested/unwanted.safetensors": b"not a supported root model shard",
    }
    calls = []
    info = SimpleNamespace(
        sha="b" * 40,
        siblings=[
            SimpleNamespace(rfilename=name, size=len(data)) for name, data in payload.items()
        ],
        card_data=SimpleNamespace(to_dict=lambda: {"license": "apache-2.0"}),
    )

    class HfApi:
        def model_info(self, repo_id, **kwargs):
            calls.append(("info", repo_id, kwargs))
            return info

    def snapshot_download(repo_id, *, revision, local_dir, allow_patterns):
        from pathlib import Path

        calls.append(("download", repo_id, revision, allow_patterns))
        for name in allow_patterns:
            (Path(local_dir) / name).write_bytes(payload[name])
        if fail:
            raise RuntimeError("simulated interrupted download")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=HfApi,
            snapshot_download=snapshot_download,
        ),
    )
    return info, calls


def test_explicit_download_pins_commit_records_hashes_and_excludes_executable_formats(
    monkeypatch, tmp_path
):
    _, calls = fake_hub(monkeypatch)
    target = tmp_path / "model"
    result = fetch_model("Owner/model", target, revision="selected-tag")
    assert result["verified"] and result["requested_revision"] == "selected-tag"
    assert calls[1][2] == "b" * 40
    assert set(calls[1][3]) == {
        "config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "model.safetensors",
    }
    assert verify_model_files(target)["files"] == result["files"]
    with pytest.raises(FileExistsError):
        fetch_model("Owner/model", target)
    assert len(calls) == 2  # Existing paths are refused before network access.


def test_byte_and_missing_size_gates_happen_before_download(monkeypatch, tmp_path):
    info, calls = fake_hub(monkeypatch)
    with pytest.raises(ValueError, match="byte budget"):
        fetch_model("Owner/model", tmp_path / "small", max_bytes=1)
    assert len(calls) == 1
    info.siblings[0].size = None
    with pytest.raises(ValueError, match="Unknown file sizes"):
        fetch_model("Owner/model", tmp_path / "unknown")
    assert all(call[0] != "download" for call in calls)


def test_download_failures_remove_only_the_owned_pending_directory(monkeypatch, tmp_path):
    fake_hub(monkeypatch, fail=True)
    keep = tmp_path / "keep.txt"
    keep.write_text("not owned")
    target = tmp_path / "model"
    with pytest.raises(RuntimeError, match="interrupted"):
        fetch_model("Owner/model", target)
    assert not target.exists() and keep.read_text() == "not owned"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]


@pytest.mark.parametrize(
    "repo", ["../evil", "https://hub.example/model", "owner/model/extra", "", None]
)
def test_invalid_repo_ids_never_start_network_access(monkeypatch, tmp_path, repo):
    _, calls = fake_hub(monkeypatch)
    with pytest.raises(ValueError):
        fetch_model(repo, tmp_path / "model")
    assert not calls


def test_github_notice_exports_measured_fields_without_paths_predictions_or_secrets(
    monkeypatch, tmp_path, capsys
):
    import json

    from scripts.run_pretrained_smoke import github_evidence

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    summary = {
        "model_parameters": 134_515_008,
        "base_restored": True,
        "source_files_unchanged": True,
        "baseline_macro_accuracy": 0.0,
        "candidate_macro_accuracies": [0.0, 0.0],
        "decisions": ["rejected", "rejected"],
        "changed_entries": [884, 884],
        "accepted_patches": 0,
        "report_path": "PRIVATE_PATH",
        "model_receipt": {"local": "PRIVATE_PATH"},
        "final_test": {"predictions": ["PRIVATE_PREDICTION"]},
    }
    report = {
        "base_state_sha256": "a" * 64,
        "restored_state_sha256": "a" * 64,
        "final_test": {"used_for_promotion": False},
    }
    github_evidence(summary, report)
    line = capsys.readouterr().out.strip()
    assert line.startswith("::notice title=Ouroboros pretrained smoke::")
    result = json.loads(line.split("::", 2)[2])
    assert result["base_restored"] and result["changed_entries"] == [884, 884]
    assert result["base_state_sha256"] == result["restored_state_sha256"]
    assert not result["final_test_used_for_promotion"]
    assert "PRIVATE_" not in line and "PRIVATE_" not in (tmp_path / "summary.md").read_text()
