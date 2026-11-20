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
