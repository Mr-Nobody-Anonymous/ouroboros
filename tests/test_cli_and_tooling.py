import contextlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from ouroboros.cli import load_run_config, main
from ouroboros.repositories import load_sources
from ouroboros.storage import audit_storage
from scripts.build_release import build
from scripts.check_docs import check
from scripts.register_artifact import register
from scripts.verify_artifacts import verify

ROOT = Path(__file__).resolve().parents[1]


def test_config_paths_resolve_relative_to_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    model, _, _, dataset = load_run_config(ROOT / "examples/local_sft.json")
    assert Path(model.model_name_or_path) == ROOT / "models/local-causal-lm"
    assert len(dataset.validation) == 4


def test_cli_doctor_check_only_and_failures():
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert main(["doctor"]) == 0
    assert json.loads(out.getvalue())["default_mode"].endswith("NOT an LLM")
    assert main(["run", "--config", str(ROOT / "examples/local_sft.json"), "--check-only"]) == 0
    assert main(["run", "--config", "/nonexistent-file.json"]) == 2
    assert main(["demo", "--generations", "0"]) == 2


def test_cli_objective_is_bounded():
    assert main(["objective", "loss * beta", "--loss", "2"]) == 0
    assert main(["objective", "__import__('os')"]) == 2
    assert main(["objective", "loss +"]) == 2
    assert main(["objective", "loss", "--loss", "1e300"]) == 2


def test_storage_audit_never_deletes_and_skips_secrets(tmp_path):
    (tmp_path / "keep.txt").write_text("keep")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv/private").write_text("do not scan")
    result = audit_storage(tmp_path, large_threshold=1)
    assert result["deleted"] == [] and result["scanned_files"] == 1
    assert (tmp_path / ".venv/private").read_text() == "do not scan"
    assert main(["storage", "audit", "--root", str(tmp_path), "--apply"]) == 2


def test_source_inventory_labels_reference_and_unknown_origins():
    sources = load_sources(ROOT)
    assert len(sources) == 12
    assert sum(s["mode"] == "submodule" for s in sources) == 7
    assert all(s["revision"] is None for s in sources if s["mode"] == "vendored")
    assert next(s for s in sources if s["name"] == "WSP-Scope")["integration"] == "paper-only"
    lock = json.loads((ROOT / "external/sources.lock.json").read_text())
    reviews = lock["optional_installed_package_reviews"]
    assert len(reviews) == 1 and reviews[0]["name"] == "lm-format-enforcer"
    for item in reviews:
        receipt = json.loads((ROOT / item["receipt"]).read_text())
        for field in ("version", "cloned_revision", "license"):
            assert item[field] == receipt[field]
        assert item["url"] == receipt["repository"]
        assert not receipt["vendored"] and not receipt["source_code_copied"]


def test_artifact_register_is_real_idempotent_and_not_upload(tmp_path):
    root = tmp_path / "repo"
    (root / "artifacts").mkdir(parents=True)
    (root / "artifacts/manifest.json").write_text('{"schema_version":1,"files":[]}')
    source = tmp_path / "fixture.dat"
    source.write_bytes(b"explicit unit test fixture")
    path = register(source, license_name="MIT", provenance="unit test", root=root)
    register(source, license_name="MIT", provenance="unit test", root=root)
    result = verify(root)
    assert result["registered_files"] == 1 and result["verified_bytes"] == source.stat().st_size
    (root / path).write_bytes(b"tampered")
    with pytest.raises(ValueError):
        verify(root)


@pytest.mark.parametrize("symlink_part", ["artifacts", "objects", "manifest"])
def test_artifact_registration_cannot_write_through_symlinks(tmp_path, symlink_part):
    root, outside = tmp_path / "repo", tmp_path / "outside"
    (root / "artifacts").mkdir(parents=True)
    outside.mkdir()
    manifest = root / "artifacts/manifest.json"
    manifest.write_text('{"schema_version":1,"files":[]}')
    source = tmp_path / "fixture.txt"
    source.write_text("test")
    if symlink_part == "artifacts":
        manifest.unlink()
        manifest.parent.rmdir()
        manifest.parent.symlink_to(outside, target_is_directory=True)
    elif symlink_part == "objects":
        (root / "artifacts/objects").symlink_to(outside, target_is_directory=True)
    else:
        manifest.unlink()
        (outside / "inventory.json").write_text('{"schema_version":1,"files":[]}')
        manifest.symlink_to(outside / "inventory.json")
    previous = {
        str(p.relative_to(outside)): p.read_bytes() for p in outside.rglob("*") if p.is_file()
    }
    with pytest.raises(ValueError, match="symlink"):
        register(source, license_name="MIT", provenance="unit test", root=root)
    assert {
        str(p.relative_to(outside)): p.read_bytes() for p in outside.rglob("*") if p.is_file()
    } == previous


def test_documentation_and_research_catalogue_are_consistent():
    assert check(ROOT)["research_references"] == 33
    catalogue = json.loads((ROOT / "papers/research.json").read_text())
    for paper in catalogue["papers"]:
        if "local_engineering_follow_up" in paper:
            follow_up = paper["local_engineering_follow_up"]
            assert (ROOT / follow_up["guide"]).is_file() and (
                ROOT / follow_up["evidence"]
            ).is_file()
            assert not follow_up["capability_gain_claimed"] and not paper["reproduced_here"]


def test_empty_manifest_is_not_fake_600mb():
    result = verify(ROOT)
    assert result["registered_files"] == 0 and result["verified_bytes"] == 0


def test_core_release_is_deterministic_and_has_honest_scope(tmp_path):
    if not (ROOT / ".git").exists():
        pytest.skip(
            "Release provenance builder requires a Git checkout, not an extracted source archive"
        )
    one = build(ROOT, tmp_path / "a", allow_dirty=True)
    two = build(ROOT, tmp_path / "b", allow_dirty=True)
    assert one["sha256"] == two["sha256"]
    with tarfile.open(one["archive"]) as archive:
        names = archive.getnames()
        assert any(name.endswith("RELEASE_CONTENTS.json") for name in names)
        assert not any("/external/peft/" in name or "/.venv/" in name for name in names)
        record = json.load(
            archive.extractfile(next(n for n in names if n.endswith("RELEASE_CONTENTS.json")))
        )
        assert "no upstream trees" in record["scope"]
