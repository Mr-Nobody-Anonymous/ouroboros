"""Lightweight config/scaffold/provisioning checks; do not import Torch or access the network."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from ouroboros.cli import main
from ouroboros.core.checkpoints import file_sha256
from ouroboros.model_files import RECEIPT, model_file, verify_model_files
from ouroboros.project import init_project
from ouroboros.self_edit import (
    SelfEditPolicy,
    load_self_edit_config,
    preflight_model,
    scoped_path,
    self_edit_config_summary,
)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "lab"
    init_project(root)
    return root


def test_installed_scaffold_is_complete_non_destructive_and_offline(project, capsys):
    config = load_self_edit_config(project / "self_edit.json")
    assert self_edit_config_summary(config)["counts"] == {"train": 2, "validation": 4, "test": 2}
    assert config["model"].prompt_format == "chat"
    assert config["policy"].require_attempt
    with pytest.raises(FileExistsError):
        init_project(project)
    assert main(["self-edit", "--config", str(project / "self_edit.json"), "--check-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] and not report["model_loaded"] and not report["network_accessed"]
    assert "Paris" not in json.dumps(report) and "Rome" not in json.dumps(report)
    # The check-only CLI remains usable in the NumPy/safetensors-only installation.
    code = (
        "import sys; from ouroboros.cli import main; "
        "assert main(['self-edit','--config',sys.argv[1],'--check-only']) == 0; "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(project / "self_edit.json")],
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_rounds", 5),
        ("max_seconds", 0),
        ("cpu_threads", True),
        ("edited_fraction", 0.1),
        ("max_relative_norm", 0.5),
        ("forgetting_threshold", float("nan")),
        ("require_attempt", "true"),
        ("seed", -1),
        ("max_model_bytes", 0),
        ("max_matrix_elements", 100_000_000),
        ("max_model_parameters", 2_000_000_000),
    ],
)
def test_policy_rejects_unbounded_or_malformed_values(key, value):
    with pytest.raises(ValueError):
        SelfEditPolicy(**{key: value})


def test_config_split_contamination_and_bad_schemas_fail(project):
    validation = json.loads((project / "validation.json").read_text())
    (project / "test.json").write_text(json.dumps(validation[:1]))
    with pytest.raises(ValueError, match="contamination"):
        load_self_edit_config(project / "self_edit.json")
    path = project / "self_edit.json"
    values = json.loads(path.read_text())
    values["code"] = "not executed"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="exactly"):
        load_self_edit_config(path)


@pytest.mark.parametrize("value", ["../outside", ".env", "data/../../outside", "data/../inside"])
def test_scoped_paths_block_hidden_traversal_and_escapes(tmp_path, value):
    with pytest.raises(ValueError):
        scoped_path(tmp_path, value)


def test_scoped_config_checks_every_configured_path(project, tmp_path):
    path = project / "self_edit.json"
    values = json.loads(path.read_text())
    values["model"]["model_name_or_path"] = str(tmp_path / "outside")
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="escapes"):
        load_self_edit_config(path, workspace=project)
    (project / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes|Symlink"):
        scoped_path(project, "link/anything")
    # Symlinks wholly inside the workspace are also rejected.
    (project / "internal").symlink_to(project / "practice.json")
    with pytest.raises(ValueError, match="Symlink"):
        scoped_path(project, "internal")


def test_model_preflight_is_size_bounded_and_does_not_load_model(project):
    config = load_self_edit_config(project / "self_edit.json")
    directory = Path(config["model"].model_name_or_path)
    directory.mkdir(parents=True)
    save_file({"weight": np.ones((4, 4), dtype=np.float32)}, str(directory / "model.safetensors"))
    result = preflight_model(config)
    assert result["stored_tensor_elements"] == 16 and result["download_receipt"] is None
    config["policy"] = SelfEditPolicy(max_model_parameters=8)
    with pytest.raises(ValueError, match="parameter-element"):
        preflight_model(config)
    config["policy"] = SelfEditPolicy(max_model_bytes=4)
    with pytest.raises(ValueError, match="byte budget"):
        preflight_model(config)


def test_model_preflight_rejects_shard_traversal_and_symlinks(project):
    config = load_self_edit_config(project / "self_edit.json")
    directory = Path(config["model"].model_name_or_path)
    directory.mkdir(parents=True)
    save_file({"weight": np.ones((4, 4), dtype=np.float32)}, str(directory / "model.safetensors"))
    index = directory / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="Shard index"):
        preflight_model(config)
    index.unlink()
    (directory / "tokenizer.json").symlink_to(project / "practice.json")
    with pytest.raises(ValueError, match="symlinks"):
        preflight_model(config)


def _receipt(directory):
    names = [p.name for p in directory.iterdir() if model_file(p.name)]
    (directory / RECEIPT).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "resolved_revision": "a" * 40,
                "files": {name: file_sha256(directory / name) for name in names},
            }
        )
    )


def test_model_receipt_detects_tampering_and_inventory_changes(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"receipt-only-fixture")
    _receipt(tmp_path)
    assert verify_model_files(tmp_path)["verified"]
    (tmp_path / "config.json").write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="checksum"):
        verify_model_files(tmp_path)
    _receipt(tmp_path)
    (tmp_path / "chat_template.jinja").write_text("unrecorded tokenizer input")
    with pytest.raises(ValueError, match="inventory"):
        verify_model_files(tmp_path)


@pytest.mark.parametrize(
    "name", ["../x.safetensors", "..\\x.safetensors", ".env", "model.py", "pytorch_model.bin"]
)
def test_model_file_inventory_does_not_allow_paths_python_or_pickle(name):
    assert not model_file(name)


@pytest.mark.parametrize(
    "profile,extras", [("llm", "mcp,llm,smoke"), ("research", "mcp,llm,smoke,math")]
)
def test_install_plan_and_cli_missing_files_fail_without_side_effects(
    tmp_path, capsys, profile, extras
):
    script = Path(__file__).resolve().parents[1] / "scripts/install.py"
    target = tmp_path / "environment with spaces"
    output = subprocess.check_output(
        [
            sys.executable,
            str(script),
            "--profile",
            profile,
            "--venv",
            str(target),
            "--dry-run",
        ],
        text=True,
    )
    plan = json.loads(output)
    assert not target.exists() and not plan["downloads_model"]
    assert f"[{extras}]" in plan["install"][5]
    assert "-c" in plan["install"] and plan["tested_ml_constraints"]
    assert main(["models", "verify", "--directory", str(tmp_path)]) == 2
    assert "ouroboros:" in capsys.readouterr().err
