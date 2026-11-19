import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from ouroboros.cli import load_weight_trial_config, main
from ouroboros.core.weight_trials import MAX_PATCH_BYTES, WeightEditPlan, verify_weight_patch
from ouroboros.tensor_inspection import MAX_HEADER_BYTES, inspect_safetensors

ROOT = Path(__file__).resolve().parents[1]


def raw_file(path, entries, data):
    header = json.dumps(entries, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(len(header).to_bytes(8, "little") + header + data)


def test_safetensors_metadata_and_samples_do_not_require_torch(tmp_path, capsys):
    path = tmp_path / "numbers.safetensors"
    matrix = np.arange(24, dtype=np.float32).reshape(4, 6)
    save_file({"matrix": matrix, "scalar": np.array(7, dtype=np.int64)}, str(path))
    original = path.read_bytes()
    result = inspect_safetensors(path, tensor="matrix", values=3, checksum=True)
    assert result["tensors"][0]["shape"] == [4, 6]
    assert result["tensors"][0]["flat_prefix_values"] == [0.0, 1.0, 2.0]
    assert result["tensors"][0]["storage_bytes"] == matrix.nbytes
    assert result["sha256"] == hashlib.sha256(original).hexdigest()
    assert inspect_safetensors(path, limit=1)["truncated"]
    assert (
        main(["weights", "inspect", "--file", str(path), "--tensor", "scalar", "--values", "1"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["tensors"][0]["flat_prefix_values"] == [7]
    assert path.read_bytes() == original


def test_bfloat16_and_nonfinite_inspection_are_json_safe(tmp_path):
    path = tmp_path / "bf16.safetensors"
    raw_file(
        path,
        {"bf": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}},
        np.array([0x3F80, 0xC000], dtype="<u2").tobytes(),
    )
    assert inspect_safetensors(path, values=2)["tensors"][0]["flat_prefix_values"] == [1.0, -2.0]
    floats = tmp_path / "nonfinite.safetensors"
    save_file({"x": np.array([np.nan, np.inf, -np.inf], dtype=np.float32)}, str(floats))
    assert inspect_safetensors(floats, values=3)["tensors"][0]["flat_prefix_values"] == [
        "NaN",
        "Infinity",
        "-Infinity",
    ]
    json.dumps(inspect_safetensors(floats, values=3), allow_nan=False)


def test_sparse_file_is_sampled_without_reading_a_full_weight_matrix(tmp_path):
    path = tmp_path / "sparse.safetensors"
    count = 64 * 1024 * 1024
    header = json.dumps(
        {"large": {"dtype": "U8", "shape": [count], "data_offsets": [0, count]}}
    ).encode()
    header += b" " * (-len(header) % 8)
    with path.open("wb") as stream:
        stream.write(len(header).to_bytes(8, "little") + header)
        stream.truncate(
            8 + len(header) + count
        )  # sparse, synthetic fixture; not 64 MiB of dataset data
    result = inspect_safetensors(path, values=4)
    assert result["tensors"][0]["numel"] == count
    assert result["tensors"][0]["flat_prefix_values"] == [0, 0, 0, 0]


@pytest.mark.parametrize(
    "kwargs",
    [{"limit": 0}, {"limit": 1001}, {"values": 65}, {"values": True}, {"tensor": "absent"}],
)
def test_inspection_limits_are_enforced(tmp_path, kwargs):
    path = tmp_path / "x.safetensors"
    save_file({"x": np.ones(2, dtype=np.float32)}, str(path))
    with pytest.raises(ValueError):
        inspect_safetensors(path, **kwargs)


def test_pickle_invalid_offsets_and_oversized_headers_are_refused(tmp_path):
    bad = tmp_path / "unsafe.bin"
    bad.write_bytes(b"never unpickled")
    assert main(["weights", "inspect", "--file", str(bad)]) == 2
    oversized = tmp_path / "oversized.safetensors"
    oversized.write_bytes((MAX_HEADER_BYTES + 1).to_bytes(8, "little") + b"{}")
    with pytest.raises(ValueError, match="header"):
        inspect_safetensors(oversized)
    offsets = tmp_path / "offsets.safetensors"
    raw_file(offsets, {"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 400]}}, b"\0" * 4)
    with pytest.raises(ValueError, match="safetensors"):
        inspect_safetensors(offsets)


def format_patch(directory, indices=None):
    directory.mkdir()
    indices = np.array([0], dtype=np.int64) if indices is None else indices
    count = len(indices)
    arrays = {
        "indices": indices,
        "before_bytes": np.full((count, 1), -1, dtype=np.float32).view(np.uint8).reshape(count, 4),
        "after_bytes": np.zeros((count, 4), dtype=np.uint8),
    }
    data = directory / "delta.safetensors"
    save_file(arrays, str(data))

    def tensor_hash(array):
        header = json.dumps(
            {"shape": list(array.shape), "dtype": "torch.float32"}, sort_keys=True
        ).encode()
        return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()

    meta = {
        "schema_version": 1,
        "kind": "ouroboros-sparse-weight-patch",
        "passed": True,
        "parameter_name": "weight",
        "shape": [2, 2],
        "dtype": "float32",
        "byte_order": "little",
        "changed_entries": count,
        "edited_fraction": 1.0,
        "max_relative_norm": 1.0,
        "max_patch_bytes": MAX_PATCH_BYTES,
        "license": "MIT",
        "provenance": "synthetic format unit test",
        "before_sha256": tensor_hash(np.diag(np.array([-1, 1], dtype=np.float32))),
        "after_sha256": tensor_hash(np.diag(np.array([0, 1], dtype=np.float32))),
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    }
    (directory / "patch.json").write_text(json.dumps(meta))
    return meta


def test_data_only_patch_verification_without_model_or_torch(tmp_path):
    patch = tmp_path / "patch"
    format_patch(patch)
    result = verify_weight_patch(patch)
    assert result["verified"] and result["requires_matching_tensor"]
    assert main(["weights", "verify-patch", "--patch", str(patch)]) == 0
    data = patch / "delta.safetensors"
    data.write_bytes(data.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_weight_patch(patch)


@pytest.mark.parametrize(
    "indices",
    [
        np.array([-1], dtype=np.int64),
        np.array([4], dtype=np.int64),
        np.array([0, 0], dtype=np.int64),
        np.array([1, 0], dtype=np.int64),
    ],
)
def test_bad_patch_indices_are_rejected(tmp_path, indices):
    patch = tmp_path / "patch"
    format_patch(patch, indices)
    with pytest.raises(ValueError, match="indices"):
        verify_weight_patch(patch)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"edited_fraction": True},
        {"max_relative_norm": -1},
        {"strength": 2},
        {"seed": -1},
        {"max_patch_bytes": MAX_PATCH_BYTES + 1},
    ],
)
def test_bad_weight_trial_plans(kwargs):
    with pytest.raises(ValueError):
        WeightEditPlan("weight", **kwargs)


def test_trial_config_check_only_is_offline_and_does_not_load_model(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    path = ROOT / "examples/weight_trial.json"
    config = load_weight_trial_config(path)
    assert Path(config["model"].model_name_or_path) == ROOT / "models/local-causal-lm"
    assert main(["weights", "trial", "--config", str(path), "--check-only"]) == 0
    assert json.loads(capsys.readouterr().out)["model_loaded"] is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("shape", [2**100, 2]),
        ("shape", [2, True]),
        ("dtype", "I64"),
        ("changed_entries", 0),
        ("max_patch_bytes", 1),
        ("license", ""),
        ("passed", False),
        ("before_sha256", "invalid"),
    ],
)
def test_malformed_patch_metadata_fails_before_any_model_is_needed(tmp_path, key, value):
    directory = tmp_path / "patch"
    meta = format_patch(directory)
    meta[key] = value
    (directory / "patch.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        verify_weight_patch(directory)


def test_extra_files_are_not_silently_treated_as_verified_patch_contents(tmp_path):
    directory = tmp_path / "patch"
    format_patch(directory)
    (directory / "unexpected.py").write_text("raise RuntimeError('never imported')")
    with pytest.raises(ValueError, match="inventory"):
        verify_weight_patch(directory)
