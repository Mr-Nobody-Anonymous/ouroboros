"""Accepted multi-tensor patches, bound to a complete input/output model state.

JSON + sparse exact before/after bytes, never pickle or executable code. Full hashes provide
integrity/identity of content, not author authentication, legal permission or behavioral safety.
"""

import math
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from safetensors import SafetensorError
from safetensors.numpy import load_file, save_file

from ..core.checkpoints import atomic_json, file_sha256
from ..core.weight_trials import DTYPE_BYTES, MAX_PATCH_BYTES, _payload, _tensor_hash
from ..self_edit import model_fingerprint
from .config import integer, number, strict_json
from .program import validate_candidate
from .state import parameters, protected_names, refuse_shared_storage, temporary_weights


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def export_bundle(copy, directory, *, provenance, license_name):
    clean, stats = validate_candidate(copy, copy.candidate)
    if not clean:
        raise ValueError("Cannot export an empty/no-op edit")
    if any(not isinstance(value, str) or not value.strip() for value in (provenance, license_name)):
        raise ValueError("Explicit provenance and license required")
    directory = Path(directory)
    if directory.exists() or directory.is_symlink():
        raise ValueError("Bundle destination must be new")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with temporary_weights(copy.model, copy.initial):
        before = model_fingerprint(copy.model)
    with copy.use(clean):
        after = model_fingerprint(copy.model)
    arrays, tensors = {}, []
    for i, (name, candidate) in enumerate(sorted(clean.items())):
        original = copy.original(name)
        payload = _payload(original, candidate)
        arrays.update({f"t{i}_{key}": value for key, value in payload.items()})
        tensors.append(
            {
                "name": name,
                "shape": list(original.shape),
                "dtype": str(original.dtype).removeprefix("torch."),
                "before_sha256": _tensor_hash(original),
                "after_sha256": _tensor_hash(candidate),
                "before_norm": float(original.double().norm()),
                "changed_entries": len(payload["indices"]),
            }
        )
    temporary = Path(tempfile.mkdtemp(prefix=".pending-bundle-", dir=directory.parent))
    try:
        data = temporary / "changes.safetensors"
        save_file(arrays, str(data))
        if data.stat().st_size > copy.policy.max_patch_bytes:
            raise ValueError("Serialized multi-tensor patch exceeds byte budget")
        manifest = {
            "schema_version": 1,
            "kind": "ouroboros-multi-tensor-patch",
            "passed": True,
            "before_state_sha256": before,
            "after_state_sha256": after,
            "data_sha256": file_sha256(data),
            "tensors": tensors,
            "total_parameters": copy.total_parameters,
            "changed_entries": stats["changed_entries"],
            "max_edited_fraction": copy.policy.max_edited_fraction,
            "max_changed_entries": copy.policy.max_changed_entries,
            "max_relative_norm": copy.policy.max_relative_norm,
            "max_patch_bytes": copy.policy.max_patch_bytes,
            "byte_order": "little",
            "license": license_name,
            "provenance": provenance,
        }
        atomic_json(temporary / "bundle.json", manifest)
        verify_bundle(temporary)
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return str(directory)


def _decode(raw, dtype):
    raw = np.ascontiguousarray(raw)
    if dtype == "bfloat16":
        return (raw.view("<u2").reshape(-1).astype(np.uint32) << 16).view("<f4").astype(np.float64)
    return (
        raw.view({"float16": "<f2", "float32": "<f4", "float64": "<f8"}[dtype])
        .reshape(-1)
        .astype(np.float64)
    )


def _load_bundle(directory):
    directory = Path(directory)
    manifest_path, data_path = directory / "bundle.json", directory / "changes.safetensors"
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or any(not p.is_file() or p.is_symlink() for p in (manifest_path, data_path))
    ):
        raise ValueError("A real bundle directory and nonsymlinked files required")
    if {p.name for p in directory.iterdir()} != {"bundle.json", "changes.safetensors"}:
        raise ValueError("Unexpected bundle inventory")
    if manifest_path.stat().st_size > 1024**2 or data_path.stat().st_size > MAX_PATCH_BYTES:
        raise ValueError("Bundle exceeds reader byte budgets")
    meta = strict_json(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "kind",
        "passed",
        "before_state_sha256",
        "after_state_sha256",
        "data_sha256",
        "tensors",
        "total_parameters",
        "changed_entries",
        "max_edited_fraction",
        "max_changed_entries",
        "max_relative_norm",
        "max_patch_bytes",
        "byte_order",
        "license",
        "provenance",
    }
    if not isinstance(meta, dict) or set(meta) != required:
        raise ValueError("Incomplete/unexpected bundle manifest")
    if (
        type(meta["schema_version"]) is not int
        or meta["schema_version"] != 1
        or meta["kind"] != "ouroboros-multi-tensor-patch"
        or meta["passed"] is not True
        or meta["byte_order"] != "little"
    ):
        raise ValueError("Not an accepted little-endian Ouroboros multi-tensor patch")
    for name in ("before_state_sha256", "after_state_sha256", "data_sha256"):
        if not _sha(meta[name]):
            raise ValueError("Invalid bundle state/data hash")
    if meta["before_state_sha256"] == meta["after_state_sha256"]:
        raise ValueError("Bundle must have a nonempty state change")
    for key in ("license", "provenance"):
        if not isinstance(meta[key], str) or not 1 <= len(meta[key].strip()) <= 4000:
            raise ValueError("Bundle license/provenance required")
    integer("total_parameters", meta["total_parameters"], 1, 100_000_000_000)
    integer("max_changed_entries", meta["max_changed_entries"], 1, 4_000_000)
    integer("changed_entries", meta["changed_entries"], 1, meta["max_changed_entries"])
    integer("max_patch_bytes", meta["max_patch_bytes"], 1024, MAX_PATCH_BYTES)
    number("max_edited_fraction", meta["max_edited_fraction"], 0, 1)
    number("max_relative_norm", meta["max_relative_norm"], 0, 2)
    if meta["changed_entries"] > math.floor(meta["total_parameters"] * meta["max_edited_fraction"]):
        raise ValueError("Bundle exceeds global edited fraction")
    if (
        data_path.stat().st_size > meta["max_patch_bytes"]
        or file_sha256(data_path) != meta["data_sha256"]
    ):
        raise ValueError("Bundle checksum/byte budget mismatch")
    if not isinstance(meta["tensors"], list) or not 1 <= len(meta["tensors"]) <= 128:
        raise ValueError("Bundle tensor count exceeds reader budget")
    try:
        arrays = load_file(str(data_path))
    except SafetensorError as exc:
        raise ValueError(f"Invalid bundle safetensors: {exc}") from exc
    expected, names, count = set(), set(), 0
    for i, tensor in enumerate(meta["tensors"]):
        keys = {
            "name",
            "shape",
            "dtype",
            "before_sha256",
            "after_sha256",
            "before_norm",
            "changed_entries",
        }
        if not isinstance(tensor, dict) or set(tensor) != keys:
            raise ValueError("Invalid bundle tensor descriptor")
        name, shape, dtype = tensor["name"], tensor["shape"], tensor["dtype"]
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9_.]+", name) is None
            or name in names
        ):
            raise ValueError("Invalid/duplicate tensor name")
        names.add(name)
        if (
            not isinstance(shape, list)
            or not 1 <= len(shape) <= 32
            or any(type(n) is not int or n < 1 for n in shape)
            or math.prod(shape) > 512_000_000
            or dtype not in DTYPE_BYTES
        ):
            raise ValueError("Invalid bundle tensor shape/dtype")
        if (
            not _sha(tensor["before_sha256"])
            or not _sha(tensor["after_sha256"])
            or tensor["before_sha256"] == tensor["after_sha256"]
        ):
            raise ValueError("Invalid tensor hashes")
        integer(
            "tensor changed_entries",
            tensor["changed_entries"],
            1,
            min(math.prod(shape), meta["changed_entries"]),
        )
        number("before_norm", tensor["before_norm"], 1e-300, 1e100)
        prefix = f"t{i}_"
        expected.update(prefix + key for key in ("indices", "before_bytes", "after_bytes"))
        if not expected <= set(arrays):
            raise ValueError("Missing bundle payload arrays")
        indices, before, after = (
            arrays[prefix + key] for key in ("indices", "before_bytes", "after_bytes")
        )
        n = tensor["changed_entries"]
        if (
            indices.dtype != np.int64
            or indices.shape != (n,)
            or indices[0] < 0
            or indices[-1] >= math.prod(shape)
            or np.any(indices[1:] <= indices[:-1])
        ):
            raise ValueError("Invalid/duplicate/out-of-range bundle indices")
        if (
            before.dtype != np.uint8
            or after.dtype != np.uint8
            or before.shape != (n, DTYPE_BYTES[dtype])
            or after.shape != before.shape
        ):
            raise ValueError("Invalid bundle byte arrays")
        left, right = _decode(before, dtype), _decode(after, dtype)
        if not np.isfinite(left).all() or not np.isfinite(right).all() or np.any(left == right):
            raise ValueError("Nonfinite/equal entries are not valid sparse changes")
        if (
            float(np.linalg.norm(right - left)) / tensor["before_norm"]
            > meta["max_relative_norm"] + 1e-12
        ):
            raise ValueError("Serialized norm budget exceeded")
        count += n
    if set(arrays) != expected or count != meta["changed_entries"]:
        raise ValueError("Unexpected payload arrays or changed-entry count")
    return meta, arrays


def verify_bundle(directory):
    meta, _ = _load_bundle(directory)
    return {"valid": True, **meta}


def _stage(model, directory, *, reverse, max_snapshot_elements):
    import sys

    import torch

    if (
        sys.byteorder != "little"
        or getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
    ):
        raise ValueError("Bundle application requires ordinary weights on a little-endian host")
    meta, arrays = _load_bundle(directory)
    integer("max_snapshot_elements", max_snapshot_elements, 1, 4_000_000_000)
    if sum(math.prod(t["shape"]) for t in meta["tensors"]) > max_snapshot_elements:
        raise ValueError("Bundle exceeds the explicit dense snapshot element budget")
    source = "after" if reverse else "before"
    destination = "before" if reverse else "after"
    if model_fingerprint(model) != meta[source + "_state_sha256"]:
        raise ValueError("Complete model state does not match the bundle input")
    if sum(p.numel() for p in model.parameters()) != meta["total_parameters"]:
        raise ValueError("Full model parameter count mismatch")
    values, forbidden, staged = parameters(model), protected_names(model), {}
    for i, tensor in enumerate(meta["tensors"]):
        name = tensor["name"]
        if name not in values or name in forbidden:
            raise ValueError("Missing/protected bundle target")
        refuse_shared_storage(model, name)
        parameter = values[name]
        if (
            list(parameter.shape) != tensor["shape"]
            or str(parameter.dtype).removeprefix("torch.") != tensor["dtype"]
            or _tensor_hash(parameter) != tensor[source + "_sha256"]
        ):
            raise ValueError("Bundle target tensor shape/dtype/hash mismatch")
        original = parameter.detach().cpu().contiguous().clone()
        indices = torch.from_numpy(arrays[f"t{i}_indices"].copy())
        source_bytes = torch.from_numpy(arrays[f"t{i}_{source}_bytes"].copy())
        target_bytes = torch.from_numpy(arrays[f"t{i}_{destination}_bytes"].copy())
        raw = original.reshape(-1).view(torch.uint8).reshape(-1, original.element_size())
        if not torch.equal(raw[indices], source_bytes):
            raise ValueError("Bundle source entry bytes mismatch")
        raw[indices] = target_bytes
        if (
            not torch.isfinite(original).all()
            or _tensor_hash(original) != tensor[destination + "_sha256"]
        ):
            raise ValueError("Bundle reconstructed tensor mismatch")
        left = original if reverse else parameter.detach().cpu()
        right = parameter.detach().cpu() if reverse else original
        before_norm = float(left.double().norm())
        if not math.isclose(before_norm, tensor["before_norm"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Bundle original norm mismatch")
        if (
            float((right.double() - left.double()).norm()) / before_norm
            > meta["max_relative_norm"] + 1e-12
        ):
            raise ValueError("Bundle actual norm budget exceeded")
        scale = float((left.double() * right.double()).sum()) / before_norm**2
        if torch.allclose(right.double(), scale * left.double(), rtol=1e-6, atol=1e-12):
            raise ValueError("Whole-tensor rescaling is prohibited")
        staged[name] = original
    return meta, staged, destination


@contextmanager
def use_bundle(model, directory, *, reverse=False, max_snapshot_elements=8_000_000):
    meta, staged, destination = _stage(
        model, directory, reverse=reverse, max_snapshot_elements=max_snapshot_elements
    )
    with temporary_weights(model, staged):
        if model_fingerprint(model) != meta[destination + "_state_sha256"]:
            raise ValueError("Bundle output full-state mismatch")
        yield meta


def apply_bundle(model, directory, *, reverse=False, max_snapshot_elements=8_000_000):
    """Explicit all-target preflight followed by an all-or-rollback mutation. Not exposed by MCP."""
    import torch

    meta, staged, destination = _stage(
        model, directory, reverse=reverse, max_snapshot_elements=max_snapshot_elements
    )
    values = parameters(model)
    snapshots = {name: values[name].detach().cpu().clone() for name in staged}
    try:
        with torch.no_grad():
            for name, candidate in staged.items():
                values[name].copy_(candidate.to(values[name].device))
        if model_fingerprint(model) != meta[destination + "_state_sha256"]:
            raise ValueError("Bundle output full-state mismatch")
    except BaseException:
        with torch.no_grad():
            for name, original in snapshots.items():
                values[name].copy_(original.to(values[name].device))
        raise
    return {
        "applied": True,
        "reverse": reverse,
        "state_sha256": meta[destination + "_state_sha256"],
    }
