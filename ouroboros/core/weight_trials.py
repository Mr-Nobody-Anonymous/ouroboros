"""Measured single-tensor trials and sparse, reversible, non-executable weight patches.

A trial ALWAYS restores the live model, even when accepted. Applying an exported patch is a
separate explicit operation. This is not Astra's diagnosis or a self-improvement guarantee.
"""

import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from safetensors import SafetensorError
from safetensors.numpy import load_file, save_file

from ..config import probability
from ..data import Sample, unique_samples
from .checkpoints import atomic_json, file_sha256
from .evaluator import SelfEvaluator, evaluate_samples
from .weight_space_programmer import WeightSpaceProgrammer

MAX_PATCH_BYTES = 64 * 1024 * 1024
DTYPE_BYTES = {"float16": 2, "float32": 4, "float64": 8, "bfloat16": 2}


@dataclass(frozen=True)
class WeightEditPlan:
    parameter_name: str
    edited_fraction: float = 0.00016
    strength: float = 0.5
    max_relative_norm: float = 0.05
    forgetting_threshold: float = 0.05
    improvement_threshold: float = 0.0
    seed: int = 42
    max_patch_bytes: int = MAX_PATCH_BYTES

    def __post_init__(self):
        if not isinstance(self.parameter_name, str) or not self.parameter_name.strip():
            raise ValueError("A parameter name is required")
        for key in (
            "edited_fraction",
            "strength",
            "max_relative_norm",
            "forgetting_threshold",
            "improvement_threshold",
        ):
            probability(key, getattr(self, key))
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if (
            type(self.max_patch_bytes) is not int
            or not 1 <= self.max_patch_bytes <= MAX_PATCH_BYTES
        ):
            raise ValueError("max_patch_bytes must be an integer in [1, 64 MiB]")


def _parameter(model, name):
    if sys.byteorder != "little":
        raise ValueError("This patch implementation currently requires a little-endian host")
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise ValueError("Quantized models are not supported by direct weight trials")
    parameters = dict(model.named_parameters())
    if name not in parameters:
        raise ValueError(f"Parameter not found: {name}")
    parameter = parameters[name]
    if parameter.ndim != 2 or str(parameter.dtype).removeprefix("torch.") not in DTYPE_BYTES:
        raise ValueError("Direct patches require an ordinary floating-point 2D parameter")
    return parameter


def _tensor_hash(tensor):
    import torch

    tensor = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}, sort_keys=True
    ).encode()
    return hashlib.sha256(
        header + b"\0" + tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _seed(torch, seed, devices):
    torch.random.default_generator.manual_seed(seed)
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def _payload(original, candidate):
    import torch

    original, candidate = original.cpu().contiguous(), candidate.cpu().contiguous()
    indices = torch.nonzero(original.reshape(-1) != candidate.reshape(-1)).reshape(-1)
    size = original.element_size()
    return {
        "indices": indices.numpy().copy(),
        "before_bytes": original.reshape(-1)[indices]
        .contiguous()
        .view(torch.uint8)
        .reshape(-1, size)
        .numpy()
        .copy(),
        "after_bytes": candidate.reshape(-1)[indices]
        .contiguous()
        .view(torch.uint8)
        .reshape(-1, size)
        .numpy()
        .copy(),
    }


def _export(directory, payload, metadata, max_bytes):
    target = directory / "patch"
    temporary = Path(tempfile.mkdtemp(prefix=".pending-patch-", dir=directory))
    try:
        data = temporary / "delta.safetensors"
        save_file(payload, str(data))
        if data.stat().st_size > max_bytes:
            raise ValueError("Serialized patch exceeds its byte budget")
        atomic_json(temporary / "patch.json", {**metadata, "data_sha256": file_sha256(data)})
        verify_weight_patch(temporary)
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return str(target)


def run_weight_edit_trial(
    model,
    predict,
    validation,
    direction,
    plan: WeightEditPlan,
    output_dir,
    *,
    provenance,
    license_name,
    model_info=None,
    reference_scores=None,
):
    """Test one ablation, enforce count/norm/byte/retention gates, export only accepted patches.

    `predict(prompts)` is a trusted, deterministic, read-only callback. Labels are never passed
    to it. Only the selected parameter is snapshotted; arbitrary callback writes to other
    tensors/files are NOT isolated. Dense temporary CPU copies may require substantial RAM.
    """
    import torch

    if any(not isinstance(value, str) or not value.strip() for value in (provenance, license_name)):
        raise ValueError("Explicit model provenance and license are required")
    json.dumps(model_info or {}, allow_nan=False)
    validation = tuple(unique_samples(validation))
    if not validation:
        raise ValueError("Nonempty independent validation data are required")
    parameter = _parameter(model, plan.parameter_name)
    root = Path(output_dir)
    if root.is_symlink():
        raise ValueError("Trial output root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    directory = root / uuid.uuid4().hex
    directory.mkdir()
    started = time.monotonic()
    report = {
        "schema_version": 1,
        "kind": "single-tensor-ablation-trial",
        "run_id": directory.name,
        "plan": asdict(plan),
        "provenance": provenance,
        "license": license_name,
        "model_info": model_info or {},
        "passed": False,
        "status": "running",
        "patch": None,
        "baseline": None,
        "candidate": None,
        "model_restored": False,
        "validation_sha256": hashlib.sha256(
            json.dumps([s.as_dict() for s in validation], sort_keys=True).encode()
        ).hexdigest(),
    }
    original = parameter.detach().cpu().clone()
    modes = [(module, module.training) for module in model.modules()]
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
    payload, metadata = None, None
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            _seed(torch, plan.seed, devices)
            with torch.inference_mode():
                report["baseline"] = evaluate_samples(predict, validation)
            edit = WeightSpaceProgrammer(plan.edited_fraction).apply_to_torch_model(
                model,
                plan.parameter_name,
                direction,
                strength=plan.strength,
            )
            stats = report["edit"] = edit["stats"]
            if not all(
                math.isfinite(stats[key]) for key in ("frobenius_norm_delta", "relative_norm_delta")
            ):
                raise ValueError("Nonfinite edit norm")
            if not stats["edited_weights"]:
                report["reason"] = "No representable change within the entry budget"
            elif stats["relative_norm_delta"] > plan.max_relative_norm + 1e-12:
                report["reason"] = "Relative norm budget exceeded"
            elif (
                stats["edited_weights"] * (8 + 2 * original.element_size()) + 1024
                > plan.max_patch_bytes
            ):
                report["reason"] = (
                    "Sparse patch byte budget exceeded (including reserved header space)"
                )
            else:
                _seed(torch, plan.seed, devices)
                with torch.inference_mode():
                    report["candidate"] = evaluate_samples(predict, validation)
                evaluator = SelfEvaluator(
                    report["baseline"]["scores"] if reference_scores is None else reference_scores
                )
                passed, reason = evaluator.verify_improvement(
                    report["candidate"]["scores"],
                    previous_scores=report["baseline"]["scores"],
                    forgetting_threshold=plan.forgetting_threshold,
                    improvement_threshold=plan.improvement_threshold,
                )
                report.update(passed=passed, reason=reason)
                if passed:
                    candidate = parameter.detach().cpu().clone()
                    payload = _payload(original, candidate)
                    metadata = {
                        "schema_version": 1,
                        "kind": "ouroboros-sparse-weight-patch",
                        "passed": True,
                        "parameter_name": plan.parameter_name,
                        "shape": list(original.shape),
                        "dtype": str(original.dtype).removeprefix("torch."),
                        "byte_order": "little",
                        "before_sha256": _tensor_hash(original),
                        "after_sha256": _tensor_hash(candidate),
                        "changed_entries": stats["edited_weights"],
                        "edited_fraction": plan.edited_fraction,
                        "max_relative_norm": plan.max_relative_norm,
                        "max_patch_bytes": plan.max_patch_bytes,
                        "provenance": provenance,
                        "license": license_name,
                        "validation_sha256": report["validation_sha256"],
                        "trial_id": directory.name,
                    }
            report["status"] = "accepted" if report["passed"] else "rejected"
    except BaseException as exc:
        report.update(
            passed=False,
            status="interrupted" if not isinstance(exc, Exception) else "failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        # Restore using the saved parameter reference, even when validation/export is interrupted.
        with torch.no_grad():
            parameter.copy_(original.to(parameter.device))
        for module, training in modes:
            module.training = training
        report.update(model_restored=True, elapsed_seconds=time.monotonic() - started)
        atomic_json(directory / "trial.json", report)
    if report["passed"]:
        try:
            report["patch"] = _export(directory, payload, metadata, plan.max_patch_bytes)
            atomic_json(directory / "trial.json", report)
        except BaseException as exc:
            report.update(
                passed=False,
                patch=None,
                status="interrupted" if not isinstance(exc, Exception) else "failed",
                reason=f"Patch export failed: {type(exc).__name__}: {exc}",
            )
            atomic_json(directory / "trial.json", report)
            raise
    return {**report, "report_path": str(directory / "trial.json")}


def _load_patch(directory):
    directory = Path(directory)
    manifest_path, data_path = directory / "patch.json", directory / "delta.safetensors"
    if directory.is_symlink() or any(
        p.is_symlink() or not p.is_file() for p in (manifest_path, data_path)
    ):
        raise ValueError("A real patch directory with no symlinked files is required")
    if {entry.name for entry in directory.iterdir()} != {"patch.json", "delta.safetensors"}:
        raise ValueError("Unexpected patch file inventory")
    if manifest_path.stat().st_size > 1024 * 1024 or data_path.stat().st_size > MAX_PATCH_BYTES:
        raise ValueError("Patch file exceeds the reader's byte budget")
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "kind",
        "passed",
        "parameter_name",
        "shape",
        "dtype",
        "changed_entries",
        "byte_order",
        "edited_fraction",
        "max_relative_norm",
        "max_patch_bytes",
        "license",
        "provenance",
        "before_sha256",
        "after_sha256",
        "data_sha256",
    }
    if not isinstance(meta, dict) or not required <= set(meta):
        raise ValueError("Incomplete patch manifest")
    if (
        type(meta.get("schema_version")) is not int
        or meta["schema_version"] != 1
        or meta.get("kind") != "ouroboros-sparse-weight-patch"
        or meta.get("passed") is not True
    ):
        raise ValueError("Not an accepted Ouroboros weight patch")
    WeightEditPlan(
        meta["parameter_name"],
        edited_fraction=meta["edited_fraction"],
        max_relative_norm=meta["max_relative_norm"],
        max_patch_bytes=meta["max_patch_bytes"],
    )
    shape, dtype, count = meta["shape"], meta["dtype"], meta["changed_entries"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(n) is not int or not 1 <= n < 2**63 for n in shape)
        or math.prod(shape) >= 2**63
        or not isinstance(dtype, str)
        or dtype not in DTYPE_BYTES
        or meta.get("byte_order") != "little"
        or type(count) is not int
        or not 1 <= count <= math.floor(math.prod(shape) * meta["edited_fraction"])
    ):
        raise ValueError("Invalid patch shape, dtype or entry budget")
    if any(
        not isinstance(meta.get(key), str) or not meta[key].strip()
        for key in ("license", "provenance")
    ):
        raise ValueError("Patch license and provenance are required")
    for key in ("before_sha256", "after_sha256", "data_sha256"):
        value = meta[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Invalid patch SHA-256")
    if meta["before_sha256"] == meta["after_sha256"]:
        raise ValueError("A nonempty patch must change its tensor fingerprint")
    if (
        data_path.stat().st_size > meta["max_patch_bytes"]
        or file_sha256(data_path) != meta["data_sha256"]
    ):
        raise ValueError("Patch checksum or byte budget mismatch")
    try:
        arrays = load_file(str(data_path))
    except SafetensorError as exc:
        raise ValueError(f"Invalid patch safetensors: {exc}") from exc
    if set(arrays) != {"indices", "before_bytes", "after_bytes"}:
        raise ValueError("Unexpected patch tensor inventory")
    indices = arrays["indices"]
    if (
        indices.dtype != np.int64
        or indices.shape != (count,)
        or np.any(indices < 0)
        or np.any(indices >= math.prod(shape))
        or np.any(indices[1:] <= indices[:-1])
    ):
        raise ValueError("Patch indices must be sorted, unique and in bounds")
    for name in ("before_bytes", "after_bytes"):
        raw = arrays[name]
        if raw.dtype != np.uint8 or raw.shape != (count, DTYPE_BYTES[dtype]):
            raise ValueError("Patch value shape/dtype mismatch")

    def values(raw):
        if dtype == "bfloat16":
            return (raw.reshape(-1).view("<u2").astype(np.uint32) << 16).view(np.float32)
        return raw.reshape(-1).view({"float16": "<f2", "float32": "<f4", "float64": "<f8"}[dtype])

    before, after = values(arrays["before_bytes"]), values(arrays["after_bytes"])
    if not np.isfinite(before).all() or not np.isfinite(after).all() or np.any(before == after):
        raise ValueError("Patch entries must be finite and actually changed")
    return meta, arrays


def verify_weight_patch(directory):
    """Validate a data-only patch without Torch or access to the original model."""
    meta, _ = _load_patch(directory)
    return {
        key: meta[key]
        for key in (
            "kind",
            "parameter_name",
            "shape",
            "dtype",
            "changed_entries",
            "provenance",
            "license",
        )
    } | {
        "verified": True,
        "requires_matching_tensor": True,
        "note": "Checksums do not authenticate the author or prove behavior on a different full model.",
    }


def apply_weight_patch(model, directory, *, reverse=False):
    """Explicitly apply/undo a verified patch only to the exact expected tensor; never execute code.

    Both directions verify the complete input/output tensor hashes and the relative-norm bound.
    This binds the TARGET tensor, not every other weight in an arbitrary supplied model.
    """
    import torch

    if type(reverse) is not bool:
        raise ValueError("reverse must be a boolean")
    meta, arrays = _load_patch(directory)
    parameter = _parameter(model, meta["parameter_name"])
    if list(parameter.shape) != meta["shape"] or str(parameter.dtype) != "torch." + meta["dtype"]:
        raise ValueError("Patch target shape/dtype mismatch")
    original = parameter.detach().cpu().contiguous()
    expected_input = meta["after_sha256"] if reverse else meta["before_sha256"]
    expected_output = meta["before_sha256"] if reverse else meta["after_sha256"]
    if _tensor_hash(original) != expected_input:
        raise ValueError("Patch target tensor fingerprint mismatch; model was not changed")
    indices = torch.from_numpy(arrays["indices"].copy())
    raw = arrays["before_bytes" if reverse else "after_bytes"].copy()
    values = torch.from_numpy(raw).view(parameter.dtype).reshape(-1)
    expected_current = arrays["after_bytes" if reverse else "before_bytes"]
    current_bytes = (
        original.reshape(-1)[indices]
        .contiguous()
        .view(torch.uint8)
        .reshape(expected_current.shape)
        .numpy()
    )
    if not np.array_equal(current_bytes, expected_current):
        raise ValueError(
            "Patch payload does not match the input tensor values; model was not changed"
        )
    candidate = original.clone()
    candidate.reshape(-1)[indices] = values
    if _tensor_hash(candidate) != expected_output:
        raise ValueError("Patched tensor fingerprint mismatch; model was not changed")
    delta = float(torch.linalg.vector_norm(candidate.double() - original.double()))
    base = candidate if reverse else original
    base_norm = float(torch.linalg.vector_norm(base.double()))
    if (
        not math.isfinite(delta)
        or not math.isfinite(base_norm)
        or not base_norm
        or delta / base_norm > meta["max_relative_norm"] + 1e-12
    ):
        raise ValueError("Patch norm budget exceeded; model was not changed")
    with torch.no_grad():
        parameter.copy_(candidate.to(parameter.device))
    return {
        "parameter_name": meta["parameter_name"],
        "changed_entries": meta["changed_entries"],
        "reversed": reverse,
        "tensor_sha256": expected_output,
    }


def run_weight_edit_demo(output_dir):
    """A real two-feature/two-label Torch classifier fixture, NOT an LLM/math benchmark."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("The weight-edit demo requires Torch; install the train extra") from exc
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[-1.0, 0.0], [0.0, 1.0]]))
        features = {"repair fixture": [1.0, -0.5], "retention fixture": [1.0, 1.0]}

        def predict(prompts):
            with torch.inference_mode():
                logits = model(torch.tensor([features[p] for p in prompts]))
            return [str(index) for index in logits.argmax(dim=-1).tolist()]

        validation = [
            Sample("repair fixture", "0", "repair", "synthetic-fixture", True),
            Sample("retention fixture", "1", "retention", "synthetic-fixture", True),
        ]
        return run_weight_edit_trial(
            model,
            predict,
            validation,
            [1.0, 0.0],
            WeightEditPlan(
                "weight",
                edited_fraction=0.25,
                strength=1,
                max_relative_norm=0.75,
                forgetting_threshold=0,
            ),
            output_dir,
            provenance="Ouroboros deterministic two-feature classifier fixture, not pretrained weights",
            license_name="MIT",
            model_info={"is_llm": False, "fixture": "two-feature-classifier"},
        )
