"""Owned multi-tensor overlays. The resident model is the frozen editor, never the working copy."""

from contextlib import contextmanager

from ..core.weight_trials import DTYPE_BYTES, _tensor_hash


def parameters(model):
    return dict(model.named_parameters())


def storage_key(parameter):
    return (
        None
        if parameter.is_meta or str(parameter.layout) != "torch.strided"
        else (str(parameter.device), parameter.untyped_storage().data_ptr())
    )


def refuse_shared_storage(model, name):
    values = parameters(model)
    selected = values[name]
    key = storage_key(selected)
    if key is not None and any(
        other is not selected and storage_key(other) == key for other in values.values()
    ):
        raise ValueError("Distinct parameters sharing storage are not supported by research edits")


def protected_names(model):
    """Protect tied aliases too; architectural name heuristics supplement embedding APIs."""
    protected_ids = set()
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        method = getattr(model, getter, None)
        module = method() if callable(method) else None
        if module is not None:
            protected_ids.update(id(p) for p in module.parameters())
    all_names = list(model.named_parameters(remove_duplicate=False))
    for name, parameter in all_names:
        parts = name.split(".")
        if any(
            part in {"embed_tokens", "word_embeddings", "wte", "lm_head", "embed_out", "ln_f"}
            for part in parts
        ):
            protected_ids.add(id(parameter))
        if name in {
            "norm.weight",
            "norm.bias",
            "model.norm.weight",
            "model.norm.bias",
            "model.language_model.norm.weight",
            "model.language_model.norm.bias",
            "transformer.final_layernorm.weight",
            "transformer.final_layernorm.bias",
        }:
            protected_ids.add(id(parameter))
    protected_storages = {
        storage_key(p) for _, p in all_names if id(p) in protected_ids and not p.is_meta
    }
    return {
        name
        for name, parameter in all_names
        if id(parameter) in protected_ids
        or (not parameter.is_meta and storage_key(parameter) in protected_storages)
    }


def editable_parameter(model, name, policy):
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise ValueError("Quantized model editing is not supported")
    values = parameters(model)
    if name not in values:
        raise ValueError(f"Unknown canonical parameter: {name}")
    if name in protected_names(model):
        raise ValueError(
            "Embeddings, output layers, final normalization and tied aliases are protected"
        )
    value = values[name]
    if (
        not 1 <= value.ndim <= 32
        or str(value.dtype).removeprefix("torch.") not in DTYPE_BYTES
        or str(value.layout) != "torch.strided"
    ):
        raise ValueError("Edits require ordinary dense floating parameters of rank 1-32")
    if not 1 <= value.numel() <= policy.max_matrix_elements or value.is_meta:
        raise ValueError("Tensor exceeds the configured dense working-copy budget")
    refuse_shared_storage(model, name)
    return value


@contextmanager
def temporary_weights(model, overlay):
    """Validate and snapshot ALL named tensors before any write; restore even on cancellation.

    No full-model copy: unrelated trusted callback writes are detected by the outer full hash,
    not magically undone. Never run this concurrently with inference/training on the same model.
    """
    import torch

    values = parameters(model)
    snapshots, modes = {}, [(module, module.training) for module in model.modules()]
    for name, candidate in overlay.items():
        if name not in values:
            raise ValueError(f"Overlay parameter missing: {name}")
        parameter = values[name]
        refuse_shared_storage(model, name)
        if (
            parameter.shape != candidate.shape
            or parameter.dtype != candidate.dtype
            or not torch.isfinite(candidate).all()
        ):
            raise ValueError("Overlay shape/dtype/finite mismatch")
        snapshots[name] = parameter.detach().cpu().clone()
    try:
        model.eval()
        with torch.no_grad():
            for name, candidate in overlay.items():
                values[name].copy_(candidate.to(values[name].device))
        yield
    finally:
        with torch.no_grad():
            for name, original in snapshots.items():
                values[name].copy_(original.to(values[name].device))
        for module, training in modes:
            module.training = training
        if any(
            _tensor_hash(values[name]) != _tensor_hash(original)
            for name, original in snapshots.items()
        ):
            raise RuntimeError("Owned tensor restoration failed")


class WorkingCopy:
    def __init__(self, model, policy, *, initial=None):
        self.model, self.policy = model, policy
        self.initial = {
            name: value.detach().cpu().clone() for name, value in (initial or {}).items()
        }
        self.candidate = {}
        self.saved = {}
        self.directions = {}
        self.total_parameters = sum(p.numel() for p in model.parameters())

    def original(self, name):
        parameter = editable_parameter(self.model, name, self.policy)
        return self.initial.get(name, parameter.detach()).cpu().clone()

    def current(self, name, candidate=None):
        candidate = self.candidate if candidate is None else candidate
        return candidate[name].clone() if name in candidate else self.original(name)

    def overlay(self, candidate=None):
        return {**self.initial, **(self.candidate if candidate is None else candidate)}

    def use(self, candidate=None):
        return temporary_weights(self.model, self.overlay(candidate))

    def inventory(self, *, offset=0, limit=40):
        forbidden = protected_names(self.model)
        rows = []
        for name, parameter in self.model.named_parameters():
            rows.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "numel": parameter.numel(),
                    "protected": name in forbidden,
                    "within_working_copy_budget": 1 <= parameter.ndim <= 32
                    and str(parameter.dtype).removeprefix("torch.") in DTYPE_BYTES
                    and str(parameter.layout) == "torch.strided"
                    and not parameter.is_meta
                    and 1 <= parameter.numel() <= self.policy.max_matrix_elements,
                }
            )
        return {
            "parameters": self.total_parameters,
            "tensors": rows[offset : offset + limit],
            "total_tensors": len(rows),
            "next_offset": offset + limit if offset + limit < len(rows) else None,
        }


def make_damage(model, policy, spec):
    """Operator-only fixture generation. Never return this specification to the editor."""
    import numpy as np
    import torch

    if spec is None:
        return {}
    parameter = editable_parameter(model, spec["parameter_name"], policy)
    if parameter.ndim != 2 or spec["first_row"] + spec["rows"] > parameter.shape[0]:
        raise ValueError("Damage rows must fit a supported matrix")
    value = parameter.detach().cpu().clone()
    std = float(value.double().std(unbiased=False))
    if not std:
        raise ValueError("Cannot generate noise damage from a zero-variance matrix")
    rng = np.random.default_rng(spec["seed"])
    noise = rng.normal(0, std * spec["std_multiplier"], (spec["rows"], parameter.shape[1]))
    first = spec["first_row"]
    value[first : first + spec["rows"]] = torch.as_tensor(noise, dtype=value.dtype)
    if not torch.isfinite(value).all():
        raise ValueError("Damage overflowed the stored dtype")
    return {spec["parameter_name"]: value}
