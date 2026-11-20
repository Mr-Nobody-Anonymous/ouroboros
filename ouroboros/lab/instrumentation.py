"""Metered read-only weights/hidden/MLP/attention/log-probability tools for trusted models."""

import inspect
import re

import numpy as np

from ..core.weight_space_programmer import WeightSpaceProgrammer
from .backend import frozen_eval
from .config import integer
from .state import editable_parameter, protected_names


def weight_statistics(copy, *, names=None, limit=20, values=0, weights="edited"):
    integer("weight statistic limit", limit, 1, 100)
    integer("weight prefix values", values, 0, 256)
    if weights not in ("original", "edited"):
        raise ValueError("weights must be original or edited")
    if names is None:
        forbidden = protected_names(copy.model)
        names = [
            name
            for name, p in copy.model.named_parameters()
            if name not in forbidden
            and 1 <= p.ndim <= 32
            and 1 <= p.numel() <= copy.policy.max_matrix_elements
        ]
    if (
        not isinstance(names, list)
        or len(names) > 1000
        or any(not isinstance(name, str) for name in names)
    ):
        raise ValueError("Expected at most 1000 canonical tensor names")
    rows = []
    for name in names:
        editable_parameter(copy.model, name, copy.policy)
        tensor = copy.current(name) if weights == "edited" else copy.original(name)
        value = tensor.double().numpy()
        absolute = np.abs(value)
        median, maximum = float(np.median(absolute)), float(absolute.max())
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite weights in inspected tensor")
        row = {
            "name": name,
            "shape": list(value.shape),
            "mean": float(value.mean()),
            "std": float(value.std()),
            "median_absolute": median,
            "max_absolute": maximum,
            "max_to_median": maximum / median if median else None,
            "flat_prefix": value.reshape(-1)[:values].tolist(),
        }
        if value.ndim == 2:
            norms = np.linalg.norm(value, axis=1)
            largest = int(np.argmax(norms))
            row.update(largest_row_norm_index=largest, largest_row_norm=float(norms[largest]))
        rows.append(row)
    rows.sort(key=lambda row: (-(row["max_to_median"] or 0), row["name"]))
    return {
        "weights": weights,
        "tensors": rows[:limit],
        "scanned_tensors": len(rows),
        "truncated": len(rows) > limit,
        "interpretation": "Numeric outliers only; not a semantic or causal diagnosis.",
    }


def _positions(length, last, stride):
    start = max(0, length - last) if last else 0
    return list(range(start, length, stride))


def read_states(backend, *, policy, kind, texts, modules=(), pooling="mean", last=1, stride=1):
    import torch

    if kind not in ("hidden", "mlp_input", "attention", "logprobs"):
        raise ValueError("Unknown read kind")
    if (
        not isinstance(texts, list)
        or not 1 <= len(texts) <= policy.max_read_texts
        or any(not isinstance(text, str) or not 1 <= len(text) <= 32000 for text in texts)
    ):
        raise ValueError("Read texts exceed the configured text/character budget")
    if (
        not isinstance(modules, (list, tuple))
        or len(modules) > 16
        or any(not isinstance(name, str) for name in modules)
        or len(set(modules)) != len(modules)
    ):
        raise ValueError("Read requires at most 16 unique module names")
    if pooling not in ("mean", "positions"):
        raise ValueError("pooling must be mean or positions")
    integer("last positions (0 means all)", last, 0, policy.max_read_tokens)
    integer("position stride", stride, 1, policy.max_read_tokens)
    model_modules = dict(backend.model.named_modules())
    if kind in ("hidden", "mlp_input") and (
        not modules or any(name not in model_modules for name in modules)
    ):
        raise ValueError("Choose existing named modules for hidden/MLP reads")
    if kind == "attention" and not policy.attention_maps:
        raise ValueError(
            "Attention maps require explicit policy.attention_maps and an eager-compatible architecture"
        )
    layer_indices = []
    if kind == "attention":
        for name in modules:
            match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", name)
            if name not in model_modules or match is None:
                raise ValueError("Attention modules must name indexed decoder layers")
            layer_indices.append(int(match.group(1)))
        if not layer_indices:
            raise ValueError("Choose attention layer modules")
    config = getattr(backend.model.config, "text_config", backend.model.config)
    vocabulary = getattr(config, "vocab_size", None)
    handles, rows, elements = [], [], 0
    captured, current_length = {}, 0

    def hook(name):
        def capture(module, args, output):
            nonlocal elements
            value = args[0] if kind == "mlp_input" else output
            if isinstance(value, tuple):
                value = value[0]
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 3
                or value.shape[:2] != (1, current_length)
            ):
                raise ValueError("Selected module does not expose a [1, tokens, features] tensor")
            if name in captured:
                raise ValueError("Selected module ran more than once; ambiguous read")
            positions = _positions(current_length, last, stride)
            selected = value[0, positions, :]
            if pooling == "mean":
                selected = selected.mean(dim=0)
            elements += selected.numel()
            if elements > policy.max_read_elements:
                raise ValueError("Read output element budget exceeded")
            if not torch.isfinite(selected).all():
                raise ValueError("Nonfinite hidden/MLP measurement")
            captured[name] = {
                "positions": positions,
                "shape": list(selected.shape),
                "values": selected.detach().float().cpu().tolist(),
            }

        return capture

    try:
        if kind in ("hidden", "mlp_input"):
            handles = [model_modules[name].register_forward_hook(hook(name)) for name in modules]
        with frozen_eval(backend.model):
            for index, text in enumerate(texts):
                encoded = backend.tokenizer(
                    text, add_special_tokens=True, return_token_type_ids=False
                )
                tokens = encoded["input_ids"]
                current_length = min(len(tokens), policy.max_read_tokens, backend.context_size)
                if current_length < 2:
                    raise ValueError("A read needs at least two tokens")
                ids = torch.tensor(
                    [tokens[:current_length]], dtype=torch.long, device=backend.device
                )
                kwargs = {
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                    "use_cache": False,
                    "output_attentions": kind == "attention",
                }
                last_only = (
                    kind not in ("logprobs",)
                    and "logits_to_keep" in inspect.signature(backend.model.forward).parameters
                )
                if last_only:
                    kwargs["logits_to_keep"] = 1
                if (
                    vocabulary is None
                    or (1 if last_only else current_length) * vocabulary > policy.max_logit_elements
                ):
                    raise ValueError("Forward logits exceed the configured element budget")
                if kind == "attention":
                    layers, heads = (
                        getattr(config, "num_hidden_layers", None),
                        getattr(config, "num_attention_heads", None),
                    )
                    if (
                        not layers
                        or not heads
                        or layers * heads * current_length**2 > policy.max_read_elements
                    ):
                        raise ValueError(
                            "Full attention allocation exceeds the read element budget"
                        )
                captured = {}
                output = backend.model(**kwargs)
                row = {
                    "text_index": index,
                    "tokens": current_length,
                    "tokens_truncated": len(tokens) > current_length,
                }
                if kind in ("hidden", "mlp_input"):
                    if set(captured) != set(modules):
                        raise ValueError("Some selected modules were not executed")
                    row["modules"] = dict(captured)
                elif kind == "logprobs":
                    logits = output.logits[0, :-1].float()
                    logprobs = logits.gather(1, ids[0, 1:, None]).squeeze(1) - torch.logsumexp(
                        logits, dim=1
                    )
                    positions = [p for p in _positions(current_length, last, stride) if p > 0]
                    elements += len(positions)
                    if elements > policy.max_read_elements or not torch.isfinite(logprobs).all():
                        raise ValueError("Invalid/oversized token-logprob measurement")
                    row["token_logprobs"] = [
                        {
                            "position": p,
                            "token_id": tokens[p],
                            "logprob": float(logprobs[p - 1].cpu()),
                        }
                        for p in positions
                    ]
                else:
                    attentions = getattr(output, "attentions", None)
                    if attentions is None or any(
                        attentions[layer] is None for layer in layer_indices
                    ):
                        raise ValueError(
                            "Architecture/backend did not return attention maps; no fallback inference"
                        )
                    row["modules"] = {}
                    for name, layer in zip(modules, layer_indices):
                        positions = _positions(current_length, last, stride)
                        value = attentions[layer][0, :, positions, :]
                        elements += value.numel()
                        if elements > policy.max_read_elements or not torch.isfinite(value).all():
                            raise ValueError("Invalid/oversized attention output")
                        row["modules"][name] = {
                            "query_positions": positions,
                            "key_token_ids": tokens[:current_length],
                            "shape": list(value.shape),
                            "values": value.float().cpu().tolist(),
                        }
                rows.append(row)
                del output
    finally:
        for handle in handles:
            handle.remove()
    return {
        "kind": kind,
        "pooling": pooling,
        "elements": elements,
        "texts": rows,
        "note": "Teacher-forced raw text, no sampled generation. Reads use the requested target weights, not the clean editor.",
    }


def measure_direction(copy, backend, *, module, failing, successful, last=0):
    if (
        not isinstance(failing, list)
        or not failing
        or not isinstance(successful, list)
        or not successful
    ):
        raise ValueError("Both failing and successful text groups are required")
    result = read_states(
        backend,
        policy=copy.policy,
        kind="hidden",
        texts=failing + successful,
        modules=[module],
        pooling="mean",
        last=last,
    )
    vectors = np.asarray([row["modules"][module]["values"] for row in result["texts"]])
    direction = WeightSpaceProgrammer.contrastive_direction(
        vectors[: len(failing)], vectors[len(failing) :]
    ).reshape(-1)
    identifier = f"d{len(copy.directions)}"
    copy.directions[identifier] = direction
    return {
        "direction_id": identifier,
        "module": module,
        "dimension": direction.size,
        "failing_texts": len(failing),
        "successful_texts": len(successful),
        "unit_norm": float(np.linalg.norm(direction)),
        "measurement": result,
        "interpretation": "Measured group-mean difference, not proof of a causal repetition direction.",
    }


def rank_writers(copy, name, direction, *, limit=16):
    integer("writer limit", limit, 1, 256)
    if direction not in copy.directions:
        raise ValueError("Unknown measured direction id")
    matrix, unit = copy.current(name).double().numpy(), copy.directions[direction]
    if matrix.ndim != 2 or matrix.shape[0] != len(unit):
        raise ValueError("Measured direction does not match the matrix output dimension")
    projections = unit @ matrix
    indices = np.argsort(-np.abs(projections), kind="stable")[:limit]
    return {
        "name": name,
        "direction_id": direction,
        "columns": [{"index": int(j), "projection": float(projections[j])} for j in indices],
        "method": "absolute output-column projection (no causal or activation-frequency claim)",
    }
