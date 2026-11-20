"""Data-only weight programs: no eval, exec, imports, filesystem paths or model calls.

Projection/column selection follows the published weight-space experiment's algebra. It does
not imply a discovered semantic circuit. Every rounded candidate still needs independent tests.
"""

import math

import numpy as np

from .config import integer, number
from .state import editable_parameter


def _array(value, *, ndim=None):
    def contains_boolean(item):
        return isinstance(item, (bool, np.bool_)) or (
            isinstance(item, (list, tuple)) and any(contains_boolean(part) for part in item)
        )

    try:
        if contains_boolean(value):
            raise ValueError("Boolean entries are not numeric edit values")
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu":
            raise ValueError(
                "Numeric arrays cannot contain booleans, strings, objects or complex values"
            )
        array = raw.astype(np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Finite numeric array required") from exc
    if not array.size or not np.isfinite(array).all() or (ndim is not None and array.ndim != ndim):
        raise ValueError("Finite nonempty array with the expected dimensions required")
    return array


def _indices(value, size):
    if (
        not isinstance(value, list)
        or not value
        or any(type(i) is not int or not 0 <= i < size for i in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("Indices must be nonempty, unique integers within the target dimension")
    return np.asarray(value, dtype=np.int64)


def _keys(operation, required, optional=()):
    if (
        not isinstance(operation, dict)
        or not set(required) <= set(operation)
        or set(operation) - set(required) - set(optional)
    ):
        raise ValueError("Unexpected/missing fields in weight operation")


def _round(original, value):
    import torch

    candidate = torch.as_tensor(np.ascontiguousarray(value), dtype=original.dtype)
    if not torch.isfinite(candidate).all():
        raise ValueError("Candidate overflowed its stored dtype")
    # Preserve signed zeros and any equal-value original bits.
    return torch.where(candidate == original, original, candidate)


def validate_candidate(copy, candidate):
    clean, stats, changed, byte_estimate = {}, [], 0, 1024
    if len(candidate) > copy.policy.max_edit_tensors:
        raise ValueError("Multi-tensor edit count budget exceeded")
    for name, value in sorted(candidate.items()):
        editable_parameter(copy.model, name, copy.policy)
        original = copy.original(name)
        if value.shape != original.shape or value.dtype != original.dtype:
            raise ValueError("Candidate changed tensor shape/dtype")
        left, right = original.double().numpy(), value.double().numpy()
        if not np.isfinite(right).all():
            raise ValueError("Candidate must contain finite values")
        count = int(np.count_nonzero(left != right))
        if not count:
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            norm = float(np.linalg.norm(left))
            delta_norm = float(np.linalg.norm(right - left))
        if not math.isfinite(norm) or not math.isfinite(delta_norm):
            raise ValueError("Tensor/delta norm must be finite at working precision")
        relative = delta_norm / norm if norm else math.inf
        if not math.isfinite(relative) or relative > copy.policy.max_relative_norm + 1e-12:
            raise ValueError(f"Relative norm budget exceeded for {name}")
        if norm:
            scale = float(np.vdot(left, right) / (norm * norm))
            if np.allclose(right, scale * left, rtol=1e-6, atol=1e-12):
                raise ValueError("Whole-tensor rescaling is outside the research edit rules")
        clean[name] = value.detach().cpu().clone()
        changed += count
        byte_estimate += count * (8 + 2 * original.element_size()) + 1024
        stats.append(
            {
                "name": name,
                "changed_entries": count,
                "delta_norm": delta_norm,
                "relative_norm": relative,
            }
        )
    if changed > min(
        copy.policy.max_changed_entries,
        math.floor(copy.total_parameters * copy.policy.max_edited_fraction),
    ):
        raise ValueError("Global changed-entry/fraction budget exceeded (after dtype rounding)")
    if byte_estimate > copy.policy.max_patch_bytes:
        raise ValueError("Estimated multi-tensor patch byte budget exceeded")
    return clean, {
        "tensors": stats,
        "changed_entries": changed,
        "fraction_of_model": changed / copy.total_parameters,
        "estimated_patch_bytes": byte_estimate,
    }


def build_program(copy, operations, *, replace=False):
    if not isinstance(operations, list) or not 1 <= len(operations) <= 128:
        raise ValueError("A program needs 1-128 data-only operations")
    staged = {} if replace else dict(copy.candidate)
    resolved = []
    for operation in operations:
        if not isinstance(operation, dict) or not isinstance(operation.get("name"), str):
            raise ValueError("Each operation requires a canonical parameter name")
        name, kind = operation["name"], operation.get("op")
        editable_parameter(copy.model, name, copy.policy)
        original = copy.current(name, staged)
        value = original.double().numpy().copy()
        info = {"name": name, "op": kind}
        common = {"name", "op"}
        if kind == "project":
            _keys(
                operation,
                common | {"direction", "strength"},
                {"columns", "top_columns", "max_entries"},
            )
            if value.ndim != 2:
                raise ValueError("Projection requires a matrix")
            raw = operation["direction"]
            if isinstance(raw, str):
                if raw not in copy.directions:
                    raise ValueError("Unknown measured direction id")
                raw = copy.directions[raw]
            vector = _array(raw, ndim=1)
            with np.errstate(over="ignore", invalid="ignore"):
                vector_norm = float(np.linalg.norm(vector))
            if (
                vector.shape != (value.shape[0],)
                or not math.isfinite(vector_norm)
                or vector_norm == 0
            ):
                raise ValueError(
                    "Direction must have a finite nonzero norm and one entry per output row"
                )
            number("projection strength", operation["strength"], -8, 8)
            unit = vector / vector_norm
            projection = unit @ value
            if "columns" in operation and "top_columns" in operation:
                raise ValueError("Choose explicit columns OR top_columns")
            if "columns" in operation:
                columns = _indices(operation["columns"], value.shape[1])
            elif "top_columns" in operation:
                integer("top_columns", operation["top_columns"], 1, value.shape[1])
                columns = np.argsort(-np.abs(projection), kind="stable")[: operation["top_columns"]]
            else:
                columns = np.arange(value.shape[1])
            value[:, columns] -= operation["strength"] * np.outer(unit, projection[columns])
            info.update(columns=columns.tolist(), strength=operation["strength"])
        elif kind == "clip_outliers":
            _keys(operation, common | {"factor"})
            number("median clipping factor", operation["factor"], 1, 1000)
            median = float(np.median(np.abs(value)))
            if not median:
                raise ValueError("Median absolute weight is zero; no clipping scale inferred")
            bound = median * operation["factor"]
            value = np.clip(value, -bound, bound)
            info.update(median_absolute=median, threshold=bound)
        elif kind in ("set_entries", "add"):
            _keys(
                operation,
                common | ({"indices", "values"} if kind == "set_entries" else {"value"}),
                {"indices"} if kind == "add" else (),
            )
            indices = (
                _indices(operation["indices"], value.size)
                if "indices" in operation
                else np.arange(value.size)
            )
            if kind == "set_entries":
                new = _array(operation["values"], ndim=1)
                if new.shape != indices.shape:
                    raise ValueError("Set-entry index/value lengths differ")
                value.reshape(-1)[indices] = new
            else:
                number("add value", operation["value"], -1_000_000, 1_000_000)
                value.reshape(-1)[indices] += operation["value"]
        elif kind in ("scale_rows", "scale_columns"):
            _keys(operation, common | {"indices", "factor"})
            if value.ndim != 2:
                raise ValueError("Row/column scaling requires a matrix")
            number("scale factor", operation["factor"], -8, 8)
            axis = 0 if kind == "scale_rows" else 1
            indices = _indices(operation["indices"], value.shape[axis])
            if axis == 0:
                value[indices, :] *= operation["factor"]
            else:
                value[:, indices] *= operation["factor"]
        elif kind == "low_rank_add":
            _keys(operation, common | {"left", "right"})
            left, right = _array(operation["left"], ndim=2), _array(operation["right"], ndim=2)
            if (
                value.ndim != 2
                or not 1 <= left.shape[1] <= 16
                or left.shape[1] != right.shape[0]
                or (left.shape[0], right.shape[1]) != value.shape
            ):
                raise ValueError("Low-rank factors do not match the target or rank budget")
            if left.size + right.size > 65536:
                raise ValueError("Low-rank input factors exceed their element budget")
            value += left @ right
        else:
            raise ValueError("Unknown weight operation; arbitrary Python is not accepted")
        if kind == "project" and "max_entries" in operation:
            integer("max_entries", operation["max_entries"], 1, copy.policy.max_changed_entries)
            before = original.double().numpy()
            delta = (value - before).reshape(-1)
            keep = np.argsort(-np.abs(delta), kind="stable")[: operation["max_entries"]]
            sparse = before.copy().reshape(-1)
            sparse[keep] = value.reshape(-1)[keep]
            value = sparse.reshape(value.shape)
        staged[name] = _round(original, value)
        resolved.append(info)
    clean, stats = validate_candidate(copy, staged)
    return clean, {**stats, "operations": resolved}


def norm_matched_control(copy, *, kind="random_direction", seed=42):
    """Same affected entries/columns, random direction or random delta, matched per tensor.

    Matching is before stored-dtype rounding. Report actual norms rather than claim exact BF16
    equality. A null result is evidence about this control, not a semantic interpretation proof.
    """
    if kind not in ("random_direction", "random_delta"):
        raise ValueError("Control kind must be random_direction or random_delta")
    integer("control seed", seed, 0, 2**63 - 1)
    if not copy.candidate:
        raise ValueError("A nonempty candidate is needed for a norm-matched control")
    rng, result, matches = np.random.default_rng(seed), {}, []
    for name, candidate in copy.candidate.items():
        original = copy.original(name)
        base, target = original.double().numpy(), candidate.double().numpy()
        mask = target != base
        desired = float(np.linalg.norm(target - base))
        if kind == "random_direction":
            if base.ndim != 2:
                raise ValueError(
                    "Random-direction control requires matrix candidates; use random_delta for nonmatrices"
                )
            unit = rng.normal(size=base.shape[0])
            unit /= np.linalg.norm(unit)
            delta = -np.outer(unit, unit @ base)
        else:
            delta = rng.normal(size=base.shape)
        delta = np.where(mask, delta, 0.0)
        norm = float(np.linalg.norm(delta))
        if not norm:
            raise ValueError("Control has zero norm on the selected support")
        value = _round(original, base + delta * (desired / norm))
        actual = float(np.linalg.norm(value.double().numpy() - base))
        result[name] = value
        matches.append(
            {
                "name": name,
                "candidate_delta_norm": desired,
                "control_delta_norm": actual,
                "relative_rounding_error": abs(actual - desired) / desired,
            }
        )
    result, stats = validate_candidate(copy, result)
    return result, {**stats, "kind": kind, "norm_matches": matches}
