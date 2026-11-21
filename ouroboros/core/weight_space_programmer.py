"""Experimental, bounded matrix edits. These utilities do not reproduce the Astra experiment.

A singular vector of a weight matrix alone is NOT evidence of a repetition mechanism.
Prefer contrastive activation measurements and independent held-out evaluation.
"""

import math
import warnings

import numpy as np


def _matrix(value):
    value = np.asarray(value)
    if value.ndim != 2 or not value.size or not np.issubdtype(value.dtype, np.floating):
        raise ValueError("Expected a nonempty floating-point matrix")
    if not np.isfinite(value).all():
        raise ValueError("Weights must be finite")
    return value


class WeightSpaceProgrammer:
    def __init__(self, target_sparsity=0.00016):
        self._ratio(target_sparsity)
        # Retained name for compatibility: the value is the EDITED fraction, not zero sparsity.
        self.target_sparsity = target_sparsity

    @staticmethod
    def _ratio(value):
        if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
            raise ValueError("Edited fraction must be finite and in [0, 1]")

    def dominant_weight_direction(self, weight):
        u, _, _ = np.linalg.svd(_matrix(weight), full_matrices=False)
        return u[:, :1]

    def diagnose_repetitive_attractor(self, down_proj_weight):
        warnings.warn(
            "SVD identifies a weight direction, not a diagnosed repetitive attractor. "
            "Use measured contrastive activations and evaluate edits.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dominant_weight_direction(down_proj_weight)

    @staticmethod
    def contrastive_direction(failing_activations, successful_activations):
        failing = _matrix(failing_activations)
        successful = _matrix(successful_activations)
        if failing.shape[1] != successful.shape[1]:
            raise ValueError("Activation feature dimensions differ")
        direction = failing.mean(axis=0) - successful.mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("No measured contrastive direction")
        return (direction / norm).reshape(-1, 1)

    def surgical_directional_ablation(self, weight_matrix, attractor_vector, ablation_strength=0.5):
        weight = _matrix(weight_matrix)
        vector = np.asarray(attractor_vector)
        if vector.shape not in {(weight.shape[0],), (weight.shape[0], 1)}:
            raise ValueError("Direction must have one entry per output row")
        if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError("Direction must be finite and nonzero")
        self._ratio(ablation_strength)
        u = vector.reshape(-1, 1) / np.linalg.norm(vector)
        return weight - ablation_strength * (u @ (u.T @ weight))

    def sparse_topk_edit(self, base_weights, target_weights, sparsity_ratio=None):
        base, target = _matrix(base_weights), _matrix(target_weights)
        if base.shape != target.shape:
            raise ValueError("Base and target shapes must match; broadcasting is not an edit")
        ratio = self.target_sparsity if sparsity_ratio is None else sparsity_ratio
        self._ratio(ratio)
        delta = target - base
        # Floor enforces an upper bound. A tiny matrix may legitimately have a zero edit budget.
        k = min(base.size, math.floor(base.size * ratio))
        flat = delta.reshape(-1)
        # Copy unselected values, not base + zeros: adding +0 can destroy a stored -0 bit.
        # Use C order so flat indexed writes also work with Fortran/noncontiguous inputs.
        final = np.array(base, dtype=np.result_type(base.dtype, target.dtype), order="C", copy=True)
        if k:
            # Stable ordering breaks ties by index; threshold masks could edit every tied parameter.
            indices = np.argsort(-np.abs(flat), kind="stable")[:k]
            values = target.reshape(-1)[indices]
            changed = values != final.reshape(-1)[indices]
            final.reshape(-1)[indices[changed]] = values[changed]
        edited = int(np.count_nonzero(final != base))
        return final, {
            "total_weights": int(base.size),
            "edited_weights": edited,
            "edit_budget": k,
            "actual_sparsity_percentage": 100 * edited / base.size,
            "frobenius_norm_delta": float(np.linalg.norm(final - base)),
        }

    def apply_to_torch_model(
        self, model, parameter_name, direction, *, strength=0.5, edited_fraction=None
    ):
        """Explicitly apply a bounded candidate edit to ONE nonquantized tensor; return rollback data.

        No semantic diagnosis is performed. The caller must evaluate and either keep or restore it.
        """
        import torch

        parameters = dict(model.named_parameters())
        if parameter_name not in parameters:
            raise KeyError(parameter_name)
        parameter = parameters[parameter_name]
        if parameter.ndim != 2 or parameter.dtype not in {
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        }:
            raise ValueError("Only ordinary floating-point 2D parameters are supported")
        original = parameter.detach().cpu().clone()
        # Preserve F64 exactly; promote BF16/F16 weights through F32 before one final cast.
        # A higher-precision caller direction may promote intermediate NumPy arithmetic further.
        base = original.numpy() if original.dtype == torch.float64 else original.float().numpy()
        candidate = self.surgical_directional_ablation(base, direction, strength)
        edited, stats = self.sparse_topk_edit(base, candidate, edited_fraction)
        candidate_tensor = torch.as_tensor(edited, dtype=original.dtype)
        # A rounded-away edit must not flip signed zeros or otherwise change equal-value bytes.
        candidate_tensor = torch.where(candidate_tensor == original, original, candidate_tensor)
        if not torch.isfinite(candidate_tensor).all():
            raise FloatingPointError("Nonfinite rounded edit; model was not changed")
        delta_norm = float(torch.linalg.vector_norm(candidate_tensor.double() - original.double()))
        base_norm = float(torch.linalg.vector_norm(original.double()))
        stats.update(
            edited_weights=int(torch.count_nonzero(candidate_tensor != original)),
            frobenius_norm_delta=delta_norm,
            relative_norm_delta=delta_norm / base_norm if base_norm else 0.0,
        )
        stats["actual_sparsity_percentage"] = 100 * stats["edited_weights"] / original.numel()
        with torch.no_grad():
            parameter.copy_(candidate_tensor.to(parameter.device))
        return {"parameter_name": parameter_name, "original": original, "stats": stats}

    @staticmethod
    def restore_torch_edit(model, edit):
        import torch

        parameter = dict(model.named_parameters())[edit["parameter_name"]]
        if parameter.shape != edit["original"].shape:
            raise ValueError("Rollback tensor shape mismatch")
        with torch.no_grad():
            parameter.copy_(edit["original"].to(parameter.device, dtype=parameter.dtype))
