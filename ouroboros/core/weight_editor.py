"""Finite, shape-checked tensor inspection, reference rank-one updates, and norm-bounded edits."""

import math

import numpy as np

from .task_arithmetic import compatible


class WeightEditor:
    def __init__(self, model=None):
        self.model = model

    def inspect_matrix_stats(self, matrix):
        compatible(matrix)
        return {
            "frobenius_norm": float(np.linalg.norm(matrix)),
            "spectral_norm": float(np.linalg.norm(matrix, 2)) if matrix.ndim == 2 else 0.0,
            "mean": float(np.mean(matrix)),
            "std": float(np.std(matrix)),
            "sparsity": float(np.count_nonzero(matrix == 0) / matrix.size),
            "shape": list(matrix.shape),
        }

    def compute_lora_delta(self, lora_a, lora_b, alpha, r):
        if not isinstance(r, int) or r < 1 or not math.isfinite(alpha):
            raise ValueError("Positive integer rank and finite alpha required")
        for tensor in (lora_a, lora_b):
            compatible(tensor)
            if tensor.ndim != 2:
                raise ValueError("LoRA weights must be matrices")
        if lora_a.shape[0] != r or lora_b.shape[1] != r:
            raise ValueError("LoRA tensor dimensions do not match the rank")
        return (alpha / r) * (lora_b @ lora_a)

    def surgical_rank_one_edit(self, base_weights, key_vector, value_vector, lambda_reg=1e-4):
        """Regularized least-squares rank-one update, NOT a full ROME implementation.

        ROME additionally requires layer/key/value selection and covariance statistics.
        """
        compatible(base_weights)
        if base_weights.ndim != 2 or not math.isfinite(lambda_reg) or lambda_reg < 0:
            raise ValueError("Expected a weight matrix and nonnegative regularization")
        k, v = np.asarray(key_vector).reshape(-1, 1), np.asarray(value_vector).reshape(-1, 1)
        if k.shape[0] != base_weights.shape[1] or v.shape[0] != base_weights.shape[0]:
            raise ValueError("Key/value dimensions do not match the weight matrix")
        if not np.isfinite(k).all() or not np.isfinite(v).all():
            raise ValueError("Key and value must be finite")
        denominator = (k.T @ k).item() + lambda_reg
        if denominator <= 0:
            raise ValueError("Zero key without regularization")
        return base_weights + ((v - base_weights @ k) @ k.T) / denominator

    def merge_and_clamp(self, base_weights, delta, max_deviation_ratio=0.15):
        """Bound update norm; does not guarantee behavioral preservation, especially at zero base."""
        compatible(base_weights, delta)
        if not math.isfinite(max_deviation_ratio) or max_deviation_ratio < 0:
            raise ValueError("Deviation ratio must be finite and nonnegative")
        allowed = np.linalg.norm(base_weights) * max_deviation_ratio
        norm = np.linalg.norm(delta)
        return base_weights + delta * (min(1.0, allowed / norm) if norm else 1.0)
