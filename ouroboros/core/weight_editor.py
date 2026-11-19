"""
Weight Editor: Low-level tensor inspection, direct weight surgery, and delta merging.
"""
import math
from typing import Dict, Any, Tuple, Optional
import numpy as np

class WeightEditor:
    """
    Directly modifies model weight tensors or performs surgical delta injections.
    Supports SVD-based low-rank projection, weight clipping, and adapter merging.
    """

    def __init__(self, model: Optional[Any] = None):
        self.model = model

    def inspect_matrix_stats(self, matrix: np.ndarray) -> Dict[str, float]:
        """Calculates key mathematical statistics of a weight matrix."""
        frobenius_norm = float(np.linalg.norm(matrix))
        spectral_norm = float(np.linalg.norm(matrix, 2)) if matrix.ndim == 2 else 0.0
        mean = float(np.mean(matrix))
        std = float(np.std(matrix))
        sparsity = float(np.count_nonzero(matrix == 0) / matrix.size)
        
        return {
            "frobenius_norm": frobenius_norm,
            "spectral_norm": spectral_norm,
            "mean": mean,
            "std": std,
            "sparsity": sparsity,
            "shape": list(matrix.shape),
        }

    def compute_lora_delta(self, lora_a: np.ndarray, lora_b: np.ndarray, alpha: float, r: int) -> np.ndarray:
        """
        Computes the low-rank delta weight matrix:
        Delta_W = (alpha / r) * (B @ A)
        """
        scaling = alpha / float(r)
        delta_w = scaling * (lora_b @ lora_a)
        return delta_w

    def surgical_rank_one_edit(
        self,
        base_weights: np.ndarray,
        key_vector: np.ndarray,
        value_vector: np.ndarray,
        lambda_reg: float = 1e-4
    ) -> np.ndarray:
        """
        Performs direct rank-one weight editing (e.g., ROME-style):
        Delta_W = (v - W0 @ k) @ (k^T / (k^T @ k + lambda))
        """
        k = key_vector.reshape(-1, 1)
        v = value_vector.reshape(-1, 1)
        
        current_pred = base_weights @ k
        residual = v - current_pred
        norm_factor = float(k.T @ k) + lambda_reg
        delta_w = (residual @ k.T) / norm_factor
        
        return base_weights + delta_w

    def merge_and_clamp(
        self,
        base_weights: np.ndarray,
        delta: np.ndarray,
        max_deviation_ratio: float = 0.15
    ) -> np.ndarray:
        """
        Merges delta weights into base weights with norm-bounding to prevent catastrophic collapse.
        """
        base_norm = np.linalg.norm(base_weights)
        delta_norm = np.linalg.norm(delta)
        max_allowed_delta = base_norm * max_deviation_ratio
        
        if delta_norm > max_allowed_delta and delta_norm > 0:
            scale = max_allowed_delta / delta_norm
            delta = delta * scale
            
        updated_weights = base_weights + delta
        return updated_weights
