"""
Weight-Space Programmer: Implements zero-gradient direct neural weight surgery
inspired by 'Sparks of Weight-Space Programming' (Takehi, Teoh, Zhang, Khattab, MIT CSAIL 2026).
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np

class WeightSpaceProgrammer:
    """
    Enables models to natively inspect and edit raw parameter matrices,
    performing directional ablation on degenerate attractor states (such as repetitive reasoning loops)
    without gradient descent or backpropagation.
    """

    def __init__(self, target_sparsity: float = 0.00016):
        # Default sparsity corresponds to editing 0.016% of weights
        self.target_sparsity = target_sparsity

    def diagnose_repetitive_attractor(self, down_proj_weight: np.ndarray) -> np.ndarray:
        """
        Uses SVD to identify the primary directional vector causing repetitive reasoning loops.
        """
        # Compute economy SVD
        u, s, vt = np.linalg.svd(down_proj_weight, full_matrices=False)
        # Dominant singular vector corresponding to strongest attractor direction
        dominant_direction = u[:, 0:1]
        return dominant_direction

    def surgical_directional_ablation(
        self,
        weight_matrix: np.ndarray,
        attractor_vector: np.ndarray,
        ablation_strength: float = 0.5
    ) -> np.ndarray:
        """
        Ablates the projection along the degenerate attractor subspace:
        W' = W - eta * (u @ u.T @ W)
        """
        u = attractor_vector / np.linalg.norm(attractor_vector)
        projection = (u @ u.T) @ weight_matrix
        ablated_weights = weight_matrix - (ablation_strength * projection)
        return ablated_weights

    def sparse_topk_edit(
        self,
        base_weights: np.ndarray,
        target_weights: np.ndarray,
        sparsity_ratio: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies weight updates strictly to the top K most impactful parameters (e.g. 0.016%),
        zeroing out deltas across the remaining 99.984% of weights to preserve unedited behaviors.
        """
        ratio = sparsity_ratio or self.target_sparsity
        delta = target_weights - base_weights
        flat_abs_delta = np.abs(delta).flatten()
        
        # Calculate Top-K threshold
        k = max(1, int(flat_abs_delta.size * ratio))
        threshold = np.partition(flat_abs_delta, -k)[-k]
        
        # Binary mask for top 0.016% weights
        mask = np.abs(delta) >= threshold
        sparse_delta = np.where(mask, delta, 0.0)
        
        final_weights = base_weights + sparse_delta
        
        stats = {
            "total_weights": int(base_weights.size),
            "edited_weights": int(np.count_nonzero(mask)),
            "actual_sparsity_percentage": float(np.count_nonzero(mask) / base_weights.size) * 100,
            "frobenius_norm_delta": float(np.linalg.norm(sparse_delta))
        }
        return final_weights, stats
