"""
Task Arithmetic & Weight Vector Operations.
Implements weight addition, task unlearning, TIES-Merging, and DARE algorithms.
"""
from typing import Dict, Any, List, Optional
import numpy as np

class TaskArithmeticEngine:
    """
    Performs direct linear and non-linear algebra on neural network parameter tensors.
    """

    @staticmethod
    def compute_task_vector(fine_tuned_weights: np.ndarray, base_weights: np.ndarray) -> np.ndarray:
        """Computes tau = theta_ft - theta_base."""
        return fine_tuned_weights - base_weights

    @staticmethod
    def apply_task_vector(base_weights: np.ndarray, task_vector: np.ndarray, scaling: float = 1.0) -> np.ndarray:
        """Applies task vector: theta_new = theta_base + lambda * tau."""
        return base_weights + (scaling * task_vector)

    @staticmethod
    def unlearn_behavior(base_weights: np.ndarray, negative_task_vector: np.ndarray, beta: float = 0.5) -> np.ndarray:
        """Removes undesirable capabilities: theta_clean = theta_base - beta * tau_negative."""
        return base_weights - (beta * negative_task_vector)

    @staticmethod
    def dare_merge(
        base_weights: np.ndarray,
        task_vectors: List[np.ndarray],
        drop_rate: float = 0.9,
        scale: float = 1.0
    ) -> np.ndarray:
        """
        Drop And REscale (DARE): Sparsifies weight deltas randomly and rescales remainder.
        Delta_DARE = (m * Delta) / (1 - drop_rate)
        """
        merged_delta = np.zeros_like(base_weights)
        keep_prob = 1.0 - drop_rate

        for tv in task_vectors:
            # Generate Bernoulli mask
            mask = np.random.binomial(1, keep_prob, size=tv.shape).astype(np.float32)
            rescaled_tv = (mask * tv) / keep_prob
            merged_delta += rescaled_tv

        avg_delta = (merged_delta / len(task_vectors)) * scale
        return base_weights + avg_delta

    @staticmethod
    def slerp_interpolate(v0: np.ndarray, v1: np.ndarray, t: float, eps: float = 1e-8) -> np.ndarray:
        """
        Spherical Linear Interpolation (SLERP) between two weight vectors.
        """
        flat0 = v0.flatten()
        flat1 = v1.flatten()
        
        norm0 = np.linalg.norm(flat0)
        norm1 = np.linalg.norm(flat1)
        
        if norm0 == 0 or norm1 == 0:
            return (1 - t) * v0 + t * v1
            
        u0 = flat0 / norm0
        u1 = flat1 / norm1
        
        dot = np.dot(u0, u1)
        dot = np.clip(dot, -1.0, 1.0)
        
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        
        if sin_theta_0 < eps:
            return (1 - t) * v0 + t * v1
            
        theta_t = theta_0 * t
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = np.sin(theta_t) / sin_theta_0
        
        res = (s0 * norm0 * u0) + (s1 * norm1 * u1)
        return res.reshape(v0.shape)
