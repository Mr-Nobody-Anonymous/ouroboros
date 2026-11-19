"""
Architecture Mutator: Structural alterations to neural topologies and internal mechanisms.
"""
from typing import Dict, Any, List, Optional
import numpy as np

class ArchitectureMutator:
    """
    Mutates model architecture: layer stacking, attention head scaling,
    activation steering vector injection, and selective neuron pruning.
    """

    @staticmethod
    def layer_stack_blueprint(base_layer_count: int, repeat_indices: List[int]) -> List[int]:
        """
        Creates an expanded layer routing blueprint (e.g. Solar 10.7B / Frankenmerging style).
        E.g. [0..23, 16..31] turns 32 layers into 40 layers.
        """
        sequence = list(range(base_layer_count))
        for idx in repeat_indices:
            sequence.insert(idx + 1, idx)
        return sequence

    @staticmethod
    def scale_attention_heads(qkv_weights: np.ndarray, num_heads: int, head_scales: Dict[int, float]) -> np.ndarray:
        """
        Reweights specific attention heads to amplify reasoning attention and attenuate distractions.
        """
        modified = qkv_weights.copy()
        head_dim = qkv_weights.shape[0] // num_heads
        
        for head_idx, scale in head_scales.items():
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            modified[start:end, :] *= scale
            
        return modified

    @staticmethod
    def compute_steering_vector(positive_activations: np.ndarray, negative_activations: np.ndarray) -> np.ndarray:
        """
        Computes an activation steering vector (Representation Engineering):
        v_steer = mean(Act_pos) - mean(Act_neg)
        Can be added to residual stream to guide generation towards truthfulness or reasoning.
        """
        mean_pos = np.mean(positive_activations, axis=0)
        mean_neg = np.mean(negative_activations, axis=0)
        steering_vec = mean_pos - mean_neg
        norm = np.linalg.norm(steering_vec)
        if norm > 0:
            steering_vec = steering_vec / norm
        return steering_vec

    @staticmethod
    def prune_least_significant_neurons(weight_matrix: np.ndarray, prune_ratio: float = 0.1) -> np.ndarray:
        """
        Prunes weights with lowest absolute magnitude in MLP matrices to improve sparsity and compute efficiency.
        """
        pruned = weight_matrix.copy()
        threshold = np.percentile(np.abs(pruned), prune_ratio * 100)
        pruned[np.abs(pruned) < threshold] = 0.0
        return pruned
