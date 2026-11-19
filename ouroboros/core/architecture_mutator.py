"""Explicit architecture candidates and array-level steering/pruning utilities."""

import copy
import math

import numpy as np


class ArchitectureMutator:
    @staticmethod
    def layer_stack_blueprint(base_layer_count, repeat_indices):
        if not isinstance(base_layer_count, int) or base_layer_count < 1:
            raise ValueError("Positive integer layer count required")
        if any(not isinstance(i, int) or not 0 <= i < base_layer_count for i in repeat_indices):
            raise ValueError("Repeat index outside the original layer range")
        sequence = []
        for i in range(base_layer_count):
            sequence.extend([i] * (1 + repeat_indices.count(i)))
        return sequence

    @staticmethod
    def build_layer_stacked_candidate(model, repeat_indices):
        """Copy a Llama/Qwen-style model, duplicate layers, and adjust cache indices/config.

        Supports ordinary nonquantized `model.layers` decoder stacks only. Requires enough
        memory for a full copy; never used implicitly by the recursive loop. Evaluate before use.
        """
        import torch

        if getattr(model, "is_quantized", False):
            raise ValueError("Structural copies of quantized models are not supported")
        if not hasattr(model, "model") or not isinstance(
            getattr(model.model, "layers", None), torch.nn.ModuleList
        ):
            raise ValueError("Expected a Llama/Qwen-style model.model.layers ModuleList")
        layers = model.model.layers
        if len(repeat_indices) > len(layers):
            raise ValueError("A candidate may at most double the layer count")
        blueprint = ArchitectureMutator.layer_stack_blueprint(len(layers), repeat_indices)
        candidate = copy.deepcopy(model)
        candidate.model.layers = torch.nn.ModuleList([copy.deepcopy(layers[i]) for i in blueprint])
        for i, layer in enumerate(candidate.model.layers):
            for module in layer.modules():
                if hasattr(module, "layer_idx"):
                    module.layer_idx = i
        candidate.config.num_hidden_layers = len(blueprint)
        candidate.model.config.num_hidden_layers = len(blueprint)
        return candidate

    @staticmethod
    def scale_attention_heads(qkv_weights, num_heads, head_scales):
        if qkv_weights.ndim != 2 or num_heads < 1 or qkv_weights.shape[0] % num_heads:
            raise ValueError("Rows must be divisible into a positive number of heads")
        if any(
            not 0 <= i < num_heads or not math.isfinite(scale) for i, scale in head_scales.items()
        ):
            raise ValueError("Invalid head index or scale")
        modified = qkv_weights.copy()
        head_dim = qkv_weights.shape[0] // num_heads
        for i, scale in head_scales.items():
            modified[i * head_dim : (i + 1) * head_dim] *= scale
        return modified

    @staticmethod
    def compute_steering_vector(positive_activations, negative_activations):
        if (
            positive_activations.ndim != 2
            or negative_activations.ndim != 2
            or (positive_activations.shape[1] != negative_activations.shape[1])
            or not len(positive_activations)
            or not len(negative_activations)
        ):
            raise ValueError(
                "Nonempty activation matrices with matching feature dimensions required"
            )
        if (
            not np.isfinite(positive_activations).all()
            or not np.isfinite(negative_activations).all()
        ):
            raise ValueError("Activations must be finite")
        vector = positive_activations.mean(axis=0) - negative_activations.mean(axis=0)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    @staticmethod
    def prune_least_significant_neurons(weight_matrix, prune_ratio=0.1):
        """Legacy name: unstructured weight pruning, NOT neuron removal or a speed guarantee."""
        if (
            not np.isfinite(weight_matrix).all()
            or not math.isfinite(prune_ratio)
            or not 0 <= prune_ratio <= 1
        ):
            raise ValueError("Finite weights and prune_ratio in [0, 1] required")
        result = weight_matrix.copy()
        k = math.floor(result.size * prune_ratio)
        ids = np.argsort(np.abs(result).reshape(-1), kind="stable")[:k]
        result.reshape(-1)[ids] = 0
        return result
