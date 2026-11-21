"""Small NumPy adapter utility. For real language-model training use the Transformers backend."""

import json
from pathlib import Path

import numpy as np

from ..config import TuningConfig
from .checkpoints import atomic_json


class AdapterEngine:
    def __init__(self, config=None, seed=42):
        self.config = config or TuningConfig()
        self.active_adapters = {}
        self.rng = np.random.default_rng(seed)

    def initialize_lora_layers(self, layer_names, in_features, out_features):
        if (
            min(in_features, out_features) < 1
            or not layer_names
            or len(set(layer_names)) != len(layer_names)
        ):
            raise ValueError("Unique layer names and positive dimensions are required")
        r = self.config.lora_r
        self.active_adapters = {
            name: {
                "lora_A": self.rng.normal(0, 1 / np.sqrt(r), (r, in_features)).astype(np.float32),
                "lora_B": np.zeros((out_features, r), dtype=np.float32),
                "alpha": self.config.lora_alpha,
                "r": r,
            }
            for name in layer_names
        }
        return self.active_adapters

    def train_step(self, layer_name, grad_b, grad_a, lr=None):
        """Apply caller-supplied gradients; return their squared norm, NOT a training loss."""
        rate = self.config.learning_rate if lr is None else lr
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("Learning rate must be finite and nonnegative")
        layer = self.active_adapters[layer_name]
        for name, grad in (("lora_A", grad_a), ("lora_B", grad_b)):
            if layer[name].shape != grad.shape or not np.isfinite(grad).all():
                raise ValueError("Gradient shape/value mismatch")
        layer["lora_A"] -= rate * grad_a
        layer["lora_B"] -= rate * grad_b
        return float((grad_b**2).sum() + (grad_a**2).sum())

    def save_adapter_checkpoint(self, path, metadata=None):
        if not self.active_adapters:
            raise ValueError("No adapters to save")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tensors, layers = {}, []
        for i, (name, layer) in enumerate(self.active_adapters.items()):
            layers.append({"name": name, "index": i, "alpha": layer["alpha"], "r": layer["r"]})
            for key in ("lora_A", "lora_B"):
                tensors[f"{i}_{key}"] = layer[key]
        np.savez_compressed(path / "adapter_weights.npz", **tensors)
        atomic_json(
            path / "adapter_config.json",
            {"format": "ouroboros-numpy-adapter-v1", "layers": layers, "metadata": metadata or {}},
        )

    def load_adapter_checkpoint(self, path):
        path = Path(path)
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        if config.get("format") != "ouroboros-numpy-adapter-v1":
            raise ValueError("Not a NumPy adapter checkpoint")
        loaded = {}
        with np.load(path / "adapter_weights.npz", allow_pickle=False) as weights:
            for layer in config["layers"]:
                tensors = {
                    key: weights[f"{layer['index']}_{key}"].copy() for key in ("lora_A", "lora_B")
                }
                if not all(np.isfinite(w).all() for w in tensors.values()):
                    raise ValueError("Nonfinite adapter checkpoint")
                if (
                    tensors["lora_A"].shape[0] != layer["r"]
                    or tensors["lora_B"].shape[1] != layer["r"]
                ):
                    raise ValueError("Adapter rank mismatch")
                loaded[layer["name"]] = {**tensors, "alpha": layer["alpha"], "r": layer["r"]}
        self.active_adapters = loaded
        return config.get("metadata", {})
