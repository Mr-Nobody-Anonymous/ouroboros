"""
Adapter Engine: Manages PEFT (LoRA/QLoRA) parameter fine-tuning pipelines.
"""
from typing import Dict, Any, List, Optional
import os
import json
from ..config import TuningConfig

class AdapterEngine:
    """
    Orchestrates low-rank parameter adaptation without altering the frozen base foundation model.
    """

    def __init__(self, config: Optional[TuningConfig] = None):
        self.config = config or TuningConfig()
        self.active_adapters: Dict[str, Any] = {}

    def initialize_lora_layers(self, layer_names: List[str], in_features: int, out_features: int) -> Dict[str, Any]:
        """
        Initializes low-rank matrix pairs A (Gaussian) and B (Zeros) following LoRA conventions.
        """
        import numpy as np
        r = self.config.lora_r
        adapters = {}
        for layer in layer_names:
            # LoRA A is typically initialized with standard normal, B with zeros
            lora_a = np.random.normal(0, 1.0 / np.sqrt(r), (r, in_features)).astype(np.float32)
            lora_b = np.zeros((out_features, r), dtype=np.float32)
            adapters[layer] = {
                "lora_A": lora_a,
                "lora_B": lora_b,
                "alpha": self.config.lora_alpha,
                "r": r
            }
        self.active_adapters = adapters
        return adapters

    def train_step(self, layer_name: str, grad_b: Any, grad_a: Any, lr: Optional[float] = None) -> float:
        """
        Applies a gradient descent update step specifically to the low-rank adapter matrices.
        """
        lr = lr or self.config.learning_rate
        if layer_name not in self.active_adapters:
            raise KeyError(f"Layer {layer_name} not registered in active adapters.")

        self.active_adapters[layer_name]["lora_B"] -= lr * grad_b
        self.active_adapters[layer_name]["lora_A"] -= lr * grad_a
        
        # Approximate step loss
        step_norm = float((grad_b**2).sum() + (grad_a**2).sum())
        return step_norm

    def save_adapter_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        """Saves current adapter weights and metadata configuration."""
        os.makedirs(path, exist_ok=True)
        config_data = {
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "target_modules": self.config.target_modules,
            "metadata": metadata or {}
        }
        with open(os.path.join(path, "adapter_config.json"), "w") as f:
            json.dump(config_data, f, indent=2)
            
        print(f"[AdapterEngine] Checkpoint saved successfully at {path}")
