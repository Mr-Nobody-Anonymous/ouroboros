"""
MergeKit Bridge: Integrates Ouroboros with the MergeKit weight merging framework.
"""
import os
import sys
import yaml
from typing import Dict, Any, List, Optional

MERGEKIT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "external", "mergekit"))
if os.path.exists(MERGEKIT_PATH) and MERGEKIT_PATH not in sys.path:
    sys.path.insert(0, MERGEKIT_PATH)

class MergeKitBridge:
    """
    Interfaces Ouroboros directly with MergeKit configuration generators
    for TIES, SLERP, DARE, and Passthrough (Frankenmerge) execution.
    """

    def __init__(self):
        self.is_available = os.path.exists(MERGEKIT_PATH)

    def generate_slerp_recipe(
        self,
        base_model_a: str,
        base_model_b: str,
        output_yaml_path: str,
        interpolation_t: float = 0.5
    ) -> str:
        """Generates a SLERP merge configuration for MergeKit."""
        recipe = {
            "slices": [
                {
                    "sources": [
                        {"model": base_model_a, "layer_range": [0, 32]},
                        {"model": base_model_b, "layer_range": [0, 32]}
                    ]
                }
            ],
            "merge_method": "slerp",
            "base_model": base_model_a,
            "parameters": {
                "t": [
                    {"filter": "self_attn", "value": [0.0, 0.5, 0.3, 0.7, 1.0]},
                    {"filter": "mlp", "value": [1.0, 0.5, 0.7, 0.3, 0.0]},
                    {"value": interpolation_t}
                ]
            },
            "dtype": "bfloat16"
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_yaml_path)), exist_ok=True)
        with open(output_yaml_path, "w") as f:
            yaml.dump(recipe, f, default_flow_style=False)

        return output_yaml_path

    def generate_dare_ties_recipe(
        self,
        base_model: str,
        expert_models: List[str],
        output_yaml_path: str,
        density: float = 0.6,
        weight: float = 0.5
    ) -> str:
        """Generates a DARE-TIES multi-model consensus merge recipe."""
        models_block = [{"model": base_model}]
        for m in expert_models:
            models_block.append({
                "model": m,
                "parameters": {
                    "density": density,
                    "weight": weight
                }
            })

        recipe = {
            "models": models_block,
            "merge_method": "dare_ties",
            "base_model": base_model,
            "parameters": {
                "int8_mask": True
            },
            "dtype": "bfloat16"
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_yaml_path)), exist_ok=True)
        with open(output_yaml_path, "w") as f:
            yaml.dump(recipe, f, default_flow_style=False)

        return output_yaml_path
