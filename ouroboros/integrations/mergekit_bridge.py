"""Validated mergekit recipes and explicit opt-in execution in a separate environment."""

import math
import shutil
import subprocess
from pathlib import Path

import yaml


class MergeKitBridge:
    @property
    def is_available(self):
        return shutil.which("mergekit-yaml") is not None

    @staticmethod
    def _write(recipe, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
        return str(path)

    def generate_slerp_recipe(
        self, base_model_a, base_model_b, output_yaml_path, interpolation_t=0.5
    ):
        if not math.isfinite(interpolation_t) or not 0 <= interpolation_t <= 1:
            raise ValueError("interpolation_t must be in [0, 1]")
        # Whole-model form has no hardcoded assumption that every architecture has 32 layers.
        return self._write(
            {
                "models": [{"model": base_model_a}, {"model": base_model_b}],
                "merge_method": "slerp",
                "base_model": base_model_a,
                "parameters": {"t": interpolation_t},
                "dtype": "bfloat16",
            },
            output_yaml_path,
        )

    def generate_dare_ties_recipe(
        self, base_model, expert_models, output_yaml_path, density=0.6, weight=0.5
    ):
        if (
            not expert_models
            or not math.isfinite(density)
            or not 0 < density <= 1
            or not math.isfinite(weight)
        ):
            raise ValueError("Provide experts, density in (0, 1], and a finite weight")
        return self._write(
            {
                "models": [
                    {"model": model, "parameters": {"density": density, "weight": weight}}
                    for model in expert_models
                ],
                "merge_method": "dare_ties",
                "base_model": base_model,
                "dtype": "bfloat16",
            },
            output_yaml_path,
        )

    def execute(self, recipe_path, output_path, *, allow_execution=False, timeout=300):
        if not allow_execution:
            raise PermissionError(
                "Set allow_execution=True to run a merge; recipe creation does not merge weights"
            )
        executable = shutil.which("mergekit-yaml")
        if not executable:
            raise RuntimeError("Install mergekit in its own environment; no merge was run")
        recipe = Path(recipe_path).resolve()
        if not recipe.is_file():
            raise ValueError("Merge recipe not found")
        output = Path(output_path).resolve()
        if output.exists():
            raise FileExistsError("Refusing to overwrite existing merge output")
        # No shell interpolation. Models in the recipe must be local when operating offline.
        result = subprocess.run(
            [executable, str(recipe), str(output)],
            check=True,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {"status": "executed", "stdout": result.stdout, "validated_improvement": False}
