"""Explicit EasyEdit invocation; availability is never confused with a successful edit."""

import importlib
import importlib.util
from pathlib import Path


class EasyEditBridge:
    METHODS = {
        "ROME": "ROMEHyperParams",
        "MEMIT": "MEMITHyperParams",
        "MEND": "MENDHyperParams",
        "FT": "FTHyperParams",
    }

    def __init__(self, method_name="ROME"):
        if method_name not in self.METHODS:
            raise ValueError(f"Supported methods: {sorted(self.METHODS)}")
        self.method_name = method_name

    @property
    def is_available(self):
        return importlib.util.find_spec("easyeditor") is not None

    def execute_direct_edit(self, model_name, target_fact, hparams=None):
        """Return actual editor metrics/model/original weights, or raise an actionable error.

        EasyEdit must be installed in a dedicated compatible environment. This call may
        load model/statistics artifacts according to the USER-SUPPLIED upstream config.
        """
        if not self.is_available:
            raise RuntimeError(
                "EasyEdit is not installed. Initialize external/EasyEdit and follow "
                "its setup in an isolated environment; no edit was applied."
            )
        path = Path(hparams) if isinstance(hparams, (str, Path)) else None
        if path is None or not path.is_file():
            raise ValueError("Provide an existing upstream EasyEdit hyperparameter YAML path")
        if not all(
            isinstance(target_fact.get(k), str) and target_fact[k].strip()
            for k in ("prompt", "target_new", "subject")
        ):
            raise ValueError("target_fact requires prompt, target_new, and subject strings")
        easyeditor = importlib.import_module("easyeditor")
        params = getattr(easyeditor, self.METHODS[self.method_name]).from_hparams(str(path))
        params.model_name = model_name
        editor = easyeditor.BaseEditor.from_hparams(params)
        metrics, edited_model, original_weights = editor.edit(
            prompts=[target_fact["prompt"]],
            target_new=[target_fact["target_new"]],
            subject=[target_fact["subject"]],
            keep_original_weight=True,
        )
        return {
            "status": "executed",
            "method": self.method_name,
            "metrics": metrics,
            "edited_model": edited_model,
            "original_weights": original_weights,
            "validated_improvement": False,
        }
