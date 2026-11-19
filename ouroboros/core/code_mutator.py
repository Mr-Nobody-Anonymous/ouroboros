"""
Autonomous Code Mutator: Self-directing code and hyperdirective modification
inspired by MIT SEAL (Self-Adapting Language Models) and The AI Scientist.
"""
from typing import Dict, Any, List, Optional
import re

class CodeMutator:
    """
    Allows the model to propose syntactic code modifications to its own training loop,
    loss formulations (e.g. DPO temperature beta), or learning rate decay schedules.
    """

    def __init__(self, target_filepath: Optional[str] = None):
        self.target_filepath = target_filepath
        self.mutation_log: List[Dict[str, Any]] = []

    def propose_loss_mutation(self, current_code: str, adjustment_type: str, new_value: float) -> str:
        """
        Synthesizes a precise code diff modifying loss scaling hyperparameters.
        """
        if adjustment_type == "dpo_beta":
            pattern = r"(beta\s*=\s*)([0-9.]+)"
            replacement = rf"\g<1>{new_value:.3f}"
            mutated = re.sub(pattern, replacement, current_code)
        elif adjustment_type == "learning_rate":
            pattern = r"(learning_rate\s*:\s*float\s*=\s*)([0-9.e-]+)"
            replacement = rf"\g<1>{new_value:.2e}"
            mutated = re.sub(pattern, replacement, current_code)
        elif adjustment_type == "lora_dropout":
            pattern = r"(lora_dropout\s*:\s*float\s*=\s*)([0-9.]+)"
            replacement = rf"\g<1>{new_value:.2f}"
            mutated = re.sub(pattern, replacement, current_code)
        else:
            mutated = current_code

        self.mutation_log.append({
            "type": adjustment_type,
            "new_value": new_value,
            "status": "proposed"
        })
        return mutated

    def validate_syntax(self, code_string: str) -> bool:
        """Ensures that self-generated code parses without syntax errors before writing to disk."""
        try:
            compile(code_string, "<string>", "exec")
            return True
        except SyntaxError as e:
            print(f"[CodeMutator] Syntax validation failed: {e}")
            return False
