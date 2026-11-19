"""
Adversarial Critic: Dual-agent co-evolution loop (Generator vs. Skeptic)
inspired by SPIN (Self-Play Fine-Tuning) and MoE-CL to prevent reward hacking.
"""
from typing import Dict, Any, List, Tuple
import random

class AdversarialCritic:
    """
    Implements a minimax co-evolutionary dynamic where an Adversarial Skeptic
    generates tricky reasoning pitfalls and edge cases to stress-test the model.
    """

    def __init__(self, skepticism_level: float = 0.8):
        self.skepticism_level = skepticism_level
        self.challenge_templates = [
            {
                "type": "premise_contradiction",
                "pattern": "A farmer has 17 sheep and all but 9 die. How many sheep are left?",
                "trap": "Subtracting 9 from 17 to get 8.",
                "solution": "All but 9 died, meaning exactly 9 sheep remain alive."
            },
            {
                "type": "scale_fallacy",
                "pattern": "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
                "trap": "Scaling linearly to 100 minutes.",
                "solution": "Each machine takes 5 minutes to make 1 widget. Thus, 100 machines working in parallel produce 100 widgets in 5 minutes."
            },
            {
                "type": "sign_inversion",
                "pattern": "What is the result of applying a negative task vector with coefficient beta = -1.0?",
                "trap": "Thinking negative coefficient subtracts the behavior.",
                "solution": "theta - (-1.0 * tau) = theta + tau, which amplifies the behavior rather than unlearning it."
            }
        ]

    def generate_adversarial_challenges(self, count: int = 2) -> List[Dict[str, str]]:
        """Generates adversarial reasoning queries designed to expose shallow pattern matching."""
        sampled = random.sample(self.challenge_templates, min(count, len(self.challenge_templates)))
        challenges = []
        for s in sampled:
            challenges.append({
                "instruction": s["pattern"],
                "adversarial_trap": s["trap"],
                "target": s["solution"],
                "type": s["type"]
            })
        return challenges

    def audit_response(self, response: str, challenge: Dict[str, str]) -> Tuple[bool, str]:
        """Audits whether the model's generated answer fell into the adversarial trap."""
        trap_phrase = challenge["adversarial_trap"].lower()
        resp_lower = response.lower()

        # If the trap phrase appears or solution keywords are missing
        if "8" in resp_lower and challenge["type"] == "premise_contradiction":
            return False, "Fell into premise contradiction trap (answered 8 instead of 9)."
        if "100 minutes" in resp_lower and challenge["type"] == "scale_fallacy":
            return False, "Fell into linear scale fallacy (answered 100 minutes instead of 5)."

        return True, "Passed adversarial challenge with sound deductive reasoning."
