"""Verified adversarial challenge rounds; fixed fixtures are not a learned minimax adversary."""

import random

from ..data import Sample, normalize


class AdversarialCritic:
    def __init__(self, skepticism_level=0.8, *, seed=42, generator=None, verifier=None):
        if not 0 <= skepticism_level <= 1:
            raise ValueError("skepticism_level must be in [0, 1]")
        self.skepticism_level = skepticism_level
        self.rng = random.Random(seed)
        self.generator, self.verifier = generator, verifier
        self.challenge_templates = [
            {
                "type": "premise_contradiction",
                "instruction": "A farmer has 17 sheep and all but 9 die. How many are left?",
                "adversarial_trap": "8",
                "target": "9",
            },
            {
                "type": "scale_fallacy",
                "instruction": "5 machines take 5 minutes to make 5 widgets. How many minutes do 100 machines take to make 100 widgets?",
                "adversarial_trap": "100",
                "target": "5",
            },
            {
                "type": "sign_inversion",
                "instruction": "In theta - beta*tau, beta = -1. Does this subtract or add tau? Answer add or subtract.",
                "adversarial_trap": "subtract",
                "target": "add",
            },
        ]

    def generate_adversarial_challenges(self, count=2):
        if not isinstance(count, int) or not 0 <= count <= 128:
            raise ValueError("Challenge count must be an integer in [0, 128]")
        if self.generator:
            if not self.verifier:
                raise ValueError("Generated challenges need an independent verifier")
            challenges = list(self.generator(count))
            if len(challenges) > count:
                raise ValueError("Generator exceeded the challenge budget")
            return [dict(c) for c in challenges if self.verifier(c)]
        return [
            dict(c)
            for c in self.rng.sample(
                self.challenge_templates, min(count, len(self.challenge_templates))
            )
        ]

    def audit_response(self, response, challenge):
        # Canonical short answers only. Explanatory text needs a task-specific verifier/extractor.
        passed = bool(challenge.get("target", "").strip()) and normalize(response) == normalize(
            challenge["target"]
        )
        return passed, "Verified exact answer" if passed else "Unverified or incorrect answer"

    def challenge_round(self, learner, count=2):
        challenges = self.generate_adversarial_challenges(count)
        predictions = learner.predict([c["instruction"] for c in challenges])
        if len(predictions) != len(challenges):
            raise ValueError("Learner returned the wrong number of answers")
        failures, correct = [], 0
        for challenge, prediction in zip(challenges, predictions):
            passed, _ = self.audit_response(prediction, challenge)
            correct += passed
            if not passed:
                failures.append(
                    Sample(
                        challenge["instruction"],
                        challenge["target"],
                        "adversarial",
                        "verified-adversary" if self.generator else "fixture-adversary",
                        True,
                    )
                )
        return {
            "count": len(challenges),
            "correct": correct,
            "accuracy": correct / len(challenges) if challenges else 0.0,
            "failures": failures,
        }
