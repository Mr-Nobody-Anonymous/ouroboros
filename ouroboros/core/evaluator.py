"""Measured exact-match evaluation and fail-closed promotion decisions."""

import math
from typing import Dict, List, Optional, Tuple

from ..data import normalize


class SelfEvaluator:
    """No built-in benchmark numbers. Baselines must come from predictions or a caller."""

    def __init__(self, baseline_scores: Optional[Dict[str, float]] = None):
        self.baseline_scores = dict(baseline_scores or {})
        self._validate_scores(self.baseline_scores)

    @staticmethod
    def _validate_scores(scores):
        if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores.values()):
            raise ValueError("Accuracy scores must be finite and in [0, 1]")

    def evaluate_benchmark(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Whitespace/case-normalized exact match, not substring matching (1 is not 10)."""
        if len(predictions) != len(ground_truths):
            raise ValueError("Prediction and target counts differ")
        if any(not target.strip() for target in ground_truths):
            raise ValueError("Evaluation targets cannot be empty")
        if not predictions:
            return 0.0
        return sum(normalize(p) == normalize(g) for p, g in zip(predictions, ground_truths)) / len(
            predictions
        )

    def evaluate_generation(self, test_suite: Dict[str, List[Dict[str, str]]]) -> Dict[str, float]:
        if not test_suite or any(not samples for samples in test_suite.values()):
            raise ValueError("Every evaluation domain needs samples")
        return {
            domain: self.evaluate_benchmark(
                [s["prediction"] for s in samples], [s["target"] for s in samples]
            )
            for domain, samples in test_suite.items()
        }

    def verify_improvement(
        self,
        current_scores: Dict[str, float],
        forgetting_threshold=0.05,
        *,
        previous_scores=None,
        improvement_threshold=0.0,
        target_domain=None,
    ) -> Tuple[bool, str]:
        """Compare against both generation 0 and the last accepted generation.

        Thresholds are absolute accuracy fractions, not relative percentages.
        Even with a zero threshold, equality is not improvement.
        """
        self._validate_scores(current_scores)
        previous = self.baseline_scores if previous_scores is None else previous_scores
        self._validate_scores(previous)
        if not self.baseline_scores or not previous:
            return False, "A measured baseline and previous generation are required"
        if set(current_scores) != set(self.baseline_scores) or set(previous) != set(current_scores):
            return False, "Missing or unexpected evaluation domains"
        if not 0 <= forgetting_threshold <= 1 or not 0 <= improvement_threshold <= 1:
            raise ValueError("Thresholds must be in [0, 1]")
        for domain, current in current_scores.items():
            floor = max(self.baseline_scores[domain], previous[domain]) - forgetting_threshold
            if current < floor - 1e-12:
                return False, f"Forgetting detected on '{domain}': {current:.4f} < {floor:.4f}"
        gain = sum(current_scores[d] - previous[d] for d in previous) / len(previous)
        if gain <= 1e-12 or gain + 1e-12 < improvement_threshold:
            return False, f"Mean gain {gain:.6f} does not meet the promotion threshold"
        if target_domain is not None and (
            target_domain not in previous
            or current_scores[target_domain] <= previous[target_domain]
        ):
            return False, "Target domain did not improve"
        return True, f"Measured macro accuracy improved by {gain * 100:.2f} percentage points"
