"""Weak-domain selection and an explicit curriculum generation callback boundary."""

import itertools
import math


class MetaFeedbackGenerator:
    def __init__(self, curriculum_seed=42, *, generator=None):
        self.seed, self.generator = curriculum_seed, generator

    def diagnose_weakness(self, evaluation_scores):
        if not evaluation_scores or any(
            not math.isfinite(v) or not 0 <= v <= 1 for v in evaluation_scores.values()
        ):
            raise ValueError("Provide nonempty measured domain accuracies")
        return min(evaluation_scores, key=lambda d: (evaluation_scores[d], d))

    def synthesize_curriculum_batch(self, target_domain, batch_size=5):
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 128:
            raise ValueError("batch_size must be in [1, 128]")
        if self.generator is None:
            raise RuntimeError(
                "Provide a real curriculum generator; canned examples are not self-instruction"
            )
        # The caller must independently verify and deduplicate these proposals before training.
        return list(itertools.islice(self.generator(target_domain, self.seed), batch_size))
