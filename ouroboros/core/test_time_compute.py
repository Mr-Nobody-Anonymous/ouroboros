"""Best-of-N using explicit generation/verifier callbacks, with exact-answer curation."""

import math
from collections import Counter

from ..data import normalize


class TestTimeComputeEngine:
    __test__ = False

    def __init__(self, default_budget_n=4, temperature=0.7, *, generator=None, verifier=None):
        if not isinstance(default_budget_n, int) or not 1 <= default_budget_n <= 128:
            raise ValueError("Candidate budget must be an integer in [1, 128]")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Temperature must be finite and nonnegative")
        self.default_budget_n, self.temperature = default_budget_n, temperature
        self.generator, self.verifier = generator, verifier

    def sample_rollouts(self, prompt, n_candidates=None):
        n = self.default_budget_n if n_candidates is None else n_candidates
        if not isinstance(n, int) or not 1 <= n <= 128:
            raise ValueError("Candidate budget must be an integer in [1, 128]")
        if self.generator is None or self.verifier is None:
            raise RuntimeError(
                "Provide real generator and independent verifier callbacks; no random scores"
            )
        candidates = []
        for i in range(n):
            generated = self.generator(prompt, i, self.temperature)
            if not isinstance(generated, dict) or not isinstance(
                generated.get("final_answer"), str
            ):
                raise ValueError("Generator must return a record with a final_answer string")
            score = float(self.verifier(prompt, generated["final_answer"]))
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Verifier score must be finite and in [0, 1]")
            candidates.append({**generated, "candidate_id": i + 1, "verifier_score": score})
        return candidates

    def verifier_rerank(self, candidates):
        if not candidates:
            raise ValueError("No candidates to rerank")
        for candidate in candidates:
            score = candidate["verifier_score"]
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid verifier score")
        ranked = sorted(candidates, key=lambda c: c["verifier_score"], reverse=True)
        margin = (
            ranked[0]["verifier_score"] - ranked[1]["verifier_score"] if len(ranked) > 1 else 0.0
        )
        return ranked[0], float(margin)

    @staticmethod
    def majority_vote(candidates):
        if not candidates:
            raise ValueError("No candidates to vote on")
        counts = Counter(normalize(c["final_answer"]) for c in candidates)
        answer = counts.most_common(1)[0][0]
        return next(c for c in candidates if normalize(c["final_answer"]) == answer)

    def extract_star_training_pair(self, prompt, ground_truth, candidates):
        """A high self-assigned score cannot turn a wrong answer into a training label.

        This is correctness-filtered rationale export, not a complete STaR reproduction.
        """
        if not ground_truth.strip():
            raise ValueError("Ground truth must be nonempty")
        for candidate in candidates:
            if normalize(candidate["final_answer"]) == normalize(ground_truth):
                return {
                    "prompt": prompt,
                    "curated_rationale": candidate.get("reasoning_trace", ""),
                    "target": ground_truth,
                    "score": candidate["verifier_score"],
                }
        return None
