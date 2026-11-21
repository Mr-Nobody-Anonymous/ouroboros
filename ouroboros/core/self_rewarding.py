"""Preference-data curation with explicit, independently supplied judges; no style-based rewards."""

import math


class SelfRewardingEngine:
    def __init__(self, rubric=None, *, judge=None, generator=None, min_margin=0.01):
        if not math.isfinite(min_margin) or min_margin <= 0:
            raise ValueError("min_margin must be positive")
        self.rubric = rubric or {}
        self.judge, self.generator, self.min_margin = judge, generator, min_margin

    def judge_generation(self, prompt, response):
        if self.judge is None:
            raise RuntimeError(
                "An explicit judge is required; reasoning keywords do not measure correctness"
            )
        score = float(self.judge(prompt, response))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Judge score must be finite and in [0, 1]")
        return score

    def create_dpo_pair(self, prompt, candidate_a, candidate_b):
        if not prompt.strip() or not candidate_a.strip() or not candidate_b.strip():
            raise ValueError("Prompt and candidates must be nonempty")
        if candidate_a.strip() == candidate_b.strip():
            return None
        a, b = (
            self.judge_generation(prompt, candidate_a),
            self.judge_generation(prompt, candidate_b),
        )
        if abs(a - b) < self.min_margin:
            return None  # abstain on ties instead of manufacturing a preference
        chosen, rejected = (candidate_a, candidate_b) if a > b else (candidate_b, candidate_a)
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_score": max(a, b),
            "rejected_score": min(a, b),
            "margin": abs(a - b),
        }

    def synthesize_dpo_dataset(self, task_prompts):
        if self.generator is None:
            raise RuntimeError("A candidate generator is required; no canned training responses")
        result = []
        for prompt in task_prompts:
            candidates = self.generator(prompt)
            if len(candidates) != 2:
                raise ValueError("Generate exactly two candidates per prompt")
            pair = self.create_dpo_pair(prompt, *candidates)
            if pair:
                result.append(pair)
        return result
