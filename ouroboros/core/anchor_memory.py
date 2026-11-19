"""Deterministic replay of caller-verified examples; a mitigation, not a collapse guarantee."""

import copy
import math

import numpy as np


class AnchorMemoryBank:
    def __init__(self, anchor_ratio=0.25, anchors=None, seed=42):
        if not math.isfinite(anchor_ratio) or not 0 <= anchor_ratio < 1:
            raise ValueError("anchor_ratio must be in [0, 1)")
        self.anchor_ratio = anchor_ratio
        # Small illustrative defaults for the legacy utility API, not a foundational corpus.
        if anchors is None:
            anchors = [
                {
                    "domain": "math_logic",
                    "instruction": "What is 2 plus 2?",
                    "target": "4",
                    "source": "built-in-example",
                    "verified": True,
                }
            ]
        if any(not a.get("verified", False) for a in anchors):
            raise ValueError("Anchor examples must be explicitly verified")
        self._anchors = tuple(copy.deepcopy(anchors))
        self.rng = np.random.default_rng(seed)

    @property
    def anchors(self):
        return copy.deepcopy(list(self._anchors))

    def blend_curriculum(self, synthetic_samples):
        """Return fresh records with at least ratio anchors in the final batch (rounded up).

        For n new samples and ratio r, a = ceil(n*r/(1-r)), not n*r.
        Empty curricula stay empty. Duplicated replay examples do not increase corpus diversity.
        """
        blended = copy.deepcopy(list(synthetic_samples))
        if not blended or self.anchor_ratio == 0:
            return blended
        if not self._anchors:
            raise ValueError("Positive anchor_ratio requires a nonempty anchor bank")
        count = math.ceil(len(blended) * self.anchor_ratio / (1 - self.anchor_ratio))
        for i in range(count):
            anchor = copy.deepcopy(self._anchors[i % len(self._anchors)])
            anchor["is_anchor"] = True
            blended.append(anchor)
        self.rng.shuffle(blended)
        return blended
