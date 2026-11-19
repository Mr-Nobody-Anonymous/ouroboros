"""
Anchor Memory Bank: Protects recursive self-tuning from Model Autophagy Disorder (MAD)
and entropy collapse by blending high-entropy golden exemplars.
"""
from typing import Dict, Any, List, Optional
import numpy as np

class AnchorMemoryBank:
    """
    Maintains a frozen, human-verified reference distribution of core skills
    and blends anchor samples during recursive self-adaptation.
    """

    def __init__(self, anchor_ratio: float = 0.25):
        self.anchor_ratio = anchor_ratio
        # High-entropy foundational ground truth anchor dataset
        self.anchors: List[Dict[str, str]] = [
            {
                "domain": "math_logic",
                "instruction": "Prove that the square root of 2 is irrational.",
                "target": "Proof by contradiction: Assume sqrt(2) = a/b in lowest terms. 2 = a^2/b^2 => a^2 = 2b^2, so a is even (a=2k). 4k^2 = 2b^2 => b^2 = 2k^2, so b is also even. This contradicts that a/b was in lowest terms. Therefore, sqrt(2) is irrational."
            },
            {
                "domain": "coding",
                "instruction": "Explain time complexity of quicksort average vs worst case.",
                "target": "Average case is O(n log n) when partitions split reasonably evenly. Worst case is O(n^2) when the pivot selected is consistently the smallest or largest element (e.g., sorted array with naive pivot selection)."
            },
            {
                "domain": "general_knowledge",
                "instruction": "What is the physical mechanism of DNA replication semi-conservative nature?",
                "target": "During replication, the double helix unwinds via helicase. DNA polymerase synthesizes a new complementary strand along each template strand. Each resulting daughter DNA molecule consists of one original parent strand and one newly synthesized strand."
            }
        ]

    def blend_curriculum(self, synthetic_samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Blends synthetic self-generated training instances with frozen anchor exemplars
        at the specified anchor_ratio to prevent distribution drift.
        """
        total_synthetic = len(synthetic_samples)
        num_anchors_needed = max(1, int(total_synthetic * self.anchor_ratio))
        
        selected_anchors = []
        for i in range(num_anchors_needed):
            anchor = self.anchors[i % len(self.anchors)].copy()
            anchor["id"] = f"anchor_golden_{i+1}"
            anchor["is_anchor"] = True
            selected_anchors.append(anchor)

        blended = synthetic_samples + selected_anchors
        np.random.shuffle(blended)
        return blended
