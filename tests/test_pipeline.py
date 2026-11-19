"""
Unit tests for Ouroboros core modules.
"""
import unittest
import numpy as np
from ouroboros.core.weight_editor import WeightEditor
from ouroboros.core.adapter_engine import AdapterEngine
from ouroboros.core.evaluator import SelfEvaluator
from ouroboros.core.meta_feedback import MetaFeedbackGenerator

class TestOuroborosCore(unittest.TestCase):
    def setUp(self):
        self.editor = WeightEditor()
        self.engine = AdapterEngine()
        self.evaluator = SelfEvaluator()
        self.generator = MetaFeedbackGenerator()

    def test_lora_delta_computation(self):
        r = 4
        d_in = 8
        d_out = 16
        alpha = 16.0
        
        a = np.ones((r, d_in), dtype=np.float32)
        b = np.ones((d_out, r), dtype=np.float32)
        
        delta = self.editor.compute_lora_delta(a, b, alpha, r)
        self.assertEqual(delta.shape, (d_out, d_in))
        # (16 / 4) * (4 * 1.0) = 4 * 4 = 16.0
        self.assertTrue(np.allclose(delta, 16.0))

    def test_weight_clamping(self):
        base = np.ones((10, 10), dtype=np.float32)
        delta = np.ones((10, 10), dtype=np.float32) * 5.0
        merged = self.editor.merge_and_clamp(base, delta, max_deviation_ratio=0.2)
        self.assertEqual(merged.shape, (10, 10))

    def test_verification_improvement(self):
        scores_good = {
            "math_logic": 0.75,
            "coding": 0.70,
            "general_knowledge": 0.82,
            "instruction_following": 0.78
        }
        passed, msg = self.evaluator.verify_improvement(scores_good)
        self.assertTrue(passed)

    def test_catastrophic_forgetting_detection(self):
        scores_bad = {
            "math_logic": 0.90,
            "coding": 0.40,  # Dropped significantly from 0.64 baseline
            "general_knowledge": 0.81,
            "instruction_following": 0.77
        }
        passed, msg = self.evaluator.verify_improvement(scores_bad, forgetting_threshold=0.05)
        self.assertFalse(passed)
        self.assertIn("Catastrophic forgetting", msg)

if __name__ == "__main__":
    unittest.main()
