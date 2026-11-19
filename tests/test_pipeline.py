"""
Unit tests for Ouroboros core modules, task arithmetic, and architecture mutations.
"""
import unittest
import os
import numpy as np
from ouroboros.core.weight_editor import WeightEditor
from ouroboros.core.adapter_engine import AdapterEngine
from ouroboros.core.evaluator import SelfEvaluator
from ouroboros.core.meta_feedback import MetaFeedbackGenerator
from ouroboros.core.task_arithmetic import TaskArithmeticEngine
from ouroboros.core.architecture_mutator import ArchitectureMutator
from ouroboros.core.self_rewarding import SelfRewardingEngine
from ouroboros.integrations.mergekit_bridge import MergeKitBridge

class TestOuroborosCore(unittest.TestCase):
    def setUp(self):
        self.editor = WeightEditor()
        self.engine = AdapterEngine()
        self.evaluator = SelfEvaluator()
        self.generator = MetaFeedbackGenerator()
        self.arithmetic = TaskArithmeticEngine()
        self.mutator = ArchitectureMutator()
        self.rewarding = SelfRewardingEngine()
        self.mergekit = MergeKitBridge()

    def test_lora_delta_computation(self):
        r = 4
        d_in = 8
        d_out = 16
        alpha = 16.0
        
        a = np.ones((r, d_in), dtype=np.float32)
        b = np.ones((d_out, r), dtype=np.float32)
        
        delta = self.editor.compute_lora_delta(a, b, alpha, r)
        self.assertEqual(delta.shape, (d_out, d_in))
        self.assertTrue(np.allclose(delta, 16.0))

    def test_weight_clamping(self):
        base = np.ones((10, 10), dtype=np.float32)
        delta = np.ones((10, 10), dtype=np.float32) * 5.0
        merged = self.editor.merge_and_clamp(base, delta, max_deviation_ratio=0.2)
        self.assertEqual(merged.shape, (10, 10))

    def test_task_vector_arithmetic(self):
        base = np.zeros((4, 4), dtype=np.float32)
        ft = np.ones((4, 4), dtype=np.float32) * 2.0
        
        # tau = ft - base
        tau = self.arithmetic.compute_task_vector(ft, base)
        self.assertTrue(np.allclose(tau, 2.0))
        
        # New model = base + 0.5 * tau
        new_w = self.arithmetic.apply_task_vector(base, tau, scaling=0.5)
        self.assertTrue(np.allclose(new_w, 1.0))
        
        # Unlearn negative behavior
        unlearned = self.arithmetic.unlearn_behavior(new_w, tau, beta=0.2)
        self.assertTrue(np.allclose(unlearned, 0.6))

    def test_slerp_interpolation(self):
        v0 = np.array([1.0, 0.0], dtype=np.float32)
        v1 = np.array([0.0, 1.0], dtype=np.float32)
        mid = self.arithmetic.slerp_interpolate(v0, v1, t=0.5)
        self.assertAlmostEqual(float(mid[0]), float(mid[1]), places=4)

    def test_dare_merging(self):
        base = np.zeros((10, 10), dtype=np.float32)
        tv = [np.ones((10, 10), dtype=np.float32) * 2.0]
        merged = self.arithmetic.dare_merge(base, tv, drop_rate=0.5)
        self.assertEqual(merged.shape, (10, 10))

    def test_architecture_mutations(self):
        # 32 layers expanded to 34 layers via stacking
        blueprint = self.mutator.layer_stack_blueprint(32, [16, 24])
        self.assertEqual(len(blueprint), 34)
        
        # Attention head scaling
        qkv = np.ones((64, 64), dtype=np.float32)
        scaled_qkv = self.mutator.scale_attention_heads(qkv, num_heads=4, head_scales={0: 1.5})
        self.assertAlmostEqual(float(scaled_qkv[0, 0]), 1.5)
        self.assertAlmostEqual(float(scaled_qkv[20, 20]), 1.0)
        
        # Steering vector calculation
        pos = np.ones((10, 8), dtype=np.float32) * 2.0
        neg = np.ones((10, 8), dtype=np.float32) * 1.0
        steer = self.mutator.compute_steering_vector(pos, neg)
        self.assertEqual(steer.shape, (8,))
        self.assertAlmostEqual(float(np.linalg.norm(steer)), 1.0, places=4)

    def test_self_rewarding_and_dpo(self):
        prompt = "Explain backpropagation step by step."
        candidate_good = "Step 1: Compute loss. Step 2: Calculate gradients using the chain rule. Therefore weights update."
        candidate_bad = "I think it updates weights somehow."
        
        pair = self.rewarding.create_dpo_pair(prompt, candidate_good, candidate_bad)
        self.assertEqual(pair["chosen"], candidate_good)
        self.assertGreater(pair["chosen_score"], pair["rejected_score"])
        
        dataset = self.rewarding.synthesize_dpo_dataset(["Solve 2+2", "Explain LoRA"])
        self.assertEqual(len(dataset), 2)

    def test_mergekit_bridge_recipe(self):
        tmp_yaml = "./outputs/test_merge_recipe.yml"
        path = self.mergekit.generate_slerp_recipe("model_a", "model_b", tmp_yaml)
        self.assertTrue(os.path.exists(path))
        if os.path.exists(path):
            os.remove(path)

    def test_catastrophic_forgetting_detection(self):
        scores_bad = {
            "math_logic": 0.90,
            "coding": 0.40,  # Dropped from 0.64 baseline
            "general_knowledge": 0.81,
            "instruction_following": 0.77
        }
        passed, msg = self.evaluator.verify_improvement(scores_bad, forgetting_threshold=0.05)
        self.assertFalse(passed)
        self.assertIn("Catastrophic forgetting", msg)

if __name__ == "__main__":
    unittest.main()
