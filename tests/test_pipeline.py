"""
Unit tests for Ouroboros core modules, task arithmetic, architecture mutations,
anchor memory, adversarial critics, and code mutators.
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
from ouroboros.core.anchor_memory import AnchorMemoryBank
from ouroboros.core.adversarial_critic import AdversarialCritic
from ouroboros.core.code_mutator import CodeMutator
from ouroboros.core.pareto_pruner import ParetoCheckpointPruner
from ouroboros.core.weight_space_programmer import WeightSpaceProgrammer
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
        self.anchor = AnchorMemoryBank(anchor_ratio=0.25)
        self.critic = AdversarialCritic()
        self.coder = CodeMutator()
        self.pruner = ParetoCheckpointPruner(max_checkpoints_to_keep=2)
        self.wsp = WeightSpaceProgrammer(target_sparsity=0.05)
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
        tau = self.arithmetic.compute_task_vector(ft, base)
        self.assertTrue(np.allclose(tau, 2.0))
        new_w = self.arithmetic.apply_task_vector(base, tau, scaling=0.5)
        self.assertTrue(np.allclose(new_w, 1.0))

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
        blueprint = self.mutator.layer_stack_blueprint(32, [16, 24])
        self.assertEqual(len(blueprint), 34)

    def test_self_rewarding_and_dpo(self):
        prompt = "Explain backpropagation step by step."
        candidate_good = "Step 1: Compute loss. Step 2: Calculate gradients using the chain rule. Therefore weights update."
        candidate_bad = "I think it updates weights somehow."
        pair = self.rewarding.create_dpo_pair(prompt, candidate_good, candidate_bad)
        self.assertEqual(pair["chosen"], candidate_good)

    def test_anchor_memory_blending(self):
        synthetic = [{"instruction": f"Solve task {i}", "target": f"Answer {i}"} for i in range(8)]
        blended = self.anchor.blend_curriculum(synthetic)
        # Should contain synthetic + blended anchors
        self.assertGreater(len(blended), len(synthetic))
        self.assertTrue(any(sample.get("is_anchor") for sample in blended))

    def test_adversarial_critic(self):
        challenges = self.critic.generate_adversarial_challenges(count=2)
        self.assertEqual(len(challenges), 2)
        # Audit trap detection
        trap_sample = {"type": "premise_contradiction", "adversarial_trap": "8", "target": "9"}
        passed, msg = self.critic.audit_response("I think the answer is 8 sheep.", trap_sample)
        self.assertFalse(passed)

    def test_code_mutator(self):
        code_snippet = "loss = dpo_loss(beta=0.100, lr=1e-4)"
        mutated = self.coder.propose_loss_mutation(code_snippet, "dpo_beta", 0.05)
        self.assertIn("beta=0.050", mutated)
        self.assertTrue(self.coder.validate_syntax(mutated))

    def test_weight_space_programming(self):
        # Create mock down-projection matrix (64 x 128)
        w_down = np.random.normal(0, 0.02, (64, 128)).astype(np.float32)
        # Identify attractor vector
        attractor = self.wsp.diagnose_repetitive_attractor(w_down)
        self.assertEqual(attractor.shape, (64, 1))
        # Directional ablation
        ablated = self.wsp.surgical_directional_ablation(w_down, attractor, ablation_strength=0.5)
        # Sparse top-K edit
        final_w, stats = self.wsp.sparse_topk_edit(w_down, ablated, sparsity_ratio=0.016)
        self.assertEqual(final_w.shape, w_down.shape)
        self.assertGreater(stats["edited_weights"], 0)
        self.assertLessEqual(stats["actual_sparsity_percentage"], 2.0)

    def test_mergekit_bridge_recipe(self):
        tmp_yaml = "./outputs/test_merge_recipe.yml"
        path = self.mergekit.generate_slerp_recipe("model_a", "model_b", tmp_yaml)
        self.assertTrue(os.path.exists(path))
        if os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    unittest.main()
