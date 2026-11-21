"""Compatibility smoke tests for the public tensor/curation APIs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ouroboros.core.adapter_engine import AdapterEngine
from ouroboros.core.adversarial_critic import AdversarialCritic
from ouroboros.core.anchor_memory import AnchorMemoryBank
from ouroboros.core.architecture_mutator import ArchitectureMutator
from ouroboros.core.code_mutator import CodeMutator
from ouroboros.core.self_rewarding import SelfRewardingEngine
from ouroboros.core.task_arithmetic import TaskArithmeticEngine
from ouroboros.core.test_time_compute import TestTimeComputeEngine
from ouroboros.core.weight_editor import WeightEditor
from ouroboros.core.weight_space_programmer import WeightSpaceProgrammer
from ouroboros.integrations.mergekit_bridge import MergeKitBridge


class TestOuroborosCore(unittest.TestCase):
    def test_lora_delta_computation(self):
        delta = WeightEditor().compute_lora_delta(np.ones((4, 8)), np.ones((16, 4)), 16, 4)
        self.assertEqual(delta.shape, (16, 8))
        np.testing.assert_allclose(delta, 16)

    def test_weight_clamping(self):
        base, delta = np.ones((10, 10)), np.full((10, 10), 5.0)
        merged = WeightEditor().merge_and_clamp(base, delta, 0.2)
        self.assertLessEqual(np.linalg.norm(merged - base), np.linalg.norm(base) * 0.2 + 1e-10)

    def test_task_vector_arithmetic(self):
        base = np.zeros((4, 4))
        tau = TaskArithmeticEngine.compute_task_vector(np.full((4, 4), 2.0), base)
        np.testing.assert_allclose(TaskArithmeticEngine.apply_task_vector(base, tau, 0.5), 1)

    def test_slerp_interpolation(self):
        mid = TaskArithmeticEngine.slerp_interpolate(
            np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.5
        )
        self.assertAlmostEqual(mid[0], mid[1])

    def test_dare_merging(self):
        merged = TaskArithmeticEngine.dare_merge(
            np.zeros((10, 10)), [np.ones((10, 10))], drop_rate=0.5, rng=np.random.default_rng(7)
        )
        self.assertEqual(merged.shape, (10, 10))
        self.assertTrue(np.isfinite(merged).all())

    def test_architecture_mutations(self):
        blueprint = ArchitectureMutator.layer_stack_blueprint(32, [16, 24])
        self.assertEqual(len(blueprint), 34)
        self.assertEqual(blueprint.count(24), 2)

    def test_self_rewarding_and_dpo(self):
        engine = SelfRewardingEngine(judge=lambda prompt, response: float(response == "4"))
        pair = engine.create_dpo_pair("2 + 2", "4", "5")
        self.assertEqual(pair["chosen"], "4")

    def test_anchor_memory_blending(self):
        synthetic = [{"instruction": f"task {i}", "target": str(i)} for i in range(8)]
        blended = AnchorMemoryBank(0.25).blend_curriculum(synthetic)
        self.assertGreaterEqual(sum(bool(s.get("is_anchor")) for s in blended) / len(blended), 0.25)

    def test_adversarial_critic(self):
        critic = AdversarialCritic()
        self.assertEqual(len(critic.generate_adversarial_challenges(2)), 2)
        self.assertFalse(critic.audit_response("8", {"target": "9"})[0])
        self.assertFalse(critic.audit_response("no idea", {"target": "9"})[0])

    def test_code_mutator(self):
        code = "loss = dpo_loss(beta=0.100, lr=1e-4)"
        mutated = CodeMutator().propose_loss_mutation(code, "dpo_beta", 0.05)
        self.assertIn("beta=0.050", mutated)
        self.assertTrue(CodeMutator.validate_syntax(mutated))

    def test_weight_space_programming(self):
        programmer = WeightSpaceProgrammer()
        weights = np.random.default_rng(4).normal(0, 0.02, (64, 128))
        direction = programmer.dominant_weight_direction(weights)
        ablated = programmer.surgical_directional_ablation(weights, direction)
        result, stats = programmer.sparse_topk_edit(weights, ablated, 0.016)
        self.assertEqual(result.shape, weights.shape)
        self.assertLessEqual(stats["edited_weights"], int(weights.size * 0.016))

    def test_test_time_compute(self):
        engine = TestTimeComputeEngine(
            generator=lambda prompt, i, temperature: {"final_answer": "4" if i == 0 else "5"},
            verifier=lambda prompt, answer: float(answer == "4"),
        )
        rollouts = engine.sample_rollouts("2+2")
        best, margin = engine.verifier_rerank(rollouts)
        self.assertEqual(best["final_answer"], "4")
        self.assertEqual(margin, 1.0)
        self.assertIsNotNone(engine.extract_star_training_pair("2+2", "4", rollouts))

    def test_adapter_checkpoint_has_weights(self):
        engine = AdapterEngine()
        original = engine.initialize_lora_layers(["q_proj"], 8, 8)["q_proj"]["lora_A"].copy()
        with tempfile.TemporaryDirectory() as tmp:
            engine.save_adapter_checkpoint(tmp)
            self.assertTrue((Path(tmp) / "adapter_weights.npz").is_file())
            other = AdapterEngine()
            other.load_adapter_checkpoint(tmp)
            np.testing.assert_array_equal(original, other.active_adapters["q_proj"]["lora_A"])

    def test_mergekit_bridge_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = MergeKitBridge().generate_slerp_recipe("model_a", "model_b", f"{tmp}/merge.yml")
            self.assertTrue(Path(path).is_file())


if __name__ == "__main__":
    unittest.main()
