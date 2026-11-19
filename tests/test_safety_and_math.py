import json
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
import yaml

from ouroboros.core.adversarial_critic import AdversarialCritic
from ouroboros.core.architecture_mutator import ArchitectureMutator
from ouroboros.core.checkpoints import CheckpointStore
from ouroboros.core.code_mutator import CodeMutator, NumericObjective
from ouroboros.core.pareto_pruner import ParetoCheckpointPruner
from ouroboros.core.self_rewarding import SelfRewardingEngine
from ouroboros.core.task_arithmetic import TaskArithmeticEngine
from ouroboros.core.test_time_compute import TestTimeComputeEngine
from ouroboros.core.weight_editor import WeightEditor
from ouroboros.core.weight_space_programmer import WeightSpaceProgrammer
from ouroboros.integrations.easyedit_bridge import EasyEditBridge
from ouroboros.integrations.mergekit_bridge import MergeKitBridge


@pytest.mark.parametrize("ratio,expected", [(0, 0), (0.01, 0), (0.2, 2), (1, 10)])
def test_topk_budget_including_ties_and_zero(ratio, expected):
    result, stats = WeightSpaceProgrammer().sparse_topk_edit(
        np.zeros((2, 5)), np.ones((2, 5)), ratio
    )
    assert np.count_nonzero(result) == expected
    assert stats["edited_weights"] == expected


@pytest.mark.parametrize("ratio", [-1, 1.01, float("nan")])
def test_bad_edit_fraction(ratio):
    with pytest.raises(ValueError):
        WeightSpaceProgrammer(ratio)


def test_surgery_checks_shapes_and_zero_direction():
    editor = WeightSpaceProgrammer()
    with pytest.raises(ValueError):
        editor.surgical_directional_ablation(np.eye(2), np.zeros(2))
    with pytest.raises(ValueError):
        editor.sparse_topk_edit(np.zeros((2, 3)), np.ones((1, 3)))
    with pytest.raises(ValueError):
        editor.contrastive_direction(np.ones((3, 2)), np.ones((3, 2)))
    direction = editor.contrastive_direction(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))
    np.testing.assert_allclose(np.linalg.norm(direction), 1)


def test_ablation_removes_only_chosen_direction():
    result = WeightSpaceProgrammer().surgical_directional_ablation(
        np.eye(2), np.array([1.0, 0.0]), 1
    )
    np.testing.assert_allclose(result, [[0, 0], [0, 1]])


def test_ties_disjoint_signs():
    merged = TaskArithmeticEngine.ties_merge(
        np.zeros(3), [np.array([2.0, -1.0, 1.0]), np.array([-1.0, -3.0, -1.0])], density=1
    )
    np.testing.assert_allclose(merged, [2, -2, 0])


def test_dare_reproducible_and_validated():
    base, tv = np.zeros(100), [np.ones(100)]
    one = TaskArithmeticEngine.dare_merge(base, tv, rng=np.random.default_rng(7))
    two = TaskArithmeticEngine.dare_merge(base, tv, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(one, two)
    for rate in (1, -1, float("nan")):
        with pytest.raises(ValueError):
            TaskArithmeticEngine.dare_merge(base, tv, rate)
    with pytest.raises(ValueError):
        TaskArithmeticEngine.dare_merge(base, [])
    with pytest.raises(ValueError):
        TaskArithmeticEngine.compute_task_vector(np.ones((2, 2)), np.ones((1, 2)))


def test_rank_one_matches_target_and_zero_key_checked():
    base = np.eye(2)
    result = WeightEditor().surgical_rank_one_edit(
        base, np.array([1.0, 0.0]), np.array([3.0, 2.0]), 0
    )
    np.testing.assert_allclose(result @ np.array([1.0, 0.0]), [3, 2])
    with pytest.raises(ValueError):
        WeightEditor().surgical_rank_one_edit(base, np.zeros(2), np.ones(2), 0)


def test_antipodal_slerp_is_finite():
    result = TaskArithmeticEngine.slerp_interpolate(
        np.array([1.0, 0.0]), np.array([-1.0, 0.0]), 0.5
    )
    assert np.isfinite(result).all()


def test_magnitude_pruning_is_bounded_with_ties():
    assert (
        np.count_nonzero(ArchitectureMutator.prune_least_significant_neurons(np.ones((2, 5)), 0.2))
        == 8
    )
    with pytest.raises(ValueError):
        ArchitectureMutator.layer_stack_blueprint(4, [4])
    with pytest.raises(ValueError):
        ArchitectureMutator.scale_attention_heads(np.ones((5, 5)), 2, {})


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "loss.real",
        "[loss][0]",
        "loss ** 1000000",
        "loss ** step",
        "(lambda: 1)()",
        "True",
        "unknown + 1",
    ],
)
def test_objectives_never_execute_python(expression):
    with pytest.raises(ValueError):
        NumericObjective(expression)


def test_numeric_objective_and_invalid_arithmetic():
    assert NumericObjective("loss * beta / (1 + step)")(loss=2, beta=0.1, step=1) == 0.1
    with pytest.raises(ValueError):
        NumericObjective("loss / step")(loss=1, beta=1, step=0)


def test_mutator_does_not_change_strings_comments_or_exponent_suffix(tmp_path):
    code = '# beta=2\nmessage = "beta=3"\nloss = dpo(beta=1e-4)\n'
    mutator = CodeMutator()
    proposal = mutator.propose_loss_mutation(code, "dpo_beta", 0.05)
    assert '"beta=3"' in proposal and "# beta=2" in proposal and "1e-4" not in proposal
    mutator.export_proposal(code, proposal, tmp_path)
    assert not json.loads((tmp_path / "proposal.json").read_text())["executed"]
    assert (tmp_path / "proposal.patch").read_text().startswith("--- a/")
    with pytest.raises(ValueError):
        mutator.export_proposal(code, proposal, tmp_path, relative_path="../bad.py")
    with pytest.raises(ValueError):
        mutator.propose_loss_mutation("f(beta=1); f(beta=2)", "dpo_beta", 0.1)


def test_self_rewarding_requires_evidence_and_abstains():
    with pytest.raises(RuntimeError):
        SelfRewardingEngine().judge_generation("x", "Therefore step proof")
    engine = SelfRewardingEngine(judge=lambda *_: 0.5)
    assert engine.create_dpo_pair("x", "a", "b") is None
    with pytest.raises(ValueError):
        SelfRewardingEngine(judge=lambda *_: float("nan")).judge_generation("x", "y")


def test_wrong_high_score_never_becomes_a_star_label():
    engine = TestTimeComputeEngine()
    assert (
        engine.extract_star_training_pair(
            "x", "1", [{"final_answer": "10", "verifier_score": 0.99}]
        )
        is None
    )
    with pytest.raises(RuntimeError):
        engine.sample_rollouts("x")
    with pytest.raises(ValueError):
        engine.sample_rollouts("x", 0)
    ranked, margin = engine.verifier_rerank(
        [{"verifier_score": 0.9}, {"verifier_score": 0.8}, {"verifier_score": 0.1}]
    )
    assert margin == pytest.approx(0.1)  # second best, not worst
    assert ranked["verifier_score"] == 0.9


def test_adversarial_critic_cannot_accept_gibberish():
    critic = AdversarialCritic()
    for response in ("I have no idea", "18", "8", "9 or maybe 8"):
        assert not critic.audit_response(response, {"target": "9"})[0]
    assert critic.audit_response(" 9 ", {"target": "9"})[0]


def test_easyedit_missing_is_not_success():
    with patch.object(
        EasyEditBridge, "is_available", new_callable=PropertyMock, return_value=False
    ):
        with pytest.raises(RuntimeError, match="no edit"):
            EasyEditBridge().execute_direct_edit("unused", {})


def test_merge_recipes_no_hardcoded_layers_and_no_silent_execution(tmp_path):
    path = MergeKitBridge().generate_slerp_recipe("a", "b", tmp_path / "r.yml", 0.25)
    recipe = yaml.safe_load(open(path))
    assert "slices" not in recipe and recipe["parameters"]["t"] == 0.25
    with pytest.raises(PermissionError):
        MergeKitBridge().execute(path, tmp_path / "model")


class BlobBackend:
    def __init__(self, size):
        self.size = size

    def save(self, path):
        (path / "weights.dat").write_bytes(b"x" * self.size)

    def describe(self):
        return {"backend": "test-blob"}


def checkpoint(root, generation, accuracy, size, run="owned"):
    return CheckpointStore(root, run).save(BlobBackend(size), generation, scores={"a": accuracy})


def test_pareto_accounts_for_actual_size_and_preview(tmp_path):
    checkpoint(tmp_path, 0, 0.5, 100)
    checkpoint(tmp_path, 1, 0.7, 100)
    checkpoint(tmp_path, 2, 0.6, 10000)
    checkpoint(tmp_path, 3, 0.6, 10)
    pruner = ParetoCheckpointPruner(2)
    plan = pruner.plan(tmp_path, run_id="owned", protected=["generation_0000"])
    assert "generation_0002" in plan["delete"]
    assert "generation_0001" in plan["keep"] and "generation_0003" in plan["keep"]
    pruner.prune_checkpoints(tmp_path, run_id="owned")
    assert (tmp_path / "generation_0002").exists()


def test_pruner_never_deletes_unowned_unknown_symlink_or_corrupt(tmp_path):
    checkpoint(tmp_path, 0, 0.5, 10)
    checkpoint(tmp_path, 1, 0.6, 10)
    checkpoint(tmp_path, 2, 0.1, 500, run="foreign")
    corrupt = checkpoint(tmp_path, 3, 0.1, 100)
    (tmp_path / corrupt / "weights.dat").write_text("tampered")
    (tmp_path / "my-important-folder").mkdir()
    (tmp_path / "generation_0999").symlink_to(
        tmp_path / "my-important-folder", target_is_directory=True
    )
    deleted = ParetoCheckpointPruner(1).prune_checkpoints(tmp_path, run_id="owned", apply=True)
    assert deleted == ["generation_0000"]
    assert (tmp_path / "generation_0002").exists() and (tmp_path / "generation_0003").exists()
    assert (tmp_path / "my-important-folder").exists() and (
        tmp_path / "generation_0999"
    ).is_symlink()
    with pytest.raises(ValueError):
        ParetoCheckpointPruner().prune_checkpoints(tmp_path)
