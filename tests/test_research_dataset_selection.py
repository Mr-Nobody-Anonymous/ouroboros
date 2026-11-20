"""Owned-dataset and selection-policy tests. Synthetic episodes are never evidence."""

import json

import pytest

from ouroboros.research import dataset as owned
from ouroboros.research.contracts import ROLES
from ouroboros.research.memory import ExperimentMemory
from ouroboros.research.selection import (
    POLICIES,
    analyse,
    compare,
    compare_fixture,
    design,
    fixture_episodes,
)


def test_owned_dataset_is_balanced_unique_and_verifiable(tmp_path):
    root = tmp_path / "data"
    written = owned.write_dataset(root, seed=17, model_path=tmp_path / "model")
    assert written["counts"] == owned.ROLE_SIZES
    assert sum(written["counts"].values()) == 380
    assert set(written["domains"]) == set(owned.DOMAINS)
    # Every role file is present, and the manifest carries hashes plus honest non-claims.
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["generator"] == owned.GENERATOR
    assert set(manifest["files"]) == set(ROLES)
    assert len(manifest["non_claims"]) >= 4
    assert "not a benchmark" in manifest["non_claims"][0]
    # Prompts are globally unique, so a training item can never double as hidden evidence.
    prompts = [
        row["instruction"]
        for role in ROLES
        for row in json.loads((root / "data" / f"{role}.json").read_text())
    ]
    assert len(prompts) == len(set(prompts)) == 380
    assert owned.answer_shape_check(owned.build(17)) is True
    assert owned.verify_dataset(root)["status"] == "verified"


def test_owned_dataset_refuses_overwrite_and_detects_tampering(tmp_path):
    root = tmp_path / "data"
    owned.write_dataset(root, seed=17, model_path=tmp_path / "model")
    with pytest.raises(FileExistsError):
        owned.write_dataset(root, seed=17, model_path=tmp_path / "model")
    path = root / "data" / "final.json"
    rows = json.loads(path.read_text())
    rows[0]["target"] = "999999"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="changed"):
        owned.verify_dataset(root)


def test_owned_dataset_answers_are_not_a_constant_and_cover_domains():
    roles = owned.build(17)
    targets = {sample.target for sample in roles["train"]}
    # A constant answer would let the target score without reading the prompt.
    assert len(targets) > 20
    assert {"yes", "no"} <= {sample.target for row in roles.values() for sample in row}
    domains = {sample.domain for sample in roles["train"]}
    assert domains == set(owned.DOMAINS)


def test_selection_design_is_frozen_before_analysis():
    frozen = design(alpha=0.05, min_episodes=20, seed=17)
    assert tuple(frozen["policies"]) == POLICIES
    assert frozen["frozen_before_analysis"] is True
    assert frozen["online_learning"] is False
    assert frozen["information_set"] == "only outcomes from strictly earlier episodes"
    with pytest.raises(ValueError):
        design(primary="proposal_order", null="proposal_order")
    with pytest.raises(ValueError):
        design(min_episodes=2)


def test_selection_comparison_refuses_small_samples():
    frozen = design(min_episodes=20)
    result = compare(fixture_episodes(episodes=6, seed=17), frozen=frozen)
    assert result["status"] == "insufficient_data"
    assert result["improved_future_selection_demonstrated"] is False
    assert result["episodes"] == 6 and result["required"] == 20


def test_selection_detects_a_known_advantage_and_stays_quiet_otherwise():
    strong = compare_fixture(episodes=60, advantage=0.9, seed=17)
    assert strong["status"] == "estimated"
    assert strong["decision"] == "policy_better_on_replay"
    assert strong["primary_minus_null"]["episodes_null_better"] == 0
    assert strong["scientific_evidence"] is False  # synthetic machinery check only
    weak = compare_fixture(episodes=24, advantage=0.5, seed=17)
    assert weak["decision"] == "not_established"


def test_candidate_log_replay_never_sees_the_future(tmp_path):
    memory = ExperimentMemory(tmp_path / "experiments.sqlite3")
    history = []
    for index in range(24):
        experiment = f"research-{index:04d}"
        memory.register(experiment, {"index": index})
        memory.transition(experiment, "failed", {"index": index})
        memory.record_candidates(
            experiment,
            [
                {
                    "family": "strong" if index % 2 == 0 else "weak",
                    "recipe_sha256": f"{index:064d}",
                    "practice": 0.6 if index % 2 == 0 else 0.1,
                    "baseline": 0.0,
                    "selected": index % 2 == 0,
                    "gate_passed": False,
                },
                {
                    "family": "weak" if index % 2 == 0 else "strong",
                    "recipe_sha256": f"{index + 1000:064d}",
                    "practice": 0.7 if index % 2 == 0 else 0.2,
                    "baseline": 0.0,
                    "selected": index % 2 != 0,
                    "gate_passed": False,
                },
            ],
            context="c" * 64,
        )
        history.append(experiment)
    episodes = memory.candidate_episodes()
    assert [episode["experiment_id"] for episode in episodes] == history
    assert all(len(episode["candidates"]) == 2 for episode in episodes)
    memory.verify()
    result = analyse(memory, directory=tmp_path / "frozen", min_episodes=20)
    assert result["status"] == "estimated"
    assert result["design_binding"]["sha256"]
    # Design is written once, before the replay; a second freeze is refused.
    with pytest.raises(FileExistsError):
        analyse(memory, directory=tmp_path / "frozen")
    memory.close()


def test_candidate_log_rejects_a_second_selection(tmp_path):
    memory = ExperimentMemory(tmp_path / "experiments.sqlite3")
    memory.register("research-0001", {"index": 0})
    memory.transition("research-0001", "failed", {"index": 0})
    with pytest.raises(ValueError, match="At most one candidate"):
        memory.record_candidates(
            "research-0001",
            [
                {
                    "family": "lora_sft",
                    "recipe_sha256": "a" * 64,
                    "practice": 0.2,
                    "baseline": 0.0,
                    "selected": True,
                    "gate_passed": False,
                },
                {
                    "family": "lora_sft",
                    "recipe_sha256": "b" * 64,
                    "practice": 0.1,
                    "baseline": 0.0,
                    "selected": True,
                    "gate_passed": False,
                },
            ],
            context="c" * 64,
        )
    memory.close()
