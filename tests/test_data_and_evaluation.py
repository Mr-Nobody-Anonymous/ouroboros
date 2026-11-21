import math

import pytest

from ouroboros.config import ModelConfig, RecursiveCycleConfig, TuningConfig
from ouroboros.core.anchor_memory import AnchorMemoryBank
from ouroboros.core.evaluator import SelfEvaluator
from ouroboros.data import (
    DatasetBundle,
    Sample,
    curate_proposals,
    demo_dataset,
    load_samples,
    unique_samples,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lora_r": 0},
        {"lora_r": 1.5},
        {"learning_rate": -1},
        {"learning_rate": float("nan")},
        {"lora_dropout": 1},
        {"num_train_epochs": 0},
        {"batch_size": True},
        {"target_modules": []},
        {"target_modules": "q_proj"},
        {"learning_rate": "fast"},
        {"lora_dropout": True},
        {"lora_r": 10**400},
    ],
)
def test_invalid_tuning(kwargs):
    with pytest.raises(ValueError):
        TuningConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_generations": 0},
        {"max_train_steps": 0},
        {"anchor_ratio": 1},
        {"max_seconds": float("nan")},
        {"seed": -1},
        {"min_free_disk_bytes": -1},
        {"min_free_disk_bytes": 0.5},
        {"seed": True},
        {"output_dir": ""},
    ],
)
def test_invalid_cycle(kwargs):
    with pytest.raises(ValueError):
        RecursiveCycleConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_qlora": True},
        {"device": "cuda-maybe"},
        {"device": None},
        {"use_qlora": "false"},
        {"model_name_or_path": 12},
        {"torch_dtype": []},
        {"max_new_tokens": 1.5},
    ],
)
def test_invalid_model_config(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


@pytest.mark.parametrize(
    "row",
    [
        {"instruction": "x"},
        {"instruction": "x", "target": " "},
        {"instruction": "x", "target": "y", "typo": 1},
        {"instruction": "x", "target": "y", "is_anchor": True},
    ],
)
def test_invalid_samples(row):
    with pytest.raises(ValueError):
        Sample.from_dict(row)


def test_split_contamination_is_normalized():
    with pytest.raises(ValueError, match="contamination"):
        DatasetBundle([Sample(" Ａ  B ", "yes")], [Sample("a b", "yes")])


def test_anchor_contamination_and_verification():
    with pytest.raises(ValueError, match="contamination"):
        DatasetBundle(
            [Sample("train", "x")],
            [Sample("heldout", "x")],
            [Sample("heldout", "x", verified=True)],
        )
    with pytest.raises(ValueError, match="verified"):
        DatasetBundle([Sample("train", "x")], [Sample("heldout", "x")], [Sample("anchor", "x")])


def test_dedup_conflict_and_fingerprint():
    assert len(unique_samples([Sample("a", "b"), Sample(" A ", "b")])) == 1
    with pytest.raises(ValueError, match="Conflicting"):
        unique_samples([Sample("a", "b"), Sample("A", "c")])
    assert demo_dataset().fingerprint() == demo_dataset().fingerprint()


def test_proposals_need_independent_correctness_and_do_not_leak():
    heldout = [Sample("secret", "4")]
    proposals = [Sample("Secret", "4"), Sample("new", "wrong"), Sample("other", "4")]
    accepted = curate_proposals(proposals, heldout, lambda s: s.target == "4")
    assert [s.instruction for s in accepted] == ["other"]
    assert accepted[0].verified
    assert not proposals[-1].verified


def test_jsonl_is_plain_readable_training_data(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"instruction":"hi", "target":"hello"}\n')
    assert load_samples(path)[0].target == "hello"


@pytest.mark.parametrize("ratio,count", [(0, 8), (0.25, 6), (0.5, 5), (0.75, 2)])
def test_replay_final_batch_ratio_and_immutability(ratio, count):
    original = [{"instruction": f"task {i}", "target": "x"} for i in range(count)]
    bank = AnchorMemoryBank(ratio)
    result = bank.blend_curriculum(original)
    anchors = sum(bool(s.get("is_anchor")) for s in result)
    assert anchors == math.ceil(count * ratio / (1 - ratio))
    assert len(original) == count
    if ratio:
        bank.anchors[0]["target"] = "corrupt"
        assert bank.anchors[0]["target"] != "corrupt"
    assert bank.blend_curriculum([]) == []


def test_zero_ratio_does_not_inject_anchor():
    assert AnchorMemoryBank(0, anchors=[]).blend_curriculum([{"instruction": "a"}]) == [
        {"instruction": "a"}
    ]
    with pytest.raises(ValueError):
        AnchorMemoryBank(0.25, anchors=[]).blend_curriculum([{"instruction": "a"}])


@pytest.mark.parametrize(
    "prediction,target,score",
    [("10", "1", 0), ("1", "10", 0), ("  BLUE ", "blue", 1), ("", "4", 0)],
)
def test_exact_match_is_not_substring(prediction, target, score):
    assert SelfEvaluator().evaluate_benchmark([prediction], [target]) == score


def test_invalid_evaluation_counts_and_empty_target():
    evaluator = SelfEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate_benchmark(["x"], [])
    with pytest.raises(ValueError):
        evaluator.evaluate_benchmark(["x"], [""])
    assert evaluator.evaluate_benchmark([], []) == 0


def test_promotion_fails_closed():
    evaluator = SelfEvaluator({"math": 0.5, "general": 0.8})
    assert not evaluator.verify_improvement({"math": 0.9})[0]
    assert not evaluator.verify_improvement({"math": 0.5, "general": 0.8})[0]
    assert not evaluator.verify_improvement({"math": 0.7, "general": 0.7})[0]
    assert not evaluator.verify_improvement(
        {"math": 0.6, "general": 0.8}, improvement_threshold=0.1
    )[0]
    assert evaluator.verify_improvement({"math": 0.7, "general": 0.8}, improvement_threshold=0.1)[0]
    with pytest.raises(ValueError):
        evaluator.verify_improvement({"math": float("nan"), "general": 0.8})


def test_compare_latest_and_original_to_prevent_slow_forgetting():
    evaluator = SelfEvaluator({"a": 0.8, "b": 0.5})
    previous = {"a": 0.76, "b": 0.7}
    assert not evaluator.verify_improvement({"a": 0.72, "b": 0.9}, previous_scores=previous)[0]
    assert not evaluator.verify_improvement(
        {"a": 0.8, "b": 0.6}, previous_scores={"a": 0.8, "b": 0.8}
    )[0]


def test_baseline_is_measured_not_invented():
    assert SelfEvaluator().baseline_scores == {}
    assert not SelfEvaluator().verify_improvement({"math": 0.99})[0]
