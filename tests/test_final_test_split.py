import hashlib
import json

import pytest

from ouroboros.backends.numpy_backend import NumpyLoRABackend
from ouroboros.cli import main
from ouroboros.config import RecursiveCycleConfig, TuningConfig
from ouroboros.core.checkpoints import verify_checkpoint
from ouroboros.core.evaluator import evaluate_samples
from ouroboros.data import DatasetBundle, Sample, demo_dataset
from ouroboros.pipeline.self_loop import RecursiveTuningLoop


def loop_for(tmp_path, dataset=None, **kwargs):
    return RecursiveTuningLoop(
        RecursiveCycleConfig(
            max_generations=3, checkpoint_dir=str(tmp_path / "c"), output_dir=str(tmp_path / "o")
        ),
        TuningConfig(lora_r=4, lora_alpha=4, learning_rate=0.5, num_train_epochs=30, batch_size=16),
        dataset=dataset or demo_dataset(),
        **kwargs,
    )


@pytest.mark.parametrize("collision", ["train", "validation", "anchor"])
def test_final_test_must_be_disjoint_from_every_selection_split(collision):
    train = [Sample("train", "0")]
    validation = [Sample("validation", "0")]
    anchors = [Sample("anchor", "0", verified=True, is_anchor=True)]
    with pytest.raises(ValueError, match="contamination"):
        DatasetBundle(train, validation, anchors, [Sample(collision.upper(), "0")])


def test_fingerprint_covers_test_and_preserves_old_no_test_payload():
    bundle = DatasetBundle([Sample("train", "0")], [Sample("validation", "0")])
    old_payload = {
        key: [s.as_dict() for s in getattr(bundle, key)]
        for key in ("train", "validation", "anchors")
    }
    assert (
        bundle.fingerprint()
        == hashlib.sha256(json.dumps(old_payload, sort_keys=True).encode()).hexdigest()
    )
    independent = DatasetBundle(
        bundle.train, bundle.validation, test=[Sample("final", "1", "unseen-domain")]
    )
    assert independent.fingerprint() != bundle.fingerprint()


def test_final_test_is_once_after_selection_and_cannot_train_or_promote(tmp_path):
    dataset = demo_dataset()
    backend = NumpyLoRABackend([*dataset.train, *dataset.anchors], rank=4, alpha=4)
    final_prompts = [s.instruction for s in dataset.test]
    original_predict, original_train = backend.predict, backend.train
    events, verifier_inputs = [], []

    def predict(prompts):
        if prompts == final_prompts:
            assert len(loop.history) == 3
            events.append("test")
            return ["incorrect-final-answer"] * len(prompts)
        events.append("validation")
        return original_predict(prompts)

    def train(samples, *args, **kwargs):
        assert not any(s.instruction in final_prompts for s in samples)
        events.append("train")
        return original_train(samples, *args, **kwargs)

    def propose(domain, training):
        assert not any(s.instruction in final_prompts for s in training)
        # Even an explicitly adversarial proposer cannot inject normalized final-test overlaps.
        return [
            Sample(
                dataset.test[0].instruction.upper(), dataset.test[0].target, dataset.test[0].domain
            )
        ]

    def verify(sample):
        verifier_inputs.append(sample)
        return True

    backend.predict, backend.train = predict, train
    loop = loop_for(
        tmp_path, dataset, backend=backend, proposal_provider=propose, proposal_verifier=verify
    )
    loop.run_cycle()
    assert not verifier_inputs  # blocked before exposing that proposed target to the verifier
    assert events[-1] == "test" and events.count("test") == 1
    assert loop.history[0]["passed"]
    assert loop.report["final_scores"] == {"warm_colors": 1.0, "cool_colors": 1.0}
    assert loop.report["final_test"]["scores"] == {"warm_colors": 0.0, "cool_colors": 0.0}
    assert loop.report["final_test"]["used_for_promotion"] is False
    assert loop.report["final_test"]["domain_counts"] == {"warm_colors": 2, "cool_colors": 2}
    assert loop.report["latest_checkpoint"] == "generation_0001"


def test_absent_final_test_is_explicitly_reported(tmp_path):
    data = demo_dataset()
    loop = loop_for(tmp_path, DatasetBundle(data.train, data.validation, data.anchors))
    loop.run_cycle(1)
    assert loop.report["final_test"] == {"status": "not_configured", "used_for_promotion": False}


def test_final_test_error_is_nonzero_cli_exit_but_does_not_undo_accepted_model(
    tmp_path, monkeypatch
):
    original = NumpyLoRABackend.predict

    def predict(self, prompts):
        if any("label only" in prompt for prompt in prompts):
            raise RuntimeError("final grader unavailable")
        return original(self, prompts)

    monkeypatch.setattr(NumpyLoRABackend, "predict", predict)
    assert (
        main(["demo", "--output-dir", str(tmp_path / "o"), "--checkpoint-dir", str(tmp_path / "c")])
        == 1
    )
    (path,) = (tmp_path / "o").glob("*/run.json")
    report = json.loads(path.read_text())
    assert report["status"] == "final_test_failed"
    assert report["final_test"]["status"] == "failed"
    assert report["history"][0]["passed"] and report["latest_checkpoint"] == "generation_0001"
    verify_checkpoint(tmp_path / "c" / report["run_id"] / report["latest_checkpoint"])


def test_final_test_interrupt_is_persisted_and_propagated(tmp_path):
    loop = loop_for(tmp_path)
    original = loop.backend.predict

    def predict(prompts):
        if prompts == [s.instruction for s in loop.dataset.test]:
            raise KeyboardInterrupt()
        return original(prompts)

    loop.backend.predict = predict
    with pytest.raises(KeyboardInterrupt):
        loop.run_cycle(1)
    assert loop.report["status"] == "interrupted"
    assert loop.report["final_test"]["status"] == "interrupted"
    assert loop.report["latest_checkpoint"] == "generation_0001"


def test_failed_training_does_not_consult_test_split(tmp_path):
    loop = loop_for(tmp_path)
    calls = []
    original = loop.backend.predict

    def predict(prompts):
        calls.append(prompts)
        return original(prompts)

    def fail(*args, **kwargs):
        raise RuntimeError("no candidate")

    loop.backend.predict, loop.backend.train = predict, fail
    loop.run_cycle()
    assert len(calls) == 1
    assert loop.report["final_test"]["status"] == "skipped_after_failure"


@pytest.mark.parametrize("predictions", [["0"], "00", [0, 0], ["0", "0", "0"]])
def test_scoring_refuses_truncation_and_invalid_prediction_types(predictions):
    with pytest.raises(ValueError, match="predictions"):
        evaluate_samples(lambda prompts: predictions, [Sample("a", "0"), Sample("b", "0")])


def test_budget_limited_selection_still_evaluates_the_selected_model_once(tmp_path, monkeypatch):
    from types import SimpleNamespace

    loop = loop_for(tmp_path)
    ticks = iter([0.0, 1_000_000.0, 1_000_000.0])
    # Replace only the controller's clock binding, not the process-wide time module.
    monkeypatch.setattr(
        "ouroboros.pipeline.self_loop.time", SimpleNamespace(monotonic=lambda: next(ticks))
    )
    original = loop.backend.predict
    calls = []

    def predict(prompts):
        calls.append(prompts)
        return original(prompts)

    loop.backend.predict = predict
    loop.run_cycle()
    assert loop.history == []
    assert loop.report["status"] == "budget_exhausted"
    assert loop.report["latest_checkpoint"] == "generation_0000"
    assert loop.report["final_test"]["status"] == "completed"
    assert calls == [
        [s.instruction for s in loop.dataset.validation],
        [s.instruction for s in loop.dataset.test],
    ]
