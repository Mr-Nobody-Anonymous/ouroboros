import json
import time
from unittest.mock import patch

import numpy as np
import pytest

from ouroboros.backends.numpy_backend import NumpyLoRABackend
from ouroboros.config import RecursiveCycleConfig, TuningConfig
from ouroboros.core.checkpoints import CheckpointStore, verify_checkpoint
from ouroboros.data import demo_dataset
from ouroboros.pipeline.self_loop import RecursiveTuningLoop


def tuning():
    return TuningConfig(
        lora_r=4, lora_alpha=4, learning_rate=0.5, num_train_epochs=30, batch_size=16
    )


def loop_at(root, **kwargs):
    return RecursiveTuningLoop(
        RecursiveCycleConfig(
            max_generations=3, checkpoint_dir=str(root / "c"), output_dir=str(root / "o")
        ),
        tuning(),
        **kwargs,
    )


def test_real_gradients_and_rollback_state():
    dataset = demo_dataset()
    backend = NumpyLoRABackend(dataset.train, rank=4, alpha=4)
    before = backend.snapshot()
    result = backend.train(dataset.train, tuning(), max_steps=100, deadline=time.monotonic() + 10)
    assert result["steps"] > 0 and np.isfinite(result["loss"])
    assert not np.array_equal(backend.b, before["b"])
    backend.restore(before)
    np.testing.assert_array_equal(backend.a, before["a"])
    np.testing.assert_array_equal(backend.b, before["b"])
    assert backend.rng.bit_generator.state == before["rng"]


def test_measured_loop_promotes_and_really_rolls_back_plateau(tmp_path):
    loop = loop_at(tmp_path)
    history = loop.run_cycle()
    assert history[0]["passed"]
    assert not history[1]["passed"]
    assert history[-1]["scores"] == {"warm_colors": 1, "cool_colors": 1}
    run = json.loads((loop.run_directory / "run.json").read_text())
    checkpoint = loop.checkpoint_directory / run["latest_checkpoint"]
    metadata = verify_checkpoint(checkpoint)
    assert metadata["parent"] == "generation_0000"
    assert metadata["files"]["adapter.npz"]
    restored = NumpyLoRABackend.load(checkpoint)
    np.testing.assert_array_equal(loop.backend.a, restored.a)
    np.testing.assert_array_equal(loop.backend.b, restored.b)
    assert (
        loop.backend.predict([s.instruction for s in loop.dataset.validation])
        == loop.history[-1]["candidate_predictions"]
    )
    with pytest.raises(RuntimeError):
        loop.run_cycle()


def test_reproducible_seed_with_separate_artifacts(tmp_path):
    first, second = loop_at(tmp_path / "a"), loop_at(tmp_path / "b")
    a, b = first.run_cycle(), second.run_cycle()
    assert [r["scores"] for r in a] == [r["scores"] for r in b]
    assert [r["training"]["loss"] for r in a] == [r["training"]["loss"] for r in b]
    np.testing.assert_array_equal(first.backend.a, second.backend.a)
    assert first.run_id != second.run_id


def test_training_exception_restores_actual_weights(tmp_path):
    backend = NumpyLoRABackend(demo_dataset().train)
    initial = backend.snapshot()

    def fail(*args, **kwargs):
        backend.a[:] = 1234
        backend.b[:] = 99
        raise RuntimeError("training failed")

    backend.train = fail
    loop = loop_at(tmp_path, backend=backend)
    history = loop.run_cycle()
    assert history[0]["error"] and not history[0]["passed"]
    np.testing.assert_array_equal(backend.a, initial["a"])
    np.testing.assert_array_equal(backend.b, initial["b"])
    assert json.loads((loop.run_directory / "run.json").read_text())["status"] == "failed"


def test_keyboard_interrupt_restores_weights_and_records_cancellation(tmp_path):
    backend = NumpyLoRABackend(demo_dataset().train)
    initial = backend.snapshot()

    def interrupt(*args, **kwargs):
        backend.a[:] = 42
        backend.rng.random()
        raise KeyboardInterrupt()

    backend.train = interrupt
    loop = loop_at(tmp_path, backend=backend)
    with pytest.raises(KeyboardInterrupt):
        loop.run_cycle()
    np.testing.assert_array_equal(backend.a, initial["a"])
    assert backend.rng.bit_generator.state == initial["rng"]
    report = json.loads((loop.run_directory / "run.json").read_text())
    assert report["status"] == "interrupted"
    assert report["latest_checkpoint"] == "generation_0000"
    assert not report["history"][0]["passed"]


def test_checkpoint_export_failure_is_not_promoted(tmp_path):
    loop = loop_at(tmp_path)
    initial = loop.backend.snapshot()
    save = CheckpointStore.save

    def fail_candidate(store, backend, generation, **kwargs):
        if generation:
            raise OSError("disk full")
        return save(store, backend, generation, **kwargs)

    with patch.object(CheckpointStore, "save", fail_candidate):
        history = loop.run_cycle()
    assert history[0]["error"]
    np.testing.assert_array_equal(loop.backend.a, initial["a"])
    assert len(list(loop.checkpoint_directory.iterdir())) == 1


def test_no_optimization_when_deadline_exhausted():
    dataset = demo_dataset()
    backend = NumpyLoRABackend(dataset.train)
    before = backend.snapshot()
    result = backend.train(dataset.train, tuning(), max_steps=100, deadline=time.monotonic() - 1)
    assert result["steps"] == 0 and result["budget_exhausted"]
    np.testing.assert_array_equal(before["a"], backend.a)


def test_no_unverified_self_training(tmp_path):
    with pytest.raises(ValueError, match="independent"):
        loop_at(tmp_path, proposal_provider=lambda *args: [])


def test_proposals_are_bounded_and_verified(tmp_path):
    from ouroboros.data import Sample

    seen = []

    def provider(domain, training):
        assert all(s.instruction != "please classify red" for s in training)
        for i in range(100):
            yield Sample(f"red sample {i}", "warm", "warm_colors", "teacher-proposal")

    def verifier(sample):
        seen.append(sample)
        return True

    loop = loop_at(tmp_path, proposal_provider=provider, proposal_verifier=verifier)
    loop.run_cycle(1)
    assert len(seen) == 32
    assert loop.history[0]["verified_proposals"] == 32


def test_tampered_checkpoint_rejected(tmp_path):
    backend = NumpyLoRABackend(demo_dataset().train)
    store = CheckpointStore(tmp_path, "test-run")
    name = store.save(backend, 0, scores={"a": 0.5})
    file = tmp_path / name / "backend.json"
    file.write_text(file.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(tmp_path / name)


def test_checkpoints_cannot_overwrite_or_escape(tmp_path):
    backend = NumpyLoRABackend(demo_dataset().train)
    store = CheckpointStore(tmp_path, "owned")
    store.save(backend, 0, scores={"a": 0.5})
    with pytest.raises(FileExistsError):
        store.save(backend, 0, scores={"a": 0.5})
    with pytest.raises(ValueError):
        store.save(backend, -1, scores={"a": 0.5})


def test_failed_atomic_export_leaves_no_partial_checkpoint(tmp_path):
    backend = NumpyLoRABackend(demo_dataset().train)
    backend.save = lambda path: (_ for _ in ()).throw(OSError("disk full"))
    with pytest.raises(OSError):
        CheckpointStore(tmp_path, "test").save(backend, 1, scores={"a": 0.5})
    assert list(tmp_path.iterdir()) == []
