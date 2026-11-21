"""Real offline CPU SFT, LoRA/checkpoint and worker supervision integration; no downloads.

A generated random tiny LM and trivial labels exercise plumbing, not pretrained capability.
"""

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest

pytestmark = pytest.mark.ml


@pytest.fixture
def foundation_model(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from scripts.run_foundation_fixture import create_random_model

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    directory = tmp_path / "tiny-foundation-model"
    create_random_model(directory)
    return directory


def test_real_audited_lora_steps_optimizer_rng_export_reload_and_base_frozen(
    foundation_model, tmp_path
):
    import torch
    from safetensors.torch import load_file

    from ouroboros.config import ModelConfig
    from ouroboros.core.checkpoints import verify_checkpoint
    from ouroboros.data import Sample
    from ouroboros.research.contracts import Budget, Recipe, identity
    from ouroboros.research.provenance import model_inventory
    from ouroboros.research.training import load_target, train_audited

    budget = Budget(max_steps=4, max_seq_length=64, max_new_tokens=4, checkpoint_every=2)
    model = ModelConfig(model_name_or_path=str(foundation_model), max_new_tokens=4)
    recipe = Recipe(steps=4, target_modules=["c_attn"], learning_rate=0.01, schedule="cosine")
    source = model_inventory(foundation_model)
    backend = load_target(model, budget, recipe=recipe, seed=17)
    frozen = {
        name: p.detach().clone()
        for name, p in backend.model.named_parameters()
        if not p.requires_grad
    }
    initial = backend.snapshot()
    samples = [Sample("Return ready only. train record 1", "ready", verified=True)]
    result = train_audited(
        backend,
        samples,
        recipe,
        budget,
        seed=17,
        root=tmp_path / "training",
        run_id="real-lora-fixture",
        source_binding=identity(source),
        deadline=time.monotonic() + 60,
    )
    assert result["steps"] == 4 and result["is_real_training"]
    assert not result["mid_step_resume"] and result["optimizer_reset_at_start"]
    for name, parameter in backend.model.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, frozen[name])
    after = backend.snapshot()
    assert any(
        not torch.equal(tensor, after["adapter"][name])
        for name, tensor in initial["adapter"].items()
    )
    marker = verify_checkpoint(result["checkpoint"])
    assert marker["metadata"]["source_binding"] == identity(source)
    traces = [json.loads(line) for line in Path(result["trace_file"]).read_text().splitlines()]
    assert len(traces) == 4 and traces[0]["learning_rate"] > traces[-1]["learning_rate"]
    assert all(
        row["gradient_norm_before_clip"] >= 0 and row["completion_tokens"] == 2 for row in traces
    )
    tensors = load_file(str(Path(result["checkpoint"]) / "training_state.safetensors"))
    assert "torch_cpu_rng" in tensors and any("exp_avg" in key for key in tensors)
    state = json.loads((Path(result["checkpoint"]) / "training_state.json").read_text())
    assert state["step"] == 4 and not state["resume_supported"]
    reloaded = load_target(model, budget, checkpoint=result["checkpoint"])
    assert backend.predict(["Return ready only."]) == reloaded.predict(["Return ready only."])
    assert model_inventory(foundation_model) == source


def test_training_context_is_full_and_not_silently_trimmed(foundation_model):
    from ouroboros.config import ModelConfig
    from ouroboros.data import Sample
    from ouroboros.research.contracts import Budget, Recipe
    from ouroboros.research.training import load_target, strict_rows

    budget = Budget(max_seq_length=16)
    backend = load_target(
        ModelConfig(model_name_or_path=str(foundation_model), max_new_tokens=4),
        budget,
        recipe=Recipe(target_modules=["c_attn"]),
    )
    with pytest.raises(ValueError, match="no silent truncation"):
        strict_rows(backend, [Sample("ready " * 20, "ready", verified=True)], budget)


def test_real_independent_worker_loop_on_constant_answer_fixture(foundation_model, tmp_path):
    from ouroboros.research.controller import run_research
    from ouroboros.research.memory import ExperimentMemory
    from ouroboros.research.project import init_foundation, verify_run
    from ouroboros.research.provenance import model_inventory

    config_path = Path(init_foundation(tmp_path / "project", foundation_model)["config"])
    config = json.loads(config_path.read_text())
    # Generous wall limit covers independent Torch imports on slower CI. Each model is tiny.
    config["budget"]["max_seconds"] = 600
    config["budget"]["max_seq_length"] = 64
    config["budget"]["max_rss_bytes"] = 1500 * 1024**2
    config["budget"]["cpu_threads"] = 1
    config["budget"]["max_steps"] = 50
    config["budget"]["checkpoint_every"] = 50
    config["followup_recipe"]["target_modules"] = ["c_attn"]
    config_path.write_text(json.dumps(config))
    before = model_inventory(foundation_model)
    report = run_research(config_path)
    assert report["state"] == "completed", report.get("reason")
    assert report["full_gate_passed"] and report["source_files_unchanged"]
    assert report["researcher_kind"] == "host_reference"
    assert not report["autonomous_improvement_demonstrated"]
    assert not report["improved_future_selection_demonstrated"]
    assert set(report["gates"]) == {"validation", "reserved", "final", "robustness", "retention"}
    assert len(report["promoted_checkpoints"]) == 2
    for role in ("validation", "reserved", "final"):
        measured = report["gates"][role]
        assert measured["alpha"] == 0.05 / 44 and measured["passed"]
        assert len(measured["replications"]) == 2
        assert measured["problem_clustered"]["baseline"]["accuracy"] == 0
        assert measured["problem_clustered"]["candidate"]["accuracy"] == 1
    pids = set()
    operations = []
    for receipt in report["workers"]:
        assert receipt["status"] == "completed" and receipt["exit_code"] == 0
        result = json.loads((Path(receipt["worker_root"]) / "result.json").read_text())
        pids.add(result["worker"]["pid"])
        operations.append(receipt["operation"])
    assert len(pids) == len(report["workers"]) and "followup" in operations
    assert report["planned_optimizer_steps_consumed"] <= config["budget"]["max_total_steps"]
    assert verify_run(report["report_file"])["status"] == "verified"
    memory = ExperimentMemory(config_path.parent / "experiments.sqlite3", readonly=True)
    assert memory.verify() and memory.search()[0]["useful"] == 1
    memory.close()
    assert model_inventory(foundation_model) == before


def test_worker_watchdogs_end_process_groups(foundation_model, tmp_path):
    from ouroboros.research.contracts import Budget
    from ouroboros.research.controller import WorkerExecutor
    from ouroboros.research.provenance import model_inventory, software_inventory

    # 64 MiB cannot accommodate Torch import. Parent watchdog must stop this real worker,
    # keep the partial logs/receipt, and never pretend an evaluator returned a score.
    root = tmp_path / "run"
    root.mkdir()
    config = {
        "model": {"model_name_or_path": str(foundation_model), "max_new_tokens": 4},
        "seed": 17,
        "budget": asdict(replace(Budget(), max_rss_bytes=64 * 1024**2)),
    }
    executor = WorkerExecutor(
        config,
        {},
        model_inventory(foundation_model),
        software_inventory(),
        root,
        "bounded-resource-test",
        time.monotonic() + 30,
    )
    with pytest.raises(MemoryError, match="RSS"):
        executor.run("observe")
    receipt = executor.receipts[0]
    assert receipt["status"] == "failed" and receipt["exit_code"] is not None
    assert (Path(receipt["worker_root"]) / "receipt.json").is_file()
