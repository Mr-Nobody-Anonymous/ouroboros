"""Real CPU LoRA training on a generated tiny causal LM; no Hub access or pretrained downloads."""

import time

import pytest

pytestmark = pytest.mark.ml


@pytest.fixture
def local_model(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(1)
    vocabulary = {
        word: i
        for i, word in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[EOS]",
                "Instruction",
                ":",
                "Response",
                "What",
                "is",
                "2",
                "+",
                "4",
                "3",
                "6",
                "?",
                "classify",
                "red",
                "blue",
                "warm",
                "cool",
                "please",
            ]
        )
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    directory = tmp_path / "tiny-local-model"
    directory.mkdir()
    fast.save_pretrained(directory)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(vocabulary),
                n_positions=64,
                n_embd=16,
                n_layer=1,
                n_head=2,
                bos_token_id=2,
                eos_token_id=2,
                pad_token_id=0,
            )
        )
    model.save_pretrained(directory, safe_serialization=True)
    return directory


def backend_at(path, adapter_path=None):
    from ouroboros.backends.transformers_backend import TransformersLoRABackend
    from ouroboros.config import ModelConfig, TuningConfig

    tuning = TuningConfig(
        lora_r=2,
        lora_alpha=4,
        target_modules=["c_attn"],
        learning_rate=0.01,
        num_train_epochs=2,
        batch_size=2,
        max_seq_length=32,
    )
    model = ModelConfig(model_name_or_path=str(path), max_new_tokens=4)
    return TransformersLoRABackend(model, tuning, seed=7, adapter_path=adapter_path), tuning


def test_real_local_lora_updates_masks_and_rollback(local_model):
    import torch

    from ouroboros.data import Sample

    backend, config = backend_at(local_model)
    samples = [Sample("What is 2 + 2 ?", "4"), Sample("What is 3 + 3 ?", "6")]
    ids, labels = backend.encode_sample(samples[0], 32)
    assert len(ids) == len(labels)
    assert -100 in labels and labels[-1] == backend.tokenizer.eos_token_id
    assert sum(value != -100 for value in labels) == 2
    before = backend.snapshot()
    frozen = {
        name: p.detach().clone()
        for name, p in backend.model.named_parameters()
        if not p.requires_grad
    }
    stats = backend.train(samples, config, max_steps=2, deadline=time.monotonic() + 20)
    assert stats["steps"] == 2 and stats["loss"] > 0
    after = backend.snapshot()
    assert any(
        not torch.equal(value, after["adapter"][name]) for name, value in before["adapter"].items()
    )
    for name, parameter in backend.model.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, frozen[name])
    backend.restore(before)
    for name, value in before["adapter"].items():
        assert torch.equal(value, backend.snapshot()["adapter"][name])
    assert len(backend.predict([samples[0].instruction])) == 1


def test_local_export_reload_and_checksums(local_model, tmp_path):
    import torch

    from ouroboros.core.checkpoints import CheckpointStore, verify_checkpoint
    from ouroboros.data import Sample

    backend, config = backend_at(local_model)
    backend.train(
        [Sample("classify red", "warm")], config, max_steps=1, deadline=time.monotonic() + 20
    )
    name = CheckpointStore(tmp_path / "checkpoints", "llm-test").save(
        backend, 1, scores={"test": 0.0}
    )
    directory = tmp_path / "checkpoints" / name
    verify_checkpoint(directory)
    assert (directory / "adapter_model.safetensors").is_file()
    other, _ = backend_at(local_model, directory)
    for key, tensor in backend.snapshot()["adapter"].items():
        assert torch.equal(tensor, other.snapshot()["adapter"][key])
    assert backend.predict(["classify red"]) == other.predict(["classify red"])
    assert "model.safetensors" in backend.describe()["base_model_files"]
    assert "tokenizer.json" in backend.describe()["base_model_files"]
    # Even a valid JSON edit to the base configuration must not silently change provenance.
    config_path = local_model / "config.json"
    config_path.write_text(config_path.read_text() + "\n")
    with pytest.raises(ValueError, match="provenance"):
        backend_at(local_model, directory)


def test_pickle_only_local_models_and_adapters_are_rejected(local_model, tmp_path):
    import shutil

    adapter = tmp_path / "unsafe-adapter"
    adapter.mkdir()
    (adapter / "adapter_model.bin").write_bytes(b"not executed")
    with pytest.raises(ValueError, match="safetensors"):
        backend_at(local_model, adapter)
    shutil.move(local_model / "model.safetensors", local_model / "pytorch_model.bin")
    with pytest.raises(ValueError, match="safetensors"):
        backend_at(local_model)


def test_sft_cannot_silently_erase_target(local_model):
    from ouroboros.data import Sample

    backend, _ = backend_at(local_model)
    with pytest.raises(ValueError, match="Completion"):
        backend.encode_sample(Sample("x", "4 4 4 4 4 4 4 4"), 4)


def test_torch_weight_edit_is_bounded_and_reversible():
    torch = pytest.importorskip("torch")
    import numpy as np

    from ouroboros.core.weight_space_programmer import WeightSpaceProgrammer

    model = torch.nn.Linear(8, 8, bias=False)
    original = model.weight.detach().clone()
    programmer = WeightSpaceProgrammer(0.125)
    edit = programmer.apply_to_torch_model(model, "weight", np.ones(8), strength=0.5)
    assert 0 < edit["stats"]["edited_weights"] <= 8
    programmer.restore_torch_edit(model, edit)
    assert torch.equal(model.weight, original)


def test_structural_candidate_is_separate_and_updates_cache_indices():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    from ouroboros.core.architecture_mutator import ArchitectureMutator

    torch.set_num_threads(1)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    candidate = ArchitectureMutator.build_layer_stacked_candidate(model, [0])
    assert len(model.model.layers) == 2 and len(candidate.model.layers) == 3
    assert candidate.config.num_hidden_layers == 3
    assert candidate.model.layers[2].self_attn.layer_idx == 2
    with torch.inference_mode():
        result = candidate(torch.tensor([[1, 2, 3]]))
    assert result.logits.shape == (1, 3, 16)


def test_shared_bare_model_loader_runs_real_weight_trial_and_cli(local_model, tmp_path, capsys):
    import json

    import numpy as np
    import torch

    from ouroboros.backends.local_model import LocalCausalLM
    from ouroboros.cli import main
    from ouroboros.config import ModelConfig
    from ouroboros.core.weight_trials import WeightEditPlan, run_weight_edit_trial
    from ouroboros.data import Sample

    backend = LocalCausalLM(ModelConfig(model_name_or_path=str(local_model), max_new_tokens=4))
    assert not any(p.requires_grad for p in backend.model.parameters())
    name = "transformer.h.0.mlp.c_proj.weight"
    parameter = dict(backend.model.named_parameters())[name]
    original = parameter.detach().clone()
    # Unreachable canonical label: the real random LM cannot earn credit from a style heuristic.
    validation = [Sample("What is 2 + 2 ?", "unreachable-canonical-answer", "math_logic")]
    plan = WeightEditPlan(name, edited_fraction=0.01, strength=0.25, max_relative_norm=1)
    result = run_weight_edit_trial(
        backend.model,
        backend.predict,
        validation,
        np.ones(parameter.shape[0]),
        plan,
        tmp_path / "api-trials",
        provenance="generated random GPT-2 integration fixture",
        license_name="MIT",
        model_info=backend.describe(),
    )
    assert result["status"] == "rejected" and not result["passed"]
    assert result["baseline"]["examples"] == result["candidate"]["examples"] == 1
    assert torch.equal(parameter, original) and result["model_restored"]

    (tmp_path / "validation.json").write_text(json.dumps([s.as_dict() for s in validation]))
    (tmp_path / "direction.json").write_text(json.dumps([1.0] * parameter.shape[0]))
    config = {
        "model": {"model_name_or_path": str(local_model), "max_new_tokens": 4},
        "edit": {"parameter_name": name, "edited_fraction": 0.01, "max_relative_norm": 1},
        "validation": "validation.json",
        "direction_file": "direction.json",
        "output_dir": "cli-trials",
        "provenance": "generated random GPT-2 integration fixture",
        "license": "MIT",
    }
    path = tmp_path / "trial.json"
    path.write_text(json.dumps(config))
    assert main(["weights", "trial", "--config", str(path)]) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["model_restored"] and response["status"] == "rejected"
    assert main(["weights", "trial", "--config", str(path), "--list-parameters"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert any(row["name"] == name for row in listed["parameters"])
    assert listed["edited"] is False
