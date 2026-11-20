"""Real offline random-Llama forwards/hooks/sampling, not a pretrained capability evaluation."""

from dataclasses import replace

import numpy as np
import pytest

from ouroboros.config import ModelConfig
from ouroboros.lab.backend import ResearchCausalLM
from ouroboros.lab.config import LabPolicy, Sampling
from ouroboros.lab.instrumentation import measure_direction, rank_writers, read_states
from ouroboros.lab.program import build_program
from ouroboros.lab.state import WorkingCopy
from ouroboros.self_edit import model_fingerprint

pytestmark = pytest.mark.ml


@pytest.fixture
def research_backend(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    words = [
        "[PAD]",
        "[UNK]",
        "[EOS]",
        "<think>",
        "</think>",
        "a",
        "b",
        "c",
        "d",
        "Instruction",
        "Response",
        ":",
        "warm",
        "cool",
    ]
    tok = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
        additional_special_tokens=["<think>", "</think>"],
    )
    fast.save_pretrained(tmp_path)
    with torch.random.fork_rng():
        torch.manual_seed(5)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=len(words),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=256,
                pad_token_id=0,
                bos_token_id=2,
                eos_token_id=2,
            )
        )
    model.save_pretrained(tmp_path, safe_serialization=True)
    backend = ResearchCausalLM(ModelConfig(model_name_or_path=str(tmp_path), max_new_tokens=4))
    backend.model.set_attn_implementation("eager")
    return backend


def test_seeded_sampling_preserves_torch_rng_modes_and_full_weights(research_backend):
    import torch

    backend = research_backend
    before = model_fingerprint(backend.model)
    rng = torch.get_rng_state().clone()
    backend.model.train()
    flags = [module.training for module in backend.model.modules()]
    sampling = Sampling(temperature=0.6, top_p=0.95, top_k=5, presence_penalty=1.5)
    first = backend.generate_records([{"prompt": "a b", "seed": 23}], sampling=sampling)
    second = backend.generate_records([{"prompt": "a b", "seed": 23}], sampling=sampling)
    assert first == second and 1 <= first[0]["new_tokens"] <= 4
    assert torch.equal(rng, torch.get_rng_state())
    assert [module.training for module in backend.model.modules()] == flags
    assert model_fingerprint(backend.model) == before
    with pytest.raises(ValueError, match="context"):
        backend.generate_records([{"prompt": "a " * 300, "seed": 1}], sampling=Sampling())


def test_thinking_delimiters_are_not_discarded_as_special_tokens(research_backend, monkeypatch):
    import torch

    backend = research_backend

    def generated(**kwargs):
        ids = backend.tokenizer.encode("<think> a </think> b", add_special_tokens=False) + [
            backend.tokenizer.eos_token_id
        ]
        return torch.cat([kwargs["input_ids"], torch.tensor([ids], device=backend.device)], dim=1)

    monkeypatch.setattr(backend.model, "generate", generated)
    row = backend.generate_records(
        [{"prompt": "a", "seed": 1}], sampling=Sampling(thinking=True), max_new_tokens=8
    )[0]
    assert "</think>" in row["text"] and row["thinking_closed"] and not row["cut"]
    assert "[EOS]" not in row["text"]


def test_hidden_mlp_and_logprob_reads_match_real_forward_and_remove_hooks(research_backend):
    import torch

    backend, policy = research_backend, LabPolicy(max_read_tokens=32)
    before = model_fingerprint(backend.model)
    name = "model.layers.0"
    module = dict(backend.model.named_modules())[name]
    hooks = len(module._forward_hooks)
    result = read_states(
        backend,
        policy=policy,
        kind="hidden",
        texts=["a b c d"],
        modules=[name],
        pooling="positions",
        last=2,
    )
    with torch.inference_mode():
        ids = backend.tokenizer("a b c d", return_tensors="pt", return_token_type_ids=False)
        expected = backend.model(**ids, output_hidden_states=True, use_cache=False).hidden_states[
            1
        ][0, -2:, :]
    actual = result["texts"][0]["modules"][name]
    np.testing.assert_allclose(actual["values"], expected.numpy(), rtol=1e-6)
    assert actual["positions"] == [2, 3] and len(module._forward_hooks) == hooks
    mlp = read_states(
        backend,
        policy=policy,
        kind="mlp_input",
        texts=["a b c"],
        modules=["model.layers.0.mlp.down_proj"],
    )
    assert mlp["texts"][0]["modules"]["model.layers.0.mlp.down_proj"]["shape"] == [32]
    probs = read_states(backend, policy=policy, kind="logprobs", texts=["a b c"], last=0)
    assert len(probs["texts"][0]["token_logprobs"]) == 2
    assert all(
        np.isfinite(row["logprob"]) and row["logprob"] <= 0
        for row in probs["texts"][0]["token_logprobs"]
    )
    assert model_fingerprint(backend.model) == before


def test_attention_is_opt_in_bounded_and_actual_eager_maps(research_backend):
    backend = research_backend
    with pytest.raises(ValueError, match="explicit"):
        read_states(
            backend,
            policy=LabPolicy(),
            kind="attention",
            texts=["a b c"],
            modules=["model.layers.0.self_attn"],
        )
    policy = LabPolicy(attention_maps=True)
    result = read_states(
        backend,
        policy=policy,
        kind="attention",
        texts=["a b c"],
        modules=["model.layers.0.self_attn"],
    )
    row = result["texts"][0]["modules"]["model.layers.0.self_attn"]
    assert row["shape"] == [4, 1, 3]
    assert np.asarray(row["values"]).sum(axis=-1) == pytest.approx(np.ones((4, 1)))
    with pytest.raises(ValueError, match="allocation"):
        read_states(
            backend,
            policy=replace(policy, max_read_elements=1),
            kind="attention",
            texts=["a b c"],
            modules=["model.layers.0.self_attn"],
        )


def test_measured_contrast_and_writer_selection_feed_a_real_multi_layer_candidate(research_backend):
    backend = research_backend
    copy = WorkingCopy(backend.model, LabPolicy(max_edited_fraction=0.1))
    before = model_fingerprint(backend.model)
    with copy.use():
        direction = measure_direction(
            copy, backend, module="model.layers.0", failing=["a b"], successful=["c d"]
        )
    assert direction["direction_id"] == "d0" and direction["dimension"] == 16
    assert direction["unit_norm"] == pytest.approx(1)
    writers = rank_writers(copy, "model.layers.1.mlp.down_proj.weight", "d0", limit=4)
    assert len(writers["columns"]) == 4
    operations = [
        {
            "op": "project",
            "name": f"model.layers.{i}.mlp.down_proj.weight",
            "direction": "d0",
            "strength": 0.01,
            "top_columns": 1,
        }
        for i in (0, 1)
    ]
    copy.candidate, stats = build_program(copy, operations)
    assert stats["changed_entries"] > 0 and len(copy.candidate) == 2
    with copy.use():
        assert model_fingerprint(backend.model) != before
    assert model_fingerprint(backend.model) == before


def test_read_failures_remove_all_hooks_and_report_truncation(research_backend, monkeypatch):
    backend = research_backend
    name = "model.layers.0"
    module = dict(backend.model.named_modules())[name]
    count = len(module._forward_hooks)
    result = read_states(
        backend,
        policy=LabPolicy(max_read_tokens=2),
        kind="hidden",
        texts=["a b c d"],
        modules=[name],
    )
    assert result["texts"][0]["tokens_truncated"]
    with pytest.raises(ValueError, match="logits"):
        read_states(
            backend,
            policy=LabPolicy(max_logit_elements=1),
            kind="hidden",
            texts=["a b c"],
            modules=[name],
        )
    assert len(module._forward_hooks) == count

    def failing(*args, **kwargs):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(backend.model, "forward", failing)
    with pytest.raises(RuntimeError, match="forward failed"):
        read_states(backend, policy=LabPolicy(), kind="hidden", texts=["a b c"], modules=[name])
    assert len(module._forward_hooks) == count
