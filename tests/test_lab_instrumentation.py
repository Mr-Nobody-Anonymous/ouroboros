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


def test_model_stop_tokens_not_tokenizer_start_tokens_control_generation(
    research_backend, monkeypatch
):
    import torch

    from ouroboros.backends.tokens import align_generation_tokens

    backend = research_backend
    backend.tokenizer.eos_token = "[UNK]"  # reproduce the old converted SmolLM2 mismatch in memory
    backend.generation_tokens = align_generation_tokens(backend.model, backend.tokenizer)
    assert backend.generation_tokens["tokenizer_eos_before"] == 1
    assert backend.tokenizer.eos_token_id == 2
    before = model_fingerprint(backend.model)
    captured = {}

    def generated(**kwargs):
        captured.update(kwargs)
        tokens = backend.tokenizer.encode("a b", add_special_tokens=False) + [2]
        return torch.cat(
            [kwargs["input_ids"], torch.tensor([tokens], device=backend.device)], dim=1
        )

    monkeypatch.setattr(backend.model, "generate", generated)
    row = backend.generate_records(
        [{"prompt": "a", "seed": 1}], sampling=Sampling(), max_new_tokens=3
    )[0]
    assert captured["eos_token_id"] == [2] and captured["pad_token_id"] == 0
    assert row["text"] == "a b" and not row["cut"] and row["finish_reason"] == "eos"
    assert model_fingerprint(backend.model) == before


def test_practice_contrast_action_combines_real_llama_hooks_with_graded_source_records(
    research_backend, monkeypatch
):
    from ouroboros.data import DatasetBundle, Sample
    from ouroboros.lab.session import LabSession

    backend = research_backend
    dataset = DatasetBundle(
        [Sample("a b", "a b"), Sample("c d", "c d")],
        [Sample("warm a", "a")],
        test=[Sample("cool b", "b")],
    )

    # Explicit scripted completion fixture; hidden reads below are actual random-Llama forwards.
    def generated(requests, **kwargs):
        return [
            {"text": "a b", "new_tokens": 2, "cut": False, "thinking_closed": True}
            for request in requests
        ]

    monkeypatch.setattr(backend, "generate_records", generated)
    copy = WorkingCopy(backend.model, LabPolicy(max_read_tokens=32))
    session = LabSession(backend, copy, dataset, sampling=Sampling(), grader="exact")
    before = model_fingerprint(backend.model)
    result = session.act({"action": "contrast_practice", "module": "model.layers.0", "ids": [0, 1]})
    assert result["available"] and result["direction_id"] == "d0" and result["dimension"] == 16
    assert result["unit_norm"] == pytest.approx(1.0) and result["read_tokens_truncated"] == 0
    assert result["measurement"]["kind"] == "hidden" and len(result["measurement"]["texts"]) == 2
    assert (
        len(result["practice_records"]) == 2
        and sum(r["correct"] for r in result["practice_records"]) == 1
    )
    assert (
        result["counters"]["forward_calls"] == 2 and result["counters"]["generated_sequences"] == 2
    )
    assert model_fingerprint(backend.model) == before


@pytest.mark.parametrize("limit_source", ["policy", "model_context"])
def test_contrast_preflights_every_complete_text_before_any_forward_or_hook(
    research_backend, monkeypatch, limit_source
):
    from ouroboros.lab.instrumentation import ReadTextTooLong

    backend = research_backend
    policy = LabPolicy(max_read_tokens=2 if limit_source == "policy" else 256)
    if limit_source == "model_context":
        monkeypatch.setattr(backend, "context_size", 2)
    copy = WorkingCopy(backend.model, policy)
    module = dict(backend.model.named_modules())["model.layers.0"]
    hooks = dict(module._forward_hooks)
    before = model_fingerprint(backend.model)

    def forbidden(*args, **kwargs):
        raise AssertionError("No forward, including for the earlier short group, is allowed")

    monkeypatch.setattr(backend.model, "forward", forbidden)
    with pytest.raises(ReadTextTooLong, match="No hidden reads or direction") as error:
        measure_direction(
            copy, backend, module="model.layers.0", failing=["a b"], successful=["c d a b"]
        )
    assert (error.value.text_index, error.value.tokens, error.value.limit) == (1, 4, 2)
    assert not copy.directions and dict(module._forward_hooks) == hooks
    assert model_fingerprint(backend.model) == before


def test_practice_contrast_keeps_grades_but_refuses_a_completion_truncated_by_read_budget(
    research_backend, monkeypatch
):
    from ouroboros.data import DatasetBundle, Sample
    from ouroboros.lab.session import LabSession

    backend = research_backend
    dataset = DatasetBundle(
        [Sample("a b", "a b"), Sample("c d", "c d")],
        [Sample("warm a", "a")],
        test=[Sample("cool b", "b")],
    )

    # Scripted answers isolate the truncation boundary; no pretrained inference/gain is claimed.
    def generated(requests, **kwargs):
        return [
            {"text": "a b", "new_tokens": 2, "cut": False, "thinking_closed": True}
            for request in requests
        ]

    def forbidden(*args, **kwargs):
        raise AssertionError("Overlong contrast must not read a prompt-only prefix")

    monkeypatch.setattr(backend, "generate_records", generated)
    monkeypatch.setattr(backend.model, "forward", forbidden)
    copy = WorkingCopy(backend.model, LabPolicy(max_read_tokens=2))
    session = LabSession(backend, copy, dataset, sampling=Sampling(), grader="exact")
    before = model_fingerprint(backend.model)
    result = session.act({"action": "contrast_practice", "module": "model.layers.0", "ids": [0, 1]})
    assert not result["available"] and not result["read_performed"]
    assert "no input or completion was trimmed" in result["reason"]
    assert result["read_limit_tokens"] == 2 and result["overlong_text_tokens"] == 4
    assert sum(row["correct"] for row in result["practice_records"]) == 1
    assert [row["text"] for row in result["practice_records"]] == ["a b", "a b"]
    assert result["counters"]["forward_calls"] == 2  # failed read attempt remains metered
    assert result["counters"]["invalid_actions"] == 0
    assert "measurement" not in result and "direction_id" not in result and not copy.directions
    assert model_fingerprint(backend.model) == before
