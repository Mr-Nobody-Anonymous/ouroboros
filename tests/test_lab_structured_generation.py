"""Real local random-Llama constrained decoding, not pretrained capability evidence."""

import json

import pytest

from ouroboros.lab.action_schema import action_schema
from ouroboros.lab.config import Sampling
from ouroboros.lab.metrics import parse_action
from ouroboros.self_edit import model_fingerprint

pytestmark = pytest.mark.ml


def test_actual_transformers_token_masking_yields_json_without_changing_answers_or_weights(
    tmp_path,
):
    pytest.importorskip("lmformatenforcer")
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from ouroboros.config import ModelConfig
    from ouroboros.lab.backend import ResearchCausalLM

    characters = list(dict.fromkeys(chr(i) for i in range(32, 127))) + ["\n", "\r", "\t"]
    vocab = {token: i for i, token in enumerate(["[PAD]", "[UNK]", "[EOS]", *characters])}
    tokenizer = Tokenizer(BPE(vocab, merges=[], unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    assert (
        fast.decode(fast.encode('{"action":"withdraw"}', add_special_tokens=False))
        == '{"action":"withdraw"}'
    )
    fast.save_pretrained(tmp_path)
    with torch.random.fork_rng():
        torch.manual_seed(8)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=len(vocab),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=1024,
                pad_token_id=0,
                eos_token_id=2,
                bos_token_id=2,
            )
        )
    model.save_pretrained(tmp_path, safe_serialization=True)
    backend = ResearchCausalLM(ModelConfig(model_name_or_path=str(tmp_path), max_new_tokens=4))
    schema = action_schema(sealed=True)
    # Keep this mechanical grammar test short; production allows a bounded optional rationale.
    for branch in schema["anyOf"]:
        del branch["properties"]["rationale"]
    before, rng = model_fingerprint(backend.model), torch.get_rng_state().clone()
    flags = [module.training for module in backend.model.modules()]
    request = [{"prompt": "Return a JSON command.", "seed": 9}]
    first = backend.generate_records(
        request, sampling=Sampling(), max_new_tokens=64, json_schema=schema
    )[0]
    second = backend.generate_records(
        request, sampling=Sampling(), max_new_tokens=64, json_schema=schema
    )[0]
    assert first == second
    assert parse_action(first["text"]) in ({"action": "withdraw"}, {"action": "confirm"})
    assert first["decoding"] == "json_schema" and not first["cut"]
    assert json.loads(first["text"])["action"] in ("confirm", "withdraw")
    answer = backend.generate_records(request, sampling=Sampling())[0]
    assert answer["decoding"] == "unconstrained" and answer["new_tokens"] <= 4
    assert model_fingerprint(backend.model) == before and torch.equal(rng, torch.get_rng_state())
    assert flags == [module.training for module in backend.model.modules()]
