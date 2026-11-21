"""Stop-token resolution is lightweight and does not require Torch or model downloads."""

from types import SimpleNamespace

import pytest

from ouroboros.backends.tokens import align_generation_tokens


class Tokenizer:
    def __init__(self, eos=1, pad=None):
        self.names = ["[PAD]", "[START]", "[END]", "[OTHER_END]"]
        self.eos_token = None if eos is None else self.names[eos]
        self.pad_token = None if pad is None else self.names[pad]

    def __len__(self):
        return len(self.names)

    @property
    def eos_token_id(self):
        return None if self.eos_token is None else self.names.index(self.eos_token)

    @property
    def pad_token_id(self):
        return None if self.pad_token is None else self.names.index(self.pad_token)

    def convert_ids_to_tokens(self, value):
        return self.names[value]

    def convert_tokens_to_ids(self, value):
        return self.names.index(value)


def model(eos=2, generation=2):
    return SimpleNamespace(
        config=SimpleNamespace(vocab_size=4, eos_token_id=eos, pad_token_id=2),
        generation_config=SimpleNamespace(eos_token_id=generation, pad_token_id=2),
    )


def test_model_declared_end_token_wins_over_converted_tokenizer_start_token():
    tokenizer = Tokenizer(eos=1)
    metadata = align_generation_tokens(model(), tokenizer)
    assert metadata["eos_token_ids"] == [2] and metadata["source"] == "model.generation_config"
    assert metadata["tokenizer_eos_before"] == 1 and metadata["tokenizer_eos_mismatch"]
    assert tokenizer.eos_token_id == tokenizer.pad_token_id == 2
    assert not metadata["source_files_rewritten"]


def test_multiple_stops_fallbacks_and_explicit_padding_are_preserved():
    tokenizer = Tokenizer(eos=2, pad=0)
    metadata = align_generation_tokens(model(generation=[2, 3, 2]), tokenizer)
    assert metadata["eos_token_ids"] == [2, 3] and not metadata["tokenizer_eos_mismatch"]
    assert tokenizer.pad_token_id == 0
    fallback = align_generation_tokens(model(eos=3, generation=None), tokenizer)
    assert fallback["source"] == "model.config" and tokenizer.eos_token_id == 3
    last = align_generation_tokens(model(eos=None, generation=None), tokenizer)
    assert last["source"] == "tokenizer" and last["eos_token_ids"] == [3]
    align_generation_tokens(model(), tokenizer, reset_pad=True)
    assert tokenizer.pad_token_id == 2


@pytest.mark.parametrize("eos", [[], True, -1, 4, [2, None], "2"])
def test_malformed_or_out_of_vocabulary_model_stop_configuration_fails_closed(eos):
    with pytest.raises(ValueError, match="valid in-vocabulary EOS"):
        align_generation_tokens(model(generation=eos), Tokenizer())
