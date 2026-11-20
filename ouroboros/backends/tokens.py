"""Resolve declared model stop tokens without treating chat delimiters as answer text.

Some GGUF conversions give the tokenizer a start-of-turn token as EOS while the model's
config/generation_config correctly declares end-of-turn. Prefer the model's declared stops,
record the mismatch, and align the tokenizer IN MEMORY. Never rewrite a user's source files.
"""


def align_generation_tokens(model, tokenizer, *, reset_pad=False):
    before = tokenizer.eos_token_id
    ids, source = None, None
    for label, config in (
        ("model.generation_config", getattr(model, "generation_config", None)),
        ("model.config", model.config),
    ):
        value = getattr(config, "eos_token_id", None)
        if value is not None:
            ids, source = value, label
            break
    if ids is None:
        ids, source = before, "tokenizer"
    ids = ids if isinstance(ids, list) else [ids]
    vocabulary = getattr(model.config, "vocab_size", len(tokenizer))
    if not ids or any(type(value) is not int or not 0 <= value < vocabulary for value in ids):
        raise ValueError("Model/tokenizer must declare valid in-vocabulary EOS token ids")
    ids = list(dict.fromkeys(ids))
    mismatch = before not in ids
    if mismatch:
        token = tokenizer.convert_ids_to_tokens(ids[0])
        if not isinstance(token, str) or tokenizer.convert_tokens_to_ids(token) != ids[0]:
            raise ValueError("Model EOS does not round-trip through the local tokenizer")
        tokenizer.eos_token = token
    if reset_pad or tokenizer.pad_token_id is None:
        pad = getattr(getattr(model, "generation_config", None), "pad_token_id", None)
        if pad is None:
            pad = getattr(model.config, "pad_token_id", None)
        pad = ids[0] if pad is None else pad
        if type(pad) is not int or not 0 <= pad < vocabulary:
            raise ValueError("Model pad token id is outside its vocabulary")
        token = tokenizer.convert_ids_to_tokens(pad)
        if not isinstance(token, str) or tokenizer.convert_tokens_to_ids(token) != pad:
            raise ValueError("Model pad token does not round-trip through the local tokenizer")
        tokenizer.pad_token = token
    return {
        "eos_token_ids": ids,
        "source": source,
        "tokenizer_eos_before": before,
        "tokenizer_eos_mismatch": mismatch,
        "tokenizer_eos_resolved": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "source_files_rewritten": False,
    }
