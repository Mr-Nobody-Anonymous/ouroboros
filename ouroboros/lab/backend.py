"""Instrumented Transformers backend; one local model, fixed editor weights, no downloads."""

from contextlib import contextmanager

from ..backends.local_model import LocalCausalLM, format_prompt
from ..core.weight_trials import _seed
from .config import integer


@contextmanager
def frozen_eval(model):
    import torch

    modes = [(module, module.training) for module in model.modules()]
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            yield devices
    finally:
        for module, training in modes:
            module.training = training


class ResearchCausalLM(LocalCausalLM):
    name = "transformers-research-causal-lm"

    def render_research_prompt(self, instruction, sampling):
        if self.config.prompt_format == "chat":
            if not self.tokenizer.chat_template:
                raise ValueError("Chat format requires a chat template")
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=sampling.thinking,
            )
        return format_prompt(instruction)

    def generate_records(self, requests, *, sampling, max_new_tokens=None):
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList

        budget = self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
        integer("generation token budget", budget, 1, 32768)
        if not isinstance(requests, list) or not requests:
            raise ValueError("Nonempty generation requests required")
        results = []
        with frozen_eval(self.model) as devices:
            for request in requests:
                if set(request) != {"prompt", "seed"} or not isinstance(request["prompt"], str):
                    raise ValueError("Generation requires prompt and seed")
                integer("generation seed", request["seed"], 0, 2**63 - 1)
                inputs = self.tokenizer(
                    self.render_research_prompt(request["prompt"], sampling),
                    return_tensors="pt",
                    add_special_tokens=self.config.prompt_format != "chat",
                    return_token_type_ids=False,
                ).to(self.device)
                prompt_length = inputs.input_ids.shape[1]
                if prompt_length + budget > self.context_size:
                    raise ValueError("Generation prompt exceeds context; not silently truncated")
                _seed(torch, request["seed"], devices)
                kwargs = {
                    "do_sample": sampling.temperature > 0,
                    "repetition_penalty": sampling.repetition_penalty,
                }
                if sampling.temperature > 0:
                    kwargs.update(
                        temperature=sampling.temperature, top_p=sampling.top_p, top_k=sampling.top_k
                    )
                if sampling.presence_penalty:
                    penalty = sampling.presence_penalty

                    class PresencePenalty(LogitsProcessor):
                        def __call__(self, input_ids, scores):
                            generated = input_ids[0, prompt_length:].unique()
                            scores[:, generated] -= penalty
                            return scores

                    kwargs["logits_processor"] = LogitsProcessorList([PresencePenalty()])
                output = self.model.generate(
                    **inputs,
                    **kwargs,
                    max_new_tokens=budget,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                tokens = output[0, prompt_length:].tolist()
                eos = self.model.generation_config.eos_token_id
                eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
                if self.tokenizer.eos_token_id is not None:
                    eos_ids.add(self.tokenizer.eos_token_id)
                stopped = bool(tokens and tokens[-1] in eos_ids)
                # Keep thinking delimiters even when tokenizers mark them as special tokens.
                visible = tokens[:-1] if stopped else tokens
                text = self.tokenizer.decode(visible, skip_special_tokens=False).strip()
                results.append(
                    {
                        "text": text,
                        "new_tokens": len(tokens),
                        "cut": len(tokens) >= budget and not stopped,
                        "thinking_closed": not sampling.thinking or sampling.thinking_end in text,
                        "finish_reason": "eos" if stopped else "length",
                    }
                )
        return results
