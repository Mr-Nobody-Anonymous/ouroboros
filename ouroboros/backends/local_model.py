"""Shared offline safetensors loading and greedy inference; no PEFT or training required."""

import inspect
from pathlib import Path

from ..config import ModelConfig
from ..core.checkpoints import file_sha256


def format_prompt(instruction):
    """Explicit text template shared by local inference and completion-masked SFT."""
    return f"Instruction:\n{instruction}\nResponse:\n"


class LocalCausalLM:
    """An explicitly provisioned local model, never a Hub download or remote-code loader."""

    name = "transformers-local-causal-lm"

    def __init__(
        self, model_config: ModelConfig, *, fallback_context_size=512, expected_base_files=None
    ):
        path = Path(model_config.model_name_or_path or "")
        if not model_config.model_name_or_path or not path.is_dir():
            raise ValueError("A pre-downloaded local model directory is required; no Hub downloads")
        patterns = (
            "*.safetensors",
            "*.safetensors.index.json",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
            "vocab.txt",
            "added_tokens.json",
            "chat_template.jinja",
        )
        base_files = sorted(
            {file for pattern in patterns for file in path.glob(pattern) if file.is_file()}
        )
        if not any(file.suffix == ".safetensors" for file in base_files):
            raise ValueError("Local base model must contain safetensors weights")
        self.base_model_files = {file.name: file_sha256(file) for file in base_files}
        if expected_base_files is not None and expected_base_files != self.base_model_files:
            raise ValueError(
                "Base model/tokenizer files differ from the exported adapter provenance"
            )
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Install LLM dependencies: pip install -e '.[llm]' (or the train extra)"
            ) from exc
        self.torch, self.config = torch, model_config
        if model_config.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.device = torch.device(model_config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer requires a pad or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": True,
            "torch_dtype": getattr(torch, model_config.torch_dtype),
        }
        if model_config.use_qlora:
            kwargs.update(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=getattr(torch, model_config.torch_dtype),
                ),
                device_map={"": str(self.device)},
            )
        self.model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
        if not model_config.use_qlora:
            self.model.to(self.device)
        self.model.requires_grad_(False)
        self.context_size = getattr(
            self.model.config,
            "max_position_embeddings",
            getattr(self.model.config, "n_positions", fallback_context_size),
        )
        if model_config.max_new_tokens >= self.context_size:
            raise ValueError("max_new_tokens must leave room for a prompt in the model context")
        self.model.eval()

    def describe(self):
        return {
            "backend": self.name,
            "is_llm": True,
            "base_model": str(Path(self.config.model_name_or_path).resolve()),
            "base_model_files": dict(self.base_model_files),
            "quantized": self.config.use_qlora,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "model_config": dict(self.config.__dict__),
        }

    def render_prompt(self, instruction):
        if self.config.prompt_format == "chat":
            if not self.tokenizer.chat_template:
                raise ValueError("chat prompt_format requires a tokenizer chat template")
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return format_prompt(instruction)

    def prompt_inputs(self, instruction):
        inputs = self.tokenizer(
            self.render_prompt(instruction),
            return_tensors="pt",
            add_special_tokens=self.config.prompt_format != "chat",
            return_token_type_ids=False,
        ).to(self.device)
        return inputs

    def choose(self, instruction, labels):
        """Constrained next-token selection by THIS model, not an external proposer or judge.

        Labels must be distinct single-token strings. This intentionally restricts the action
        space; it is not free-form weight-program synthesis or evidence that the edit is correct.
        """
        torch = self.torch
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("Nonempty unique choice labels are required")
        encodings = [self.tokenizer.encode(label, add_special_tokens=False) for label in labels]
        if any(len(ids) != 1 for ids in encodings) or len({ids[0] for ids in encodings}) != len(
            labels
        ):
            raise ValueError("Choice labels must map to unique single tokens")
        inputs = self.prompt_inputs(instruction)
        if inputs.input_ids.shape[1] >= self.context_size:
            raise ValueError("Proposal prompt exceeds the context budget")
        self.model.eval()
        last_only = (
            {"logits_to_keep": 1}
            if "logits_to_keep" in inspect.signature(self.model.forward).parameters
            else {}
        )
        with torch.inference_mode():
            logits = self.model(**inputs, **last_only).logits[0, -1].float()
            ids = torch.tensor([encoding[0] for encoding in encodings], device=logits.device)
            selected = logits[ids]
            if not torch.isfinite(selected).all():
                raise FloatingPointError("Nonfinite proposal logits")
            probabilities = torch.softmax(selected, dim=0).cpu().tolist()
        choice = labels[max(range(len(labels)), key=lambda i: probabilities[i])]
        return {
            "choice": choice,
            "method": "constrained-next-token-choice",
            "probabilities_within_menu": dict(zip(labels, probabilities)),
            "proposer_is_target_model": True,
        }

    def predict(self, prompts):
        torch = self.torch
        answers = []
        self.model.eval()
        # Single-prompt generation avoids padding-side and batch-size dependent decoding.
        with torch.inference_mode():
            for prompt in prompts:
                inputs = self.prompt_inputs(prompt)
                if inputs.input_ids.shape[1] + self.config.max_new_tokens > self.context_size:
                    raise ValueError(
                        "Evaluation prompt exceeds the context budget; it was not truncated"
                    )
                outputs = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.config.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                continuation = outputs[0, inputs.input_ids.shape[1] :]
                answers.append(
                    self.tokenizer.decode(continuation, skip_special_tokens=True).strip()
                )
        return answers
