"""Shared offline safetensors loading and greedy inference; no PEFT or training required."""

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
            raise RuntimeError("Install the training extra: pip install -e '.[train]'") from exc
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
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "model_config": dict(self.config.__dict__),
        }

    def predict(self, prompts):
        torch = self.torch
        answers = []
        self.model.eval()
        # Single-prompt generation avoids padding-side and batch-size dependent decoding.
        with torch.inference_mode():
            for prompt in prompts:
                inputs = self.tokenizer(format_prompt(prompt), return_tensors="pt").to(self.device)
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
