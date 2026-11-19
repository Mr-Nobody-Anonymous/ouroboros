"""Opt-in real local causal-LM LoRA/QLoRA SFT, without remote code or model downloads."""

import copy
import json
import time
from pathlib import Path

import numpy as np

from ..config import ModelConfig, TuningConfig
from .local_model import LocalCausalLM, format_prompt


class TransformersLoRABackend(LocalCausalLM):
    name = "transformers-peft-lora"

    def __init__(
        self, model_config: ModelConfig, tuning_config: TuningConfig, seed=42, adapter_path=None
    ):
        expected_base_files = None
        if adapter_path is not None:
            adapter_path = Path(adapter_path)
            if not (adapter_path / "adapter_model.safetensors").is_file():
                raise ValueError("Local adapters must use safetensors, not pickle weights")
            metadata_path = adapter_path / "backend.json"
            if metadata_path.is_file():
                expected_base_files = json.loads(metadata_path.read_text(encoding="utf-8")).get(
                    "base_model_files"
                )
        super().__init__(
            model_config,
            fallback_context_size=tuning_config.max_seq_length,
            expected_base_files=expected_base_files,
        )
        try:
            from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("Install the training extra: pip install -e '.[train]'") from exc
        torch = self.torch
        self.tuning = tuning_config
        self.rng = np.random.default_rng(seed)
        model = self.model
        if model_config.use_qlora:
            model = prepare_model_for_kbit_training(model)
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if adapter_path:
                self.model = PeftModel.from_pretrained(
                    model, str(adapter_path), is_trainable=True, local_files_only=True
                )
            else:
                self.model = get_peft_model(
                    model,
                    LoraConfig(
                        task_type="CAUSAL_LM",
                        r=tuning_config.lora_r,
                        lora_alpha=tuning_config.lora_alpha,
                        lora_dropout=tuning_config.lora_dropout,
                        target_modules=tuning_config.target_modules,
                        bias="none",
                    ),
                )
            self._torch_rng = torch.get_rng_state().clone()
            self._cuda_rng = torch.cuda.get_rng_state(self.device).clone() if devices else None
        self.model.eval()

    def describe(self):
        return {**super().describe(), "tuning_config": dict(self.tuning.__dict__)}

    def snapshot(self):
        from peft import get_peft_model_state_dict

        return {
            "adapter": {
                k: v.detach().cpu().clone()
                for k, v in get_peft_model_state_dict(self.model).items()
            },
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "torch_rng": self._torch_rng.clone(),
            "cuda_rng": self._cuda_rng.clone() if self._cuda_rng is not None else None,
        }

    def restore(self, state):
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(self.model, state["adapter"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self._torch_rng = state["torch_rng"].clone()
        self._cuda_rng = state["cuda_rng"].clone() if state["cuda_rng"] is not None else None
        self.model.eval()

    def encode_sample(self, sample, max_length):
        prompt = self.tokenizer.encode(format_prompt(sample.instruction), add_special_tokens=True)
        completion = self.tokenizer.encode(sample.target, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            completion.append(self.tokenizer.eos_token_id)
        if not completion or len(completion) >= max_length:
            raise ValueError(
                "Completion does not fit the sequence budget; refusing to erase its labels"
            )
        prompt = prompt[-(max_length - len(completion)) :]
        if not prompt:
            raise ValueError("Training requires at least one prompt token")
        return prompt + completion, [-100] * len(prompt) + completion

    def _collate(self, rows):
        torch = self.torch
        length = max(len(ids) for ids, _ in rows)
        ids, labels, masks = [], [], []
        for sequence, target in rows:
            padding = length - len(sequence)
            ids.append(sequence + [self.tokenizer.pad_token_id] * padding)
            labels.append(target + [-100] * padding)
            masks.append([1] * len(sequence) + [0] * padding)
        return {
            key: torch.tensor(value, device=self.device)
            for key, value in {"input_ids": ids, "labels": labels, "attention_mask": masks}.items()
        }

    def train(self, samples, config, *, max_steps, deadline):
        torch = self.torch
        if not samples or max_steps < 1:
            raise ValueError("Training samples and a positive step budget are required")
        rows = [
            self.encode_sample(s, min(config.max_seq_length, self.context_size)) for s in samples
        ]
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
        losses, steps, exhausted = [], 0, False
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(self._torch_rng)
                if devices:
                    torch.cuda.set_rng_state(self._cuda_rng, self.device)
                self.model.train()
                for _ in range(config.num_train_epochs):
                    order = self.rng.permutation(len(rows))
                    batches = [
                        order[i : i + config.batch_size]
                        for i in range(0, len(rows), config.batch_size)
                    ]
                    for offset in range(0, len(batches), config.gradient_accumulation_steps):
                        if steps >= max_steps or time.monotonic() >= deadline:
                            exhausted = True
                            break
                        group = batches[offset : offset + config.gradient_accumulation_steps]
                        # Weight microbatches by completion-token count, including a partial last batch.
                        token_counts = [
                            sum(sum(label != -100 for label in rows[int(i)][1][1:]) for i in b)
                            for b in group
                        ]
                        total_tokens = sum(token_counts)
                        optimizer.zero_grad(set_to_none=True)
                        for batch, count in zip(group, token_counts):
                            if time.monotonic() >= deadline:
                                exhausted = True
                                break
                            output = self.model(**self._collate([rows[int(i)] for i in batch]))
                            loss = output.loss
                            if not torch.isfinite(loss):
                                raise FloatingPointError("Nonfinite training loss")
                            (loss * (count / total_tokens)).backward()
                            losses.append(float(loss.detach().cpu()))
                        if exhausted:
                            optimizer.zero_grad(set_to_none=True)
                            break
                        norm = torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                        if not torch.isfinite(norm):
                            raise FloatingPointError("Nonfinite gradient norm")
                        optimizer.step()
                        steps += 1
                    if exhausted:
                        break
                self._torch_rng = torch.get_rng_state().clone()
                if devices:
                    self._cuda_rng = torch.cuda.get_rng_state(self.device).clone()
        finally:
            optimizer.zero_grad(set_to_none=True)
            self.model.eval()
        return {
            "steps": steps,
            "loss": float(np.mean(losses)) if losses else None,
            "budget_exhausted": exhausted,
        }

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(directory), safe_serialization=True)
        self.tokenizer.save_pretrained(str(directory))
        (directory / "backend.json").write_text(
            json.dumps(self.describe(), indent=2), encoding="utf-8"
        )
        # Optimizer is deliberately reset each generation; this is an adapter export, not mid-step resume.
