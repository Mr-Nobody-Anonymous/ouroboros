"""Validated configuration. Importing configuration never loads a model or accesses the network."""

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional


def positive(name, value):
    if type(value) not in (int, float) or value <= 0 or value > 1e12 or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number in (0, 1e12]")


def probability(name, value, *, allow_one=True):
    if (
        type(value) not in (int, float)
        or value < 0
        or (value > 1 if allow_one else value >= 1)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be in [0, {'1]' if allow_one else '1)'}")


@dataclass
class ModelConfig:
    model_name_or_path: Optional[str] = None
    device: str = "cpu"
    torch_dtype: str = "float32"
    use_qlora: bool = False
    bits: int = 4
    max_new_tokens: int = 64

    def __post_init__(self):
        if self.model_name_or_path is not None and (
            not isinstance(self.model_name_or_path, str) or not self.model_name_or_path.strip()
        ):
            raise ValueError("model_name_or_path must be a nonempty local path string or None")
        if self.torch_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("torch_dtype must be float32, float16, or bfloat16")
        if not isinstance(self.device, str) or not re.fullmatch(
            r"cpu|cuda(?::[0-9]+)?", self.device
        ):
            raise ValueError("device must be cpu or cuda[:index]")
        if type(self.use_qlora) is not bool or type(self.bits) is not int or self.bits != 4:
            raise ValueError("use_qlora must be a boolean; only bits=4 is supported")
        if self.use_qlora and not self.device.startswith("cuda"):
            raise ValueError("QLoRA requires a CUDA device and 4-bit quantization")
        positive("max_new_tokens", self.max_new_tokens)
        if type(self.max_new_tokens) is not int:
            raise ValueError("max_new_tokens must be an integer")


@dataclass
class TuningConfig:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 512
    max_grad_norm: float = 1.0

    def __post_init__(self):
        for name in (
            "lora_r",
            "lora_alpha",
            "num_train_epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "max_seq_length",
        ):
            value = getattr(self, name)
            positive(name, value)
            if not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        positive("learning_rate", self.learning_rate)
        positive("max_grad_norm", self.max_grad_norm)
        probability("lora_dropout", self.lora_dropout, allow_one=False)
        if (
            self.max_seq_length < 2
            or not isinstance(self.target_modules, list)
            or not self.target_modules
            or any(not isinstance(name, str) or not name for name in self.target_modules)
        ):
            raise ValueError("max_seq_length >= 2 and nonempty target_modules are required")


@dataclass
class RecursiveCycleConfig:
    max_generations: int = 3
    curriculum_domain: str = "math_logic"
    forgetting_threshold: float = 0.05
    improvement_threshold: float = 0.0
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
    seed: int = 42
    anchor_ratio: float = 0.25
    max_checkpoints: int = 2
    max_train_steps: int = 200
    max_seconds: float = 300.0
    min_free_disk_bytes: int = 16 * 1024 * 1024

    def __post_init__(self):
        for name in ("checkpoint_dir", "output_dir", "curriculum_domain"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("max_generations", "max_checkpoints", "max_train_steps"):
            value = getattr(self, name)
            positive(name, value)
            if not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        probability("anchor_ratio", self.anchor_ratio, allow_one=False)
        probability("forgetting_threshold", self.forgetting_threshold)
        probability("improvement_threshold", self.improvement_threshold)
        positive("max_seconds", self.max_seconds)
        if type(self.min_free_disk_bytes) is not int or self.min_free_disk_bytes < 0:
            raise ValueError("min_free_disk_bytes must be nonnegative")
