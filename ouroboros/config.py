"""
Configuration parameters for recursive self-tuning cycles.
"""
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    model_name_or_path: str = "meta-llama/Llama-3-8B"
    device: str = "auto"
    torch_dtype: str = "float16"
    use_qlora: bool = True
    bits: int = 4

@dataclass
class TuningConfig:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024

@dataclass
class RecursiveCycleConfig:
    max_generations: int = 5
    curriculum_domain: str = "multi-step-reasoning"
    forgetting_threshold: float = 0.05
    improvement_threshold: float = 0.02
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
