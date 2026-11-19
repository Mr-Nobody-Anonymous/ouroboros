"""
Meta Feedback Generator: Autonomous error diagnosis and synthetic curriculum generation.
"""
from typing import Dict, Any, List, Optional
import random

class MetaFeedbackGenerator:
    """
    Simulates or orchestrates self-critique where the model analyzes its own outputs,
    flags flawed mathematical or logical reasoning steps, and synthesizes corrected pairs.
    """

    def __init__(self, curriculum_seed: int = 42):
        random.seed(curriculum_seed)

    def diagnose_weakness(self, evaluation_scores: Dict[str, float]) -> str:
        """Finds the lowest performing skill to target in the next self-tuning iteration."""
        if not evaluation_scores:
            return "general_reasoning"
        weakest = min(evaluation_scores.items(), key=lambda x: x[1])
        return weakest[0]

    def synthesize_curriculum_batch(self, target_domain: str, batch_size: int = 5) -> List[Dict[str, str]]:
        """
        Generates synthetic self-correction training data tailored to the identified weakness.
        """
        curriculum_templates = {
            "math_logic": [
                {
                    "instruction": "Solve step-by-step: A model has 70B parameters, quantized to 4 bits. What is the approximate VRAM required for weights in gigabytes?",
                    "flawed_output": "70B * 4 bits = 280GB. So 280GB of VRAM is required.",
                    "critique": "Confused bits with bytes. 4 bits is 0.5 bytes. 70B * 0.5 bytes = 35GB.",
                    "target": "Step 1: 4 bits = 0.5 bytes.\nStep 2: 70 billion parameters * 0.5 bytes/param = 35 billion bytes.\nStep 3: 35 billion bytes / (1024^3) ≈ 32.6 GiB (or 35 GB raw). Accounting for overhead, approximately 35-38 GB VRAM is needed."
                },
                {
                    "instruction": "Explain why rank-r matrix decomposition Delta_W = B @ A reduces parameter count.",
                    "flawed_output": "It just compresses the numbers into smaller numbers without changing matrix size.",
                    "critique": "Failed to explain matrix dimensions: d_in * d_out becomes r * (d_in + d_out).",
                    "target": "For a weight matrix of size d_in x d_out, storing full weights requires (d_in * d_out) values. Decomposing into B (d_out x r) and A (r x d_in) requires r * (d_in + d_out) values. When r << min(d_in, d_out), this saves orders of magnitude of memory and trainable parameters."
                }
            ],
            "coding": [
                {
                    "instruction": "Write an efficient function to compute rolling gradient clipping in PyTorch.",
                    "flawed_output": "for p in params: p.grad = max(min(p.grad, 1.0), -1.0)",
                    "critique": "Element-wise clipping alters the gradient direction. Vector norm clipping torch.nn.utils.clip_grad_norm_ preserves directional angle.",
                    "target": "torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0, norm_type=2.0)"
                }
            ]
        }
        
        pool = curriculum_templates.get(target_domain, curriculum_templates["math_logic"])
        samples = []
        for i in range(batch_size):
            item = random.choice(pool).copy()
            item["id"] = f"{target_domain}_sample_{i+1}"
            samples.append(item)
            
        return samples
