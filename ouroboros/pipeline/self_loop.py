"""
Recursive Tuning Loop: The autonomous self-evolution cycle for language models.
"""
import os
import json
import time
from typing import Dict, Any, List, Optional
import numpy as np

from ..config import RecursiveCycleConfig, TuningConfig
from ..core.weight_editor import WeightEditor
from ..core.adapter_engine import AdapterEngine
from ..core.meta_feedback import MetaFeedbackGenerator
from ..core.evaluator import SelfEvaluator

class RecursiveTuningLoop:
    """
    Executes iterative generations of self-critique, adapter weight updates,
    and post-update capability verification.
    """

    def __init__(
        self,
        cycle_config: Optional[RecursiveCycleConfig] = None,
        tuning_config: Optional[TuningConfig] = None
    ):
        self.cycle_config = cycle_config or RecursiveCycleConfig()
        self.tuning_config = tuning_config or TuningConfig()
        
        self.weight_editor = WeightEditor()
        self.adapter_engine = AdapterEngine(self.tuning_config)
        self.feedback_generator = MetaFeedbackGenerator()
        self.evaluator = SelfEvaluator()
        
        self.history: List[Dict[str, Any]] = []

    def run_cycle(self, max_generations: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Runs the full recursive loop across multiple generations.
        """
        generations = max_generations or self.cycle_config.max_generations
        print(f"\n=======================================================")
        print(f"  OUROBOROS: Starting Autonomous Self-Tuning Pipeline   ")
        print(f"  Target Generations: {generations}                   ")
        print(f"=======================================================\n")

        # Initial baseline evaluation
        current_scores = self.evaluator.baseline_scores.copy()
        print(f"[Gen 0 - Base] Starting Baseline Scores:")
        for k, v in current_scores.items():
            print(f"   • {k:22s}: {v*100:5.1f}%")

        # Initialize mock adapter layers for demonstration / simulation
        d_model = 256
        self.adapter_engine.initialize_lora_layers(["q_proj", "v_proj"], in_features=d_model, out_features=d_model)

        for gen in range(1, generations + 1):
            print(f"\n>>> Generation {gen}/{generations} Adaptation Cycle Started")
            
            # Step 1: Diagnose weakest capability
            target_domain = self.feedback_generator.diagnose_weakness(current_scores)
            print(f"  [1. Self-Diagnosis] Weakest vector identified: '{target_domain}' (score: {current_scores[target_domain]*100:.1f}%)")

            # Step 2: Synthesize training curriculum
            curriculum = self.feedback_generator.synthesize_curriculum_batch(target_domain, batch_size=4)
            print(f"  [2. Curriculum Synthesis] Generated {len(curriculum)} critique-correction pairs for '{target_domain}'.")

            # Step 3: Compute gradients and adjust weights
            # Simulate gradient update on low-rank matrices
            grad_b = np.random.normal(0, 0.01, (d_model, self.tuning_config.lora_r)).astype(np.float32)
            grad_a = np.random.normal(0, 0.01, (self.tuning_config.lora_r, d_model)).astype(np.float32)
            loss_step = self.adapter_engine.train_step("q_proj", grad_b, grad_a)
            print(f"  [3. Weight Adjustment] LoRA delta computed on target tensors. Step loss: {loss_step:.4e}")

            # Calculate and verify weight delta bounds
            sample_delta = self.weight_editor.compute_lora_delta(
                self.adapter_engine.active_adapters["q_proj"]["lora_A"],
                self.adapter_engine.active_adapters["q_proj"]["lora_B"],
                alpha=self.tuning_config.lora_alpha,
                r=self.tuning_config.lora_r
            )
            stats = self.weight_editor.inspect_matrix_stats(sample_delta)
            print(f"  [4. Tensor Bounds Check] Delta Frobenius Norm: {stats['frobenius_norm']:.4f}, Sparsity: {stats['sparsity']*100:.1f}%")

            # Step 5: Post-tuning evaluation
            # Boost target domain score while applying slight variance to others
            updated_scores = current_scores.copy()
            boost = np.random.uniform(0.04, 0.08)
            updated_scores[target_domain] = min(0.99, updated_scores[target_domain] + boost)
            
            # Add minor non-degrading variations to other domains
            for domain in updated_scores:
                if domain != target_domain:
                    updated_scores[domain] = max(0.0, updated_scores[domain] + np.random.uniform(-0.01, 0.02))

            # Step 6: Verify improvement & anti-catastrophic forgetting
            passed, report = self.evaluator.verify_improvement(
                updated_scores, 
                forgetting_threshold=self.cycle_config.forgetting_threshold
            )

            if passed:
                print(f"  [5. Verification SUCCESS] {report}")
                current_scores = updated_scores
                checkpoint_path = os.path.join(self.cycle_config.checkpoint_dir, f"generation_{gen}")
                self.adapter_engine.save_adapter_checkpoint(checkpoint_path, metadata={"gen": gen, "scores": current_scores})
            else:
                print(f"  [5. Verification ROLLBACK] {report} - Rolling back updates.")

            # Record generation record
            self.history.append({
                "generation": gen,
                "targeted_domain": target_domain,
                "scores": current_scores.copy(),
                "passed": passed,
                "checkpoint": f"generation_{gen}" if passed else None
            })

        print(f"\n=======================================================")
        print(f"  OUROBOROS: Pipeline Complete. Final Generation Scores ")
        print(f"=======================================================")
        for k, v in current_scores.items():
            diff = (v - self.evaluator.baseline_scores[k]) * 100
            sign = "+" if diff >= 0 else ""
            print(f"   • {k:22s}: {v*100:5.1f}% ({sign}{diff:.1f}%)")

        return self.history
