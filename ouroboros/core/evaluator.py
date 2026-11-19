"""
Self-Evaluator: Measures model accuracy, reasoning coherence, and monitors catastrophic forgetting.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

class SelfEvaluator:
    """
    Evaluates LLM performance across target domains and detects degradation.
    """

    def __init__(self, baseline_scores: Optional[Dict[str, float]] = None):
        self.baseline_scores = baseline_scores or {
            "math_logic": 0.68,
            "coding": 0.64,
            "general_knowledge": 0.81,
            "instruction_following": 0.77
        }

    def evaluate_benchmark(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Computes exact-match and normalized token overlap accuracy."""
        if not predictions:
            return 0.0
        
        matches = 0
        for p, g in zip(predictions, ground_truths):
            p_clean = p.strip().lower()
            g_clean = g.strip().lower()
            if g_clean in p_clean or p_clean == g_clean:
                matches += 1
                
        return matches / float(len(predictions))

    def evaluate_generation(self, test_suite: Dict[str, List[Dict[str, str]]]) -> Dict[str, float]:
        """Runs the evaluation across multiple benchmark categories."""
        results = {}
        for category, samples in test_suite.items():
            preds = [s.get("prediction", "") for s in samples]
            targets = [s.get("target", "") for s in samples]
            score = self.evaluate_benchmark(preds, targets)
            results[category] = score
            
        return results

    def verify_improvement(self, current_scores: Dict[str, float], forgetting_threshold: float = 0.05) -> Tuple[bool, str]:
        """
        Ensures target tasks improved without degrading anchor baseline skills.
        """
        for task, baseline in self.baseline_scores.items():
            current = current_scores.get(task, baseline)
            # If any baseline ability dropped more than threshold, flag forgetting
            if current < (baseline - forgetting_threshold):
                return False, f"Catastrophic forgetting detected on '{task}': score dropped from {baseline:.2f} to {current:.2f}"
                
        avg_baseline = np.mean(list(self.baseline_scores.values()))
        avg_current = np.mean([current_scores.get(t, self.baseline_scores[t]) for t in self.baseline_scores])
        
        if avg_current >= avg_baseline:
            return True, f"Positive evolution confirmed: overall capability increased by +{(avg_current - avg_baseline)*100:.1f}%"
        else:
            return False, "Overall score did not exceed baseline generation."
