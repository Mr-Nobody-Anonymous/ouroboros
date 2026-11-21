"""An actual trainable CPU LoRA classifier for testing the control loop, NOT an LLM.

A frozen zero base W plus alpha/r * B@A maps bag-of-token features to known labels.
Cross-entropy gradients are analytical; all reported scores use real predictions.
"""

import copy
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import TuningConfig
from ..data import Sample, tokens


class NumpyLoRABackend:
    name = "numpy-lora-classifier"

    def __init__(self, training_samples: Sequence[Sample], rank=4, alpha=4.0, seed=42):
        if not training_samples or rank < 1 or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Training samples, rank >= 1 and positive alpha are required")
        # Fit preprocessing on training data only; no validation labels or vocabulary are read.
        self.vocabulary = {
            token: i
            for i, token in enumerate(
                sorted(
                    {token for sample in training_samples for token in tokens(sample.instruction)}
                )
            )
        }
        self.labels = sorted({s.target for s in training_samples})
        self.rank, self.alpha = rank, alpha
        self.rng = np.random.default_rng(seed)
        self.a = self.rng.normal(0, 0.2, (rank, len(self.vocabulary) + 1))
        self.b = np.zeros((len(self.labels), rank))

    def _features(self, prompts):
        x = np.zeros((len(prompts), len(self.vocabulary) + 1))
        for i, prompt in enumerate(prompts):
            for token in tokens(prompt):
                if token in self.vocabulary:
                    x[i, self.vocabulary[token]] += 1
            norm = np.linalg.norm(x[i])
            if norm:
                x[i] /= norm
            x[i, -1] = 1  # bias feature
        return x

    def predict(self, prompts):
        logits = self._features(prompts) @ self.a.T @ self.b.T * (self.alpha / self.rank)
        return [self.labels[index] for index in logits.argmax(axis=1)]

    def snapshot(self):
        return {
            "a": self.a.copy(),
            "b": self.b.copy(),
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def restore(self, state):
        if state["a"].shape != self.a.shape or state["b"].shape != self.b.shape:
            raise ValueError("Snapshot shapes do not match the backend")
        self.a, self.b = state["a"].copy(), state["b"].copy()
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])

    def train(self, samples, config: TuningConfig, *, max_steps, deadline):
        if not samples or max_steps < 1:
            raise ValueError("Training samples and a positive step budget are required")
        if any(s.target not in self.labels for s in samples):
            raise ValueError("The demo classifier cannot add labels after initialization")
        x = self._features([s.instruction for s in samples])
        y = np.array([self.labels.index(s.target) for s in samples])
        losses, steps = [], 0
        scale = self.alpha / self.rank
        for _ in range(config.num_train_epochs):
            order = self.rng.permutation(len(samples))
            for start in range(0, len(samples), config.batch_size):
                if steps >= max_steps or time.monotonic() >= deadline:
                    return {
                        "steps": steps,
                        "loss": float(np.mean(losses)) if losses else None,
                        "budget_exhausted": True,
                    }
                ids = order[start : start + config.batch_size]
                xb, yb = x[ids], y[ids]
                h = xb @ self.a.T
                logits = h @ self.b.T * scale
                logits -= logits.max(axis=1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=1, keepdims=True)
                losses.append(
                    float(-np.log(np.maximum(probs[np.arange(len(ids)), yb], 1e-30)).mean())
                )
                grad = probs.copy()
                grad[np.arange(len(ids)), yb] -= 1
                grad /= len(ids)
                grad_b = scale * grad.T @ h
                grad_a = scale * (grad @ self.b).T @ xb
                norm = np.sqrt(np.sum(grad_a**2) + np.sum(grad_b**2))
                factor = min(1.0, config.max_grad_norm / max(norm, 1e-30))
                self.a -= config.learning_rate * grad_a * factor
                self.b -= config.learning_rate * grad_b * factor
                if not np.isfinite(self.a).all() or not np.isfinite(self.b).all():
                    raise FloatingPointError("Nonfinite adapter parameters")
                steps += 1
        return {"steps": steps, "loss": float(np.mean(losses)), "budget_exhausted": False}

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "adapter.npz", lora_A=self.a, lora_B=self.b)
        (directory / "backend.json").write_text(
            json.dumps(
                {
                    **self.describe(),
                    "vocabulary": self.vocabulary,
                    "labels": self.labels,
                    "rng_state": self.rng.bit_generator.state,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "backend.json").read_text(encoding="utf-8"))
        if metadata["backend"] != cls.name:
            raise ValueError("Not a NumPy LoRA checkpoint")
        model = cls.__new__(cls)
        model.rank, model.alpha = metadata["rank"], metadata["alpha"]
        model.vocabulary, model.labels = metadata["vocabulary"], metadata["labels"]
        model.rng = np.random.default_rng()
        model.rng.bit_generator.state = metadata["rng_state"]
        with np.load(directory / "adapter.npz", allow_pickle=False) as arrays:
            model.a, model.b = arrays["lora_A"].copy(), arrays["lora_B"].copy()
        expected = ((model.rank, len(model.vocabulary) + 1), (len(model.labels), model.rank))
        if (model.a.shape, model.b.shape) != expected or not all(
            np.isfinite(w).all() for w in (model.a, model.b)
        ):
            raise ValueError("Invalid checkpoint tensor shapes or values")
        return model

    def describe(self):
        return {
            "backend": self.name,
            "is_llm": False,
            "rank": self.rank,
            "alpha": self.alpha,
            "trainable_parameters": int(self.a.size + self.b.size),
        }
