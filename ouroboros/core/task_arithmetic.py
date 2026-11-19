"""Shape-checked task vectors, DARE, TIES sign election, and SLERP tensor utilities."""

import math

import numpy as np


def compatible(*weights):
    if not weights or not all(isinstance(w, np.ndarray) and w.size for w in weights):
        raise ValueError("Expected nonempty NumPy tensors")
    if any(w.shape != weights[0].shape for w in weights):
        raise ValueError("Tensor shapes must match exactly")
    if any(not np.issubdtype(w.dtype, np.floating) or not np.isfinite(w).all() for w in weights):
        raise ValueError("Tensors must be finite and floating-point")


class TaskArithmeticEngine:
    @staticmethod
    def compute_task_vector(fine_tuned_weights, base_weights):
        compatible(fine_tuned_weights, base_weights)
        return fine_tuned_weights - base_weights

    @staticmethod
    def apply_task_vector(base_weights, task_vector, scaling=1.0):
        compatible(base_weights, task_vector)
        if not math.isfinite(scaling):
            raise ValueError("Scaling must be finite")
        return base_weights + scaling * task_vector

    @staticmethod
    def unlearn_behavior(base_weights, negative_task_vector, beta=0.5):
        """Subtract a vector; this does not certify knowledge removal or robust unlearning."""
        return TaskArithmeticEngine.apply_task_vector(base_weights, negative_task_vector, -beta)

    @staticmethod
    def dare_merge(base_weights, task_vectors, drop_rate=0.9, scale=1.0, *, rng=None):
        if not task_vectors or not math.isfinite(drop_rate) or not 0 <= drop_rate < 1:
            raise ValueError("DARE needs task vectors and drop_rate in [0, 1)")
        compatible(base_weights, *task_vectors)
        if not math.isfinite(scale):
            raise ValueError("Scale must be finite")
        rng = rng if rng is not None else np.random.default_rng()
        keep = 1 - drop_rate
        deltas = [np.where(rng.random(tv.shape) < keep, tv / keep, 0) for tv in task_vectors]
        return base_weights + scale * np.mean(deltas, axis=0)

    @staticmethod
    def ties_merge(base_weights, task_vectors, density=0.2, scale=1.0):
        """Trim each delta, elect signs by total magnitude, average agreeing nonzero entries.

        This is a local reference implementation, not a call into mergekit.
        """
        if not task_vectors or not math.isfinite(density) or not 0 <= density <= 1:
            raise ValueError("TIES needs task vectors and density in [0, 1]")
        if not math.isfinite(scale):
            raise ValueError("Scale must be finite")
        compatible(base_weights, *task_vectors)
        trimmed = np.zeros(
            (len(task_vectors), base_weights.size), dtype=np.result_type(*task_vectors)
        )
        k = math.floor(base_weights.size * density)
        for row, vector in zip(trimmed, task_vectors):
            flat = vector.reshape(-1)
            indices = np.argsort(-np.abs(flat), kind="stable")[:k]
            row[indices] = flat[indices]
        elected = np.sign(trimmed.sum(axis=0))
        agrees = (np.sign(trimmed) == elected) & (trimmed != 0)
        count = agrees.sum(axis=0)
        merged = np.divide(
            np.where(agrees, trimmed, 0).sum(axis=0),
            count,
            out=np.zeros(base_weights.size),
            where=count > 0,
        )
        return base_weights + scale * merged.reshape(base_weights.shape)

    @staticmethod
    def slerp_interpolate(v0, v1, t, eps=1e-8):
        compatible(v0, v1)
        if not math.isfinite(t) or not 0 <= t <= 1:
            raise ValueError("Interpolation t must be in [0, 1]")
        if t == 0:
            return v0.copy()
        if t == 1:
            return v1.copy()
        norm0, norm1 = np.linalg.norm(v0), np.linalg.norm(v1)
        linear = (1 - t) * v0 + t * v1
        if min(norm0, norm1) < eps:
            return linear
        dot = np.clip(np.dot(v0.reshape(-1) / norm0, v1.reshape(-1) / norm1), -1.0, 1.0)
        angle = np.arccos(dot)
        sine = np.sin(angle)
        if abs(sine) < eps:  # parallel OR antipodal; the great-circle path is not unique
            return linear
        return (np.sin((1 - t) * angle) * v0 + np.sin(t * angle) * v1) / sine
