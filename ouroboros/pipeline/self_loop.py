"""Measured, bounded candidate training with promotion, persistent provenance, and rollback.

This is an experiment controller, not a claim of guaranteed recursive intelligence gains.
The default backend is explicitly a tiny CPU classification fixture, not a language model.
"""

import itertools
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..backends.base import TrainingBackend
from ..backends.numpy_backend import NumpyLoRABackend
from ..config import RecursiveCycleConfig, TuningConfig
from ..core.anchor_memory import AnchorMemoryBank
from ..core.checkpoints import CheckpointStore, atomic_json
from ..core.evaluator import SelfEvaluator, evaluate_samples
from ..core.pareto_pruner import ParetoCheckpointPruner
from ..data import DatasetBundle, Sample, curate_proposals, demo_dataset


class RecursiveTuningLoop:
    def __init__(
        self,
        cycle_config=None,
        tuning_config=None,
        *,
        backend: TrainingBackend = None,
        dataset: DatasetBundle = None,
        proposal_provider=None,
        proposal_verifier=None,
    ):
        self.cycle_config = cycle_config or RecursiveCycleConfig()
        self.tuning_config = tuning_config or TuningConfig()
        self.dataset = dataset or demo_dataset()
        self.backend = backend or NumpyLoRABackend(
            [*self.dataset.train, *self.dataset.anchors],
            rank=self.tuning_config.lora_r,
            alpha=self.tuning_config.lora_alpha,
            seed=self.cycle_config.seed,
        )
        if proposal_provider is not None and proposal_verifier is None:
            raise ValueError("Self-generated curricula require an independent proposal_verifier")
        self.proposal_provider = proposal_provider
        self.proposal_verifier = proposal_verifier
        self.history = []
        self.report = None
        self.run_id = uuid.uuid4().hex
        self.run_directory = Path(self.cycle_config.output_dir) / self.run_id
        self.checkpoint_directory = Path(self.cycle_config.checkpoint_dir) / self.run_id
        self.evaluator = SelfEvaluator()
        self.anchor_memory = AnchorMemoryBank(
            self.cycle_config.anchor_ratio,
            [s.as_dict() for s in self.dataset.anchors],
            seed=self.cycle_config.seed,
        )
        if self.cycle_config.anchor_ratio and not self.dataset.anchors:
            raise ValueError("Positive anchor_ratio requires explicitly verified replay data")
        self.pareto_pruner = ParetoCheckpointPruner(self.cycle_config.max_checkpoints)
        self._has_run = False

    def _evaluate(self):
        result = evaluate_samples(self.backend.predict, self.dataset.validation)
        return result["scores"], result["predictions"]

    def _persist(self, manifest):
        self.report = {**manifest, "history": self.history}
        atomic_json(self.run_directory / "run.json", self.report)

    def run_cycle(self, max_generations=None):
        if self._has_run:
            raise RuntimeError(
                "A loop instance can run only once; create a new run for another experiment"
            )
        generations = (
            self.cycle_config.max_generations if max_generations is None else max_generations
        )
        if type(generations) is not int or generations < 1:
            raise ValueError("max_generations must be a positive integer")
        self._has_run = True
        self.run_directory.mkdir(parents=True, exist_ok=False)
        store = CheckpointStore(
            self.checkpoint_directory, self.run_id, self.cycle_config.min_free_disk_bytes
        )
        started = time.monotonic()
        deadline = started + self.cycle_config.max_seconds
        current_scores, base_predictions = self._evaluate()
        self.evaluator.baseline_scores = current_scores.copy()
        baseline = store.save(
            self.backend,
            0,
            scores=current_scores,
            metadata={"dataset_sha256": self.dataset.fingerprint()},
        )
        latest = baseline
        manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "backend": self.backend.describe(),
            "measurement": "normalized_exact_match",
            "dataset_sha256": self.dataset.fingerprint(),
            "config": {"cycle": asdict(self.cycle_config), "tuning": asdict(self.tuning_config)},
            "baseline_scores": current_scores.copy(),
            "baseline_predictions": base_predictions,
            "latest_checkpoint": latest,
            "checkpoint_directory": str(self.checkpoint_directory.resolve()),
            "status": "running",
            "final_test": {
                "status": "pending" if self.dataset.test else "not_configured",
                "used_for_promotion": False,
            },
        }
        self._persist(manifest)
        print(
            f"[Ouroboros] {self.backend.name}; run {self.run_id}; measured baseline {current_scores}"
        )
        for generation in range(1, generations + 1):
            if time.monotonic() >= deadline:
                manifest["status"] = "budget_exhausted"
                break
            # An experiment targets its weakest measured domain, but keeps all supplied training data.
            target = min(current_scores, key=lambda domain: (current_scores[domain], domain))
            snapshot = self.backend.snapshot()
            before = current_scores.copy()
            record = {
                "generation": generation,
                "targeted_domain": target,
                "previous_scores": before,
                "training_examples": 0,
                "anchor_examples": 0,
                "verified_proposals": 0,
                "stage": "curriculum",
                "checkpoint": None,
                "passed": False,
            }
            try:
                training = list(self.dataset.train)
                proposal_count = 0
                if self.proposal_provider is not None:
                    proposals = list(
                        itertools.islice(self.proposal_provider(target, tuple(training)), 32)
                    )
                    curated = curate_proposals(
                        proposals,
                        [*self.dataset.validation, *self.dataset.test],
                        self.proposal_verifier,
                    )
                    # Revalidate both generated and existing labels before any weight updates.
                    training = DatasetBundle(
                        [*training, *curated],
                        self.dataset.validation,
                        self.dataset.anchors,
                        self.dataset.test,
                    ).train
                    proposal_count = len(curated)
                blended = [
                    Sample.from_dict(row)
                    for row in self.anchor_memory.blend_curriculum([s.as_dict() for s in training])
                ]
                record.update(
                    training_examples=len(blended),
                    anchor_examples=sum(s.is_anchor for s in blended),
                    verified_proposals=proposal_count,
                    stage="training",
                )
                metrics = self.backend.train(
                    blended,
                    self.tuning_config,
                    max_steps=self.cycle_config.max_train_steps,
                    deadline=deadline,
                )
                if metrics.get("steps", 0) < 1:
                    raise RuntimeError("No optimization step completed within the budget")
                if metrics.get("loss") is None or not np.isfinite(metrics["loss"]):
                    raise FloatingPointError("Backend did not return a finite measured loss")
                record.update(stage="evaluation", training=metrics)
                candidate_scores, predictions = self._evaluate()
                passed, reason = self.evaluator.verify_improvement(
                    candidate_scores,
                    forgetting_threshold=self.cycle_config.forgetting_threshold,
                    previous_scores=before,
                    improvement_threshold=self.cycle_config.improvement_threshold,
                )
                record.update(
                    candidate_scores=candidate_scores,
                    candidate_predictions=predictions,
                    reason=reason,
                )
                if passed:
                    record["stage"] = "checkpoint_export"
                    checkpoint = store.save(
                        self.backend,
                        generation,
                        scores=candidate_scores,
                        parent=latest,
                        metadata={
                            "dataset_sha256": self.dataset.fingerprint(),
                            "training": metrics,
                        },
                    )
                    current_scores, latest = candidate_scores, checkpoint
                    record.update(passed=True, checkpoint=checkpoint)
                else:
                    self.backend.restore(snapshot)
                record["stage"] = "completed"
            except BaseException as exc:
                # Restore even on cancellation; system-exit/interrupt still propagate to the caller.
                self.backend.restore(snapshot)
                if not isinstance(exc, Exception):
                    manifest["status"] = "interrupted"
                    if self.dataset.test:
                        manifest["final_test"]["status"] = "skipped_after_interruption"
                    record.update(reason=type(exc).__name__, error=True, scores=before)
                    self.history.append(record)
                    self._persist(manifest)
                    raise
                record.update(reason=f"{type(exc).__name__}: {exc}", error=True)
            record["scores"] = current_scores.copy()
            self.history.append(record)
            manifest["latest_checkpoint"] = latest
            self._persist(manifest)
            # Only this run's managed checkpoints can be removed; baseline/current are immutable anchors.
            self.pareto_pruner.prune_checkpoints(
                self.checkpoint_directory,
                run_id=self.run_id,
                protected=(baseline, latest),
                apply=True,
            )
            print(
                f"[Generation {generation}] {'PROMOTED' if record['passed'] else 'ROLLED BACK'}: {record['reason']}"
            )
            if record.get("error"):
                manifest["status"] = "failed"
                break
        if manifest["status"] == "running":
            manifest["status"] = "completed"
        if self.dataset.test:
            if manifest["status"] in {"completed", "budget_exhausted"}:
                try:
                    # Exactly one post-selection evaluation; never a baseline, proposer input or gate.
                    manifest["final_test"] = {
                        "status": "completed",
                        "used_for_promotion": False,
                        **evaluate_samples(self.backend.predict, self.dataset.test),
                    }
                except BaseException as exc:
                    interrupted = not isinstance(exc, Exception)
                    manifest["status"] = "interrupted" if interrupted else "final_test_failed"
                    manifest["final_test"] = {
                        "status": "interrupted" if interrupted else "failed",
                        "used_for_promotion": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    if interrupted:
                        manifest.update(
                            final_scores=current_scores, elapsed_seconds=time.monotonic() - started
                        )
                        self._persist(manifest)
                        raise
            else:
                manifest["final_test"] = {
                    "status": "skipped_after_failure",
                    "used_for_promotion": False,
                }
        manifest.update(final_scores=current_scores, elapsed_seconds=time.monotonic() - started)
        self._persist(manifest)
        print(f"[Ouroboros] Report: {self.run_directory / 'run.json'}")
        return self.history
