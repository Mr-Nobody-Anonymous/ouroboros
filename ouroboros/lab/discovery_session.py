"""Hypothesis-driven extension; legacy research fixtures and promotion rules stay intact."""

import copy as python_copy
import hashlib
import json
from dataclasses import asdict

from ..core.evaluator import SelfEvaluator
from ..self_edit import model_fingerprint
from .config import number
from .journal import (
    adaptive_summary,
    candidate_hash,
    compare_candidates,
    outcome,
    selection_diagnostics,
    verify_roundtrip,
)
from .metrics import paired_summary
from .sensitivity import inspect_gradients, measure_loss
from .session import BudgetExceeded, LabSession
from .state import editable_parameter
from .tools import TOOLS

DISCOVERY_TOOLS = {
    "hypothesize": {
        "required": [
            "problem",
            "targets",
            "reason",
            "ids",
            "metric",
            "expected_delta",
            "falsification",
        ],
        "purpose": "Commit a brief testable hypothesis before an edit, not private chain of thought.",
    },
    "inspect_gradients": {
        "required": ["name", "ids"],
        "optional": ["weights", "limit"],
        "purpose": "Opt-in practice autograd sensitivity. No optimizer/update; not gradient-free.",
    },
    "compare_checkpoints": {
        "optional": ["left", "right"],
        "purpose": "Compare original/edited/saved session overlays, not filesystem paths.",
    },
    "evaluate_candidate": {
        "required": ["ids"],
        "purpose": "Paired practice accuracy and completion NLL, not promotion.",
    },
    "rollback": {
        "optional": ["to"],
        "purpose": "Restore previous staged state or original; no resident editor mutation.",
    },
    "analyze_failure": {
        "required": ["experiment_id", "assessment", "evidence", "revision"],
        "purpose": "Record a brief effect classification/revision beside actual outcomes, not causal proof.",
    },
    "record_experiment": {
        "required": ["experiment_id", "summary"],
        "purpose": "Link a brief interpretation to immutable host measurements.",
    },
    "stop_and_explain": {
        "required": ["reason"],
        "purpose": "Withdraw with an evidence-limited explanation. An edit is never forced.",
    },
}


def bounded_text(name, value, maximum=600):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= maximum:
        raise ValueError(f"{name} must be nonempty text <= {maximum} characters")


class DiscoverySession(LabSession):
    terminal_actions = {"confirm", "withdraw", "stop_and_explain"}
    # Bundled sweeps / implicit random-control edits skip prospective per-trial audits.
    # The orchestrator emits explicit hypothesis/edit/evaluate/rollback commands instead.
    tools = {key: dict(value) for key, value in TOOLS.items() if key not in ("sweep", "control")}
    tools.update(DISCOVERY_TOOLS)
    for _name in ("edit", "restore"):
        tools[_name] = {**tools[_name], "required": [*tools[_name]["required"], "hypothesis_id"]}

    def __init__(self, *args, discovery, measurement_cache=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.discovery = discovery
        self.measurement_cache = measurement_cache
        self.measurement_requests = self.cache_hits = 0
        self.hypotheses, self.experiments, self.analyses, self.interpretations = [], [], [], []
        self.sensitivities, self.undo = [], []
        self.last_proposal = {}
        self.development_check = self.development_measurement = None
        self.stop_reason = None
        self.loss_evaluated_problems = self.gradient_evaluated_problems = 0
        self.resident_sha256 = model_fingerprint(self.copy.model)
        with self.copy.use({}):
            self.target_sha256 = model_fingerprint(self.copy.model)
        self._attempt = None

    def available_actions(self):
        if self.sealed:
            return sorted(self.terminal_actions)
        if self.discovery["workflow"] == "free":
            return list(self.tools)
        common = {"withdraw", "stop_and_explain", "weights", "inspect_gradients"}
        unused = any(
            not h["used"] and h["candidate_sha256"] == candidate_hash(self.copy.candidate)
            for h in self.hypotheses
        )
        if unused:
            allowed = common | {"edit"}
        elif not any(row is not self._attempt for row in self.experiments):
            allowed = common | {"practice", "hypothesize"}
        else:
            allowed = common | {
                "hypothesize",
                "evaluate_candidate",
                "rollback",
                "analyze_failure",
                "record_experiment",
                "compare_checkpoints",
                "submit",
            }
        if not self.discovery["gradient_access"]:
            allowed.discard("inspect_gradients")
        return sorted(allowed)

    def schema_context(self):
        return {
            "allowed_actions": self.available_actions(),
            **super().schema_context(),
            "unused_hypotheses": [
                h["id"]
                for h in self.hypotheses
                if not h["used"] and h["candidate_sha256"] == candidate_hash(self.copy.candidate)
            ],
            "experiments": [e["id"] for e in self.experiments],
            "gradient_access": self.discovery["gradient_access"],
            "max_edit_tensors": self.policy.max_edit_tensors,
            "operation_scope": self.discovery["operation_scope"],
        }

    def measure(self, samples, candidate, **kwargs):
        self.measurement_requests += 1
        key = hashlib.sha256(
            json.dumps(
                {
                    "target": self.target_sha256,
                    "candidate": candidate_hash(candidate),
                    "backend": self.backend.describe(),
                    "samples": [s.as_dict() for s in samples],
                    "sampling": asdict(self.sampling),
                    "grader": self.grader,
                    "answer_tokens": self.backend.config.max_new_tokens,
                    "seed": kwargs.get("seed", self.policy.seed),
                    "repeats": kwargs.get("repeats", self.policy.samples_per_problem),
                    "loss_limits": [self.policy.max_read_tokens, self.policy.max_logit_elements],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cache_key = (id(self.backend), key)
        if self.measurement_cache is not None and cache_key in self.measurement_cache:
            self.cache_hits += 1
            result = python_copy.deepcopy(self.measurement_cache[cache_key])
            result["cache_receipt"]["reused"] = True
            return result
        result = super().measure(samples, candidate, **kwargs)
        with self.copy.use(candidate):
            loss = measure_loss(self.backend, samples, self.policy, sampling=self.sampling)
        self.loss_evaluated_problems += len(samples)
        result["summary"]["loss"] = {key: value for key, value in loss.items() if key != "records"}
        result["loss_records"] = loss["records"]
        result["cache_receipt"] = {
            "key": key,
            "reused": False,
            "note": "Optional within-process exact source/state/records/config/seed cache; actual generations and cache hits are counted separately. Never reused across model/tokenizer instances.",
        }
        if self.measurement_cache is not None:
            self.measurement_cache[cache_key] = python_copy.deepcopy(result)
        return result

    def journal(self):
        return {
            "hypotheses": self.hypotheses,
            "experiments": self.experiments,
            "analyses": self.analyses,
            "sensitivities": self.sensitivities,
            "interpretations": self.interpretations,
            "development_check": self.development_check,
            "stop_reason": self.stop_reason,
            "adaptive_summary": adaptive_summary(self.hypotheses, self.experiments, self.analyses),
            "loss_evaluated_problems": self.loss_evaluated_problems,
            "measurement_requests": self.measurement_requests,
            "measurement_cache_hits": self.cache_hits,
            "gradient_evaluated_problems": self.gradient_evaluated_problems,
        }

    def editor_observation(self, observation):
        if self.discovery["feedback"] or not self.experiments or self.sealed:
            return observation
        # Raw archive/journal retain measurements, but this preregistered ablation must not
        # leak its automatic edited-practice feedback into a later editor prompt.
        return {
            key: value
            for key, value in observation.items()
            if key not in {"paired_practice", "outcome", "actual_outcome"}
        } | {
            "feedback": "Edited-practice outcomes withheld in this feedback-off arm; no learning claim."
        }

    def act(self, action):
        editing = isinstance(action, dict) and action.get("action") in (
            "edit",
            "restore",
            "rollback",
            "drop",
        )
        if not editing:
            return super().act(action)
        before = dict(self.copy.candidate)
        try:
            request = json.loads(json.dumps(action, allow_nan=False))
        except (ValueError, TypeError):
            request = {"invalid_non_json_action_repr": repr(action)[:4096]}
        attempt = {
            "id": f"e{len(self.experiments)}",
            "turn": self.turns + 1,
            "original_checkpoint_sha256": self.target_sha256,
            "resident_editor_sha256": self.resident_sha256,
            "hypothesis_id": action.get("hypothesis_id"),
            "operation": request,
            "status": "attempted",
            "validation_result": None,
            "practice_evaluation": None,
            "reserved_result": None,
            "restored_to_resident_base": None,
        }
        self.experiments.append(attempt)
        self._attempt = attempt
        try:
            result = super().act(action)
            attempt["status"] = "completed"
            return result
        except BaseException as exc:
            attempt.update(
                status="blocked_budget" if isinstance(exc, BudgetExceeded) else "failed_attempt",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            attempt["changes"] = compare_candidates(
                self.copy, before, self.copy.candidate, full=True
            )
            attempt["actual_nonempty_change"] = attempt["changes"]["changed_entries"] > 0
            attempt["roundtrip_replay"] = verify_roundtrip(
                self.copy, before, self.copy.candidate, attempt["changes"]
            )
            hypothesis = next(
                (h for h in self.hypotheses if h["id"] == attempt["hypothesis_id"]), None
            )
            attempt["selection_diagnostics"] = (
                selection_diagnostics(attempt["changes"], self.sensitivities, hypothesis)
                if hypothesis is not None
                else []
            )
            if attempt["validation_result"] is None:
                attempt["validation_result"] = {
                    "passed": False,
                    "kind": "command_or_precondition_rejected",
                }
            attempt["restored_to_resident_base"] = (
                model_fingerprint(self.copy.model) == self.resident_sha256
            )
            self._attempt = None

    def _variant(self, name):
        if name == "original":
            return {}
        if name == "edited":
            return self.copy.candidate
        if not isinstance(name, str) or name not in self.copy.saved:
            raise ValueError("Choose original, edited or an existing saved variant")
        return self.copy.saved[name]

    def _evaluate(self, samples, before, after):
        if len(samples) * self.policy.samples_per_problem * 2 > 512:
            raise ValueError("Paired request exceeds 512 answers")
        self._forward()
        left, right = self.measure(samples, before), self.measure(samples, after)
        paired = paired_summary(
            left, right, seed=self.policy.seed, draws=self.policy.permutation_draws
        )
        return {
            "paired": paired,
            "baseline_records": left["records"],
            "candidate_records": right["records"],
            "baseline_loss_records": left["loss_records"],
            "candidate_loss_records": right["loss_records"],
        }

    def _dispatch(self, kind, args):
        if kind not in self.available_actions():
            raise ValueError(
                "This action is outside the preregistered host-staged workflow at the current phase; generate your own hypothesis/edit or stop, not an inventory loop"
            )
        if (
            not self.discovery["feedback"]
            and self.experiments
            and kind
            in (
                "test",
                "evaluate_candidate",
                "contrast_practice",
                "control",
                "inspect_gradients",
                "read",
                "contrast",
            )
        ):
            if (
                kind not in ("test", "inspect_gradients", "read", "contrast", "contrast_practice")
                or args.get("weights", "edited") != "original"
            ):
                raise ValueError(
                    "Edited-practice queries are disabled by the fixed feedback-off arm; original reads remain available"
                )
        if kind == "hypothesize":
            for key in ("problem", "reason", "falsification"):
                bounded_text(key, args[key])
            samples = self._problems(args["ids"])
            targets = args["targets"]
            if (
                not isinstance(targets, list)
                or not 1 <= len(targets) <= self.policy.max_edit_tensors
                or any(not isinstance(n, str) for n in targets)
                or len(set(targets)) != len(targets)
            ):
                raise ValueError("Hypothesis needs unique bounded target parameter names")
            for name in targets:
                editable_parameter(self.copy.model, name, self.policy)
            if args["metric"] not in ("accuracy", "nll"):
                raise ValueError("Hypothesis metric is accuracy or nll")
            number("expected_delta", args["expected_delta"], -100, 1)
            if (args["metric"] == "accuracy" and args["expected_delta"] <= 0) or (
                args["metric"] == "nll" and args["expected_delta"] >= 0
            ):
                raise ValueError(
                    "Predict a positive accuracy delta or negative NLL delta, not zero"
                )
            row = {
                "id": f"h{len(self.hypotheses)}",
                **args,
                "candidate_sha256": candidate_hash(self.copy.candidate),
                "used": False,
                "confirmation": "Paired practice delta reaches expected effect; falsified at zero or opposite direction, otherwise inconclusive. Independent development/reserved accuracy gates are unchanged.",
            }
            self.hypotheses.append(row)
            return {
                "hypothesis_id": row["id"],
                "committed": True,
                "practice_problems": len(samples),
                "prediction": {"metric": row["metric"], "delta": row["expected_delta"]},
            }
        if kind == "inspect_gradients":
            if not self.discovery["gradient_access"]:
                raise ValueError(
                    "Gradient inspection disabled by preregistered profile; no implicit fallback"
                )
            samples = self._problems(args["ids"])
            weights = args.get("weights", "original")
            if weights not in ("original", "edited"):
                raise ValueError("Gradient weights must be original or edited")
            self._forward()
            with self.copy.use(self._variant(weights)):
                result = inspect_gradients(
                    self.backend,
                    self.copy,
                    samples,
                    args["name"],
                    limit=args.get("limit", 16),
                    sampling=self.sampling,
                )
            self.gradient_evaluated_problems += len(samples)
            row = {
                "weights": weights,
                "practice_ids": args["ids"],
                "candidate_sha256": candidate_hash(self._variant(weights)),
                **result,
            }
            self.sensitivities.append(row)
            return row
        if kind == "compare_checkpoints":
            return compare_candidates(
                self.copy,
                self._variant(args.get("left", "original")),
                self._variant(args.get("right", "edited")),
            )
        if kind in ("rollback", "drop"):
            if kind == "drop":
                result = super()._dispatch(kind, args)
            else:
                target = args.get("to", "previous")
                if target not in ("previous", "original"):
                    raise ValueError("Rollback to previous or original only")
                self._edit()
                if target == "previous" and not self.undo:
                    raise ValueError("No previous staged state is available")
                self.copy.candidate = self.undo.pop() if target == "previous" else {}
                result = {"rolled_back_to": target}
            self._attempt["validation_result"] = {
                "passed": True,
                "kind": "explicit_reversal_not_new_edit_hypothesis",
            }
            return {
                **result,
                "candidate_sha256": candidate_hash(self.copy.candidate),
                "resident_editor_unchanged": True,
            }
        if kind in ("edit", "restore"):
            identifier = args.pop("hypothesis_id")
            hypothesis = next((h for h in self.hypotheses if h["id"] == identifier), None)
            if hypothesis is None or hypothesis["used"]:
                raise ValueError(
                    "A fresh prospectively committed hypothesis_id is required per edit/restore"
                )
            if hypothesis["candidate_sha256"] != candidate_hash(self.copy.candidate):
                raise ValueError("Candidate changed since hypothesis; commit a new hypothesis")
            targets = (
                {op.get("name") for op in args.get("operations", []) if isinstance(op, dict)}
                if kind == "edit"
                else set(self.copy.saved.get(args["name"], {})) | set(self.copy.candidate)
            )
            if not targets or not targets <= set(hypothesis["targets"]):
                raise ValueError("Edit targets must be covered by the prospective hypothesis")
            if (
                self.discovery["operation_scope"] == "sparse"
                and kind == "edit"
                and any(op.get("op") not in ("set_entries", "add") for op in args["operations"])
            ):
                raise ValueError(
                    "This sparse-operator ablation permits set_entries/add only; protections unchanged"
                )
            if self.forward_calls >= self.policy.max_forward_calls:
                raise BudgetExceeded(
                    "No forward request remains for mandatory paired edit feedback"
                )
            before = dict(self.copy.candidate)
            hypothesis["used"] = True
            try:
                result = super()._dispatch(kind, args)
            except BaseException as exc:
                self._attempt["validation_result"] = {
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                raise
            self.undo.append(before)
            self.undo = self.undo[-4:]
            self._attempt["validation_result"] = {"passed": True, "details": result}
            if self.copy.candidate:
                self.last_proposal = dict(self.copy.candidate)
            evaluated = self._evaluate(
                self._problems(hypothesis["ids"]), before, self.copy.candidate
            )
            self._attempt["practice_evaluation"] = evaluated
            self._attempt["outcome"] = outcome(hypothesis, evaluated["paired"])
            return {
                **result,
                "experiment_id": self._attempt["id"],
                "hypothesis_id": identifier,
                "paired_practice": evaluated["paired"],
                "outcome": self._attempt["outcome"],
            }
        if kind == "evaluate_candidate":
            return self._evaluate(self._problems(args["ids"]), {}, self.copy.candidate)
        if kind in ("record_experiment", "analyze_failure"):
            experiment = next(
                (row for row in self.experiments if row["id"] == args["experiment_id"]), None
            )
            if experiment is None:
                raise ValueError("Unknown observed experiment_id")
            if kind == "record_experiment":
                bounded_text("summary", args["summary"])
                self.interpretations.append(dict(args))
            else:
                if args["assessment"] not in ("helpful", "harmful", "unchanged", "unmeasured"):
                    raise ValueError("Classify effect as helpful, harmful, unchanged or unmeasured")
                for key in ("evidence", "revision"):
                    bounded_text(key, args[key])
                self.analyses.append(dict(args))
            return {
                "recorded": True,
                "experiment_id": experiment["id"],
                "actual_outcome": experiment.get("outcome"),
                "model_statement_not_causal_proof": True,
            }
        if kind == "stop_and_explain":
            bounded_text("reason", args["reason"])
            self.stop_reason = args["reason"]
            return {**super()._dispatch("withdraw", {}), "explanation": self.stop_reason}
        if kind == "submit" and self.copy.candidate:
            # Seal BOTH independent gates BEFORE any development inference. No editing after
            # an unsuccessful development look either; otherwise it would become practice data.
            self.sealed = True
            left, right = (
                self.measure(self.discovery["development"], {}),
                self.measure(self.discovery["development"], self.copy.candidate),
            )
            self.development_measurement = {"baseline": left, "candidate": right}
            paired = paired_summary(
                left, right, seed=self.policy.seed, draws=self.policy.permutation_draws
            )
            passed, reason = SelfEvaluator(left["summary"]["scores"]).verify_improvement(
                right["summary"]["scores"],
                forgetting_threshold=self.policy.forgetting_threshold,
                improvement_threshold=self.policy.improvement_threshold,
            )
            if passed and paired["problem_sign_flip"]["p_value"] >= self.policy.alpha:
                passed, reason = (
                    False,
                    "Development problem-clustered significance threshold not met",
                )
            self.development_check = {"eligible": passed, "reason": reason, "paired": paired}
            if not passed:
                self.pending = {"eligible": False, "reason": reason, "development_failed": True}
                return {
                    "sealed": True,
                    "eligible_for_host_promotion": False,
                    "reason": reason,
                    "development_check": paired,
                    "reserved_check": "not_opened",
                    "next_actions": ["confirm", "withdraw", "stop_and_explain"],
                }
            result = super()._dispatch(kind, args)
            sha = candidate_hash(self.copy.candidate)
            for row in self.experiments:
                if row.get("changes", {}).get("right_candidate_sha256") == sha:
                    row["reserved_result"] = result.get("reserved_check")
            return {**result, "development_check": paired}
        return super()._dispatch(kind, args)
