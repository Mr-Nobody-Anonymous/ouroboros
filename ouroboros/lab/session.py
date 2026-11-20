"""One metered session over a target copy; reserved labels never enter editor observations."""

import json
import re
import time

from ..core.evaluator import SelfEvaluator
from .config import integer
from .instrumentation import (
    ReadTextTooLong,
    measure_direction,
    rank_writers,
    read_states,
    weight_statistics,
)
from .metrics import evaluate, paired_summary, problem_id, summarize
from .program import build_program, norm_matched_control, validate_candidate
from .tools import TOOLS


class BudgetExceeded(RuntimeError):
    pass


class LabSession:
    def __init__(
        self,
        backend,
        copy,
        dataset,
        *,
        sampling,
        grader,
        previous_scores=None,
        reserved_baseline=None,
    ):
        self.backend, self.copy, self.dataset = backend, copy, dataset
        self.policy, self.sampling, self.grader = copy.policy, sampling, grader
        self.previous_scores, self.reserved_baseline = previous_scores, reserved_baseline
        self.started = time.monotonic()
        self.turns = self.forward_calls = self.edit_actions = self.invalid_actions = 0
        self.generated_sequences = self.reserved_checks = 0
        self.notes, self.pending, self.decision = "", None, None
        self.sealed = False

    @property
    def done(self):
        return self.decision is not None

    def counters(self):
        return {
            "turns": self.turns,
            "forward_calls": self.forward_calls,
            "edit_actions": self.edit_actions,
            "invalid_actions": self.invalid_actions,
            "generated_sequences": self.generated_sequences,
            "reserved_checks": self.reserved_checks,
            "elapsed_seconds": time.monotonic() - self.started,
        }

    def _check(self):
        if self.done:
            raise ValueError("Session has ended")
        if (
            self.turns >= self.policy.max_turns
            or self.invalid_actions >= self.policy.max_invalid_actions
            or time.monotonic() - self.started >= self.policy.max_seconds
        ):
            raise BudgetExceeded("Turn/invalid-action/time budget exhausted")

    def _forward(self, count=1):
        if self.forward_calls + count > self.policy.max_forward_calls:
            raise BudgetExceeded("Target forward-request budget exhausted")
        self.forward_calls += count

    def _edit(self):
        if self.edit_actions >= self.policy.max_edit_actions:
            raise BudgetExceeded("Edit-action budget exhausted")
        self.edit_actions += 1

    def measure(self, samples, candidate, *, seed=None, repeats=None):
        repeats = self.policy.samples_per_problem if repeats is None else repeats
        seed = self.policy.seed if seed is None else seed
        rows = []
        # Independent final/validation splits can exceed the per-request practice batch size.
        for start in range(0, len(samples), self.policy.max_problems_per_call):
            batch = samples[start : start + self.policy.max_problems_per_call]
            with self.copy.use(candidate):
                result = evaluate(
                    self.backend,
                    batch,
                    sampling=self.sampling,
                    grader=self.grader,
                    samples_per_problem=repeats,
                    seed=seed,
                )
            rows.extend(result["records"])
            self.generated_sequences += len(result["records"])
        return {"summary": summarize(rows), "records": rows}

    def _problems(self, ids):
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= self.policy.max_problems_per_call
            or any(type(i) is not int or not 0 <= i < len(self.dataset.train) for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(
                "ids must be unique zero-based PRACTICE indices within the per-call budget"
            )
        return [self.dataset.train[i] for i in ids]

    def _pairs(self, samples, variants, *, charge=1):
        if len(samples) * self.policy.samples_per_problem * (1 + len(variants)) > 512:
            raise ValueError("Paired request exceeds 512 total answers")
        self._forward(charge)
        baseline = self.measure(samples, {})
        results = {}
        for label, candidate in variants.items():
            measured = self.measure(samples, candidate)
            results[label] = {
                "paired": paired_summary(
                    baseline, measured, seed=self.policy.seed, draws=self.policy.permutation_draws
                ),
                "records": measured["records"],
            }
        return {"baseline_records": baseline["records"], "variants": results}

    def schema_context(self):
        """Only public target metadata / practice indices / existing session ids, never values or damage specs."""
        inventory = self.copy.inventory(limit=100_000)
        modules = [name for name, _ in self.copy.model.named_modules() if name]
        return {
            "parameters": [
                row["name"]
                for row in inventory["tensors"]
                if not row["protected"] and row["within_working_copy_budget"]
            ],
            "tensor_count": inventory["total_tensors"],
            "modules": modules,
            "module_count": len(modules),
            "practice_count": len(self.dataset.train),
            "max_problems_per_call": self.policy.max_problems_per_call,
            "max_read_tokens": self.policy.max_read_tokens,
            "directions": sorted(self.copy.directions),
            "saved": sorted(self.copy.saved),
            "edited": sorted(self.copy.candidate),
        }

    def reject_reply(self, message):
        """Charge one failed generation as one turn/invalid action, preserving its parse error."""
        self._check()
        self.turns += 1
        self.invalid_actions += 1
        raise ValueError("Invalid editor reply: " + message)

    def act(self, action):
        """Execute one closed-language action. Model/agent strings never become Python or paths."""
        self._check()
        self.turns += 1
        try:
            if not isinstance(action, dict) or len(json.dumps(action, allow_nan=False)) > 64 * 1024:
                raise ValueError("Expected an action object <=64 KiB")
            kind = action.get("action")
            if not isinstance(kind, str) or kind not in TOOLS:
                raise ValueError("Unknown research action; use the tool catalogue, not Python")
            spec = TOOLS[kind]
            required = {"action", *spec.get("required", [])}
            allowed = required | set(spec.get("optional", [])) | {"rationale"}
            if not required <= set(action) or set(action) - allowed:
                raise ValueError(
                    f"Unexpected/missing fields for {kind}: missing={sorted(required - set(action))}, "
                    f"unexpected={sorted(set(action) - allowed)}"
                )
            if "rationale" in action and (
                not isinstance(action["rationale"], str) or len(action["rationale"]) > 2000
            ):
                raise ValueError("rationale must be a string <=2000 characters")
            if self.sealed and kind not in ("confirm", "withdraw"):
                raise ValueError(
                    "Reserved check sealed the edit; only confirm or withdraw is permitted"
                )
            result = self._dispatch(
                kind,
                {key: value for key, value in action.items() if key not in ("action", "rationale")},
            )
            return {**result, "counters": self.counters()}
        except BudgetExceeded:
            raise
        except Exception:
            self.invalid_actions += 1
            raise

    def _dispatch(self, kind, args):
        if kind in ("inventory", "modules", "practice"):
            offset, limit = args.get("offset", 0), args.get("limit", 20)
            integer("offset", offset, 0, 100_000)
            integer(
                "limit", limit, 1, 100 if kind != "practice" else self.policy.max_problems_per_call
            )
            if kind == "inventory":
                return self.copy.inventory(offset=offset, limit=limit)
            if kind == "practice":
                return {
                    "total_practice": len(self.dataset.train),
                    "problems": [
                        {
                            "id": index,
                            "instruction": sample.instruction,
                            "reference": sample.target,
                            "domain": sample.domain,
                        }
                        for index, sample in enumerate(
                            self.dataset.train[offset : offset + limit], offset
                        )
                    ],
                }
            prefix = args.get("prefix", "")
            if not isinstance(prefix, str) or len(prefix) > 200:
                raise ValueError("Module prefix must be a string <=200 characters")
            rows = [
                {"name": name, "type": type(module).__name__}
                for name, module in self.copy.model.named_modules()
                if name and name.startswith(prefix)
            ]
            return {"modules": rows[offset : offset + limit], "total": len(rows)}
        if kind == "weights":
            return weight_statistics(self.copy, **args)
        if kind in ("read", "contrast"):
            weights = args.pop("weights", "edited")
            if weights not in ("original", "edited"):
                raise ValueError("Read weights must be original or edited, never editor")
            self._forward()
            with self.copy.use({} if weights == "original" else self.copy.candidate):
                if kind == "read":
                    result = read_states(self.backend, policy=self.policy, **args)
                else:
                    result = measure_direction(self.copy, self.backend, **args)
            return {"weights": weights, **result}
        if kind == "contrast_practice":
            samples = self._problems(args["ids"])
            weights = args.get("weights", "original")
            if weights not in ("original", "edited"):
                raise ValueError("Practice contrast weights must be original or edited")
            if self.policy.max_read_texts < 2:
                raise ValueError("Practice contrast needs max_read_texts >= 2")
            candidate = {} if weights == "original" else self.copy.candidate
            self._forward()
            measured = self.measure(samples, candidate)
            successes = [
                row for row in measured["records"] if row["correct"] and row["text"].strip()
            ]
            failures = [
                row for row in measured["records"] if not row["correct"] and row["text"].strip()
            ]
            result = {
                "weights": weights,
                "source": "actual_graded_practice_generations",
                "practice_summary": measured["summary"],
                "successful_answers": len(successes),
                "failing_answers": len(failures),
                "practice_records": measured["records"],
            }
            if not successes or not failures:
                return {
                    **result,
                    "available": False,
                    "reason": "Both measured correct and incorrect nonempty practice outputs are required. No synthetic texts/reference answers were substituted.",
                }
            limit = self.policy.max_read_texts // 2
            failures, successes = failures[:limit], successes[:limit]
            lookup = {problem_id(sample): sample for sample in samples}

            def actual_texts(rows):
                return [lookup[row["id"]].instruction + "\n" + row["text"] for row in rows]

            self._forward()
            try:
                with self.copy.use(candidate):
                    direction = measure_direction(
                        self.copy,
                        self.backend,
                        module=args["module"],
                        failing=actual_texts(failures),
                        successful=actual_texts(successes),
                        last=args.get("last", min(16, self.policy.max_read_tokens)),
                    )
            except ReadTextTooLong as exc:
                # Keep all observed grades/completions useful to the editor, but create no
                # partial or reference-substituted direction. This attempted read stays metered.
                return {
                    **result,
                    "available": False,
                    "reason": str(exc),
                    "read_performed": False,
                    "read_limit_tokens": exc.limit,
                    "overlong_text_index": exc.text_index,
                    "overlong_text_tokens": exc.tokens,
                }
            return {
                **result,
                **direction,
                "available": True,
                "source_records": {
                    label: [
                        {key: row[key] for key in ("id", "sample", "seed", "correct", "cut")}
                        for row in rows
                    ]
                    for label, rows in (("failing", failures), ("successful", successes))
                },
                "interpretation": "Difference between real graded completion cohorts (prompt + newline + completion), potentially confounded by question content. Not a proven causal direction; no clean editor or held-out answers used.",
            }
        if kind == "writers":
            return rank_writers(self.copy, **args)
        if kind == "edit":
            replace = args.get("replace", False)
            if type(replace) is not bool:
                raise ValueError("replace must be boolean")
            self._edit()
            candidate, stats = build_program(self.copy, args["operations"], replace=replace)
            self.copy.candidate = candidate
            return {"staged": True, **stats}
        if kind == "drop":
            self._edit()
            names = args.get("names", list(self.copy.candidate))
            if not isinstance(names, list) or any(
                not isinstance(name, str) or name not in self.copy.candidate for name in names
            ):
                raise ValueError("Drop names must be currently edited tensors")
            candidate = {
                name: value for name, value in self.copy.candidate.items() if name not in names
            }
            self.copy.candidate, stats = validate_candidate(self.copy, candidate)
            return {"dropped": names, **stats}
        if kind in ("save", "restore"):
            name = args["name"]
            if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,32}", name) is None:
                raise ValueError("Saved variant name must be 1-32 letters/digits/_/-")
            if kind == "save":
                if name not in self.copy.saved and len(self.copy.saved) >= 4:
                    raise ValueError("At most four saved candidates are allowed")
                self.copy.saved[name] = {
                    key: value.clone() for key, value in self.copy.candidate.items()
                }
            else:
                self._edit()
                if name not in self.copy.saved:
                    raise ValueError("Unknown saved variant")
                self.copy.candidate, _ = validate_candidate(self.copy, self.copy.saved[name])
            return {
                "saved_variants": sorted(self.copy.saved),
                "current_tensors": sorted(self.copy.candidate),
            }
        if kind == "test":
            samples = self._problems(args["ids"])
            weights, saved = args.get("weights", "both"), args.get("saved", [])
            if weights not in ("original", "edited", "both"):
                raise ValueError("Test weights must be original, edited or both")
            if (
                not isinstance(saved, list)
                or len(saved) > 4
                or len(set(saved)) != len(saved)
                or any(not isinstance(name, str) or name not in self.copy.saved for name in saved)
            ):
                raise ValueError("Choose at most four distinct saved variants")
            if saved and weights != "both":
                raise ValueError("Saved side-by-side variants require weights=both")
            if weights == "both":
                variants = {
                    "edited": self.copy.candidate,
                    **{name: self.copy.saved[name] for name in saved},
                }
                return self._pairs(samples, variants)
            self._forward()
            return self.measure(samples, {} if weights == "original" else self.copy.candidate)
        if kind == "control":
            samples, control_kind = self._problems(args["ids"]), args["kind"]
            if control_kind == "no_op":
                control, metadata = {}, {"kind": "no_op"}
            elif control_kind == "layer_isolation":
                names = args.get("names")
                if (
                    not isinstance(names, list)
                    or not names
                    or any(
                        not isinstance(name, str) or name not in self.copy.candidate
                        for name in names
                    )
                ):
                    raise ValueError("Layer isolation needs edited tensor names")
                control = {name: self.copy.candidate[name] for name in names}
                control, metadata = validate_candidate(self.copy, control)
                metadata["kind"] = control_kind
            else:
                control, metadata = norm_matched_control(
                    self.copy, kind=control_kind, seed=args.get("seed", self.policy.seed + 1)
                )
            return {"control": metadata, **self._pairs(samples, {"control": control})}
        if kind == "sweep":
            samples, variants = self._problems(args["ids"]), args["variants"]
            if not isinstance(variants, list) or not 1 <= len(variants) <= 4:
                raise ValueError("Sweep needs 1-4 independently specified variants")
            candidates, metadata = {}, {}
            for variant in variants:
                if (
                    not isinstance(variant, dict)
                    or set(variant) != {"label", "operations"}
                    or not isinstance(variant["label"], str)
                    or not 1 <= len(variant["label"]) <= 32
                    or variant["label"] in candidates
                ):
                    raise ValueError("Unique bounded sweep label and operations required")
                candidates[variant["label"]], metadata[variant["label"]] = build_program(
                    self.copy, variant["operations"], replace=True
                )
            self._edit()
            return {
                "programs": metadata,
                **self._pairs(samples, candidates, charge=len(candidates)),
            }
        if kind == "notes":
            if not isinstance(args["text"], str) or len(args["text"]) > 4000:
                raise ValueError("Notes must be a string <=4000 characters")
            self.notes = args["text"]
            return {"notes_saved": True}
        if kind == "submit":
            self.sealed = True
            if not self.copy.candidate:
                self.decision = "withdrawn"
                return {"decision": "withdrawn", "reason": "No edit was staged"}
            self.reserved_checks += 1
            if self.reserved_baseline is None:
                self.reserved_baseline = self.measure(self.dataset.validation, {})
            measured = self.measure(self.dataset.validation, self.copy.candidate)
            paired = paired_summary(
                self.reserved_baseline,
                measured,
                seed=self.policy.seed,
                draws=self.policy.permutation_draws,
            )
            baseline_scores = self.reserved_baseline["summary"]["scores"]
            passed, reason = SelfEvaluator(baseline_scores).verify_improvement(
                measured["summary"]["scores"],
                previous_scores=self.previous_scores or baseline_scores,
                forgetting_threshold=self.policy.forgetting_threshold,
                improvement_threshold=self.policy.improvement_threshold,
            )
            if (
                passed
                and self.policy.require_significance
                and paired["problem_sign_flip"]["p_value"] >= self.policy.alpha
            ):
                passed, reason = False, "Reserved problem-clustered significance threshold not met"
            self.pending = {
                "eligible": passed,
                "reason": reason,
                "paired": paired,
                "measurement": measured,
            }
            # No reserved ids, questions, labels, raw answers or per-problem outcomes in observations.
            return {
                "sealed": True,
                "eligible_for_host_promotion": passed,
                "reason": reason,
                "reserved_check": paired,
                "next_actions": ["confirm", "withdraw"],
            }
        if kind == "confirm":
            if self.pending is None:
                raise ValueError("Submit for the one reserved check before confirming")
            self.decision = "accepted" if self.pending["eligible"] else "rejected"
            return {"decision": self.decision, "reason": self.pending["reason"]}
        if kind == "withdraw":
            self.copy.candidate = {}
            self.decision = "withdrawn"
            return {"decision": "withdrawn"}
        raise AssertionError("Unhandled research action")
