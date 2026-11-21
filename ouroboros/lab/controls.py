"""Explicit HOST policies, never substituted for a failed autonomous editor.

They emit the same bounded hypothesis/edit/evaluate commands as model editors. Gradient
control uses practice labels only; random control matches the host trial's tensor, support
and actual delta norm, with independently seeded direction. Neither is a paper-parity run.
"""

import numpy as np


class NoEditControl:
    def __call__(self, session, turn, observation):
        return {
            "action": "stop_and_explain",
            "reason": "Preregistered no-edit baseline control. No proposed parameter intervention.",
        }


class HostControl:
    def __init__(self, target, *, random_seed=None, references=None):
        self.target = target
        self.random_seed, self.references = random_seed, references
        self.attempt, self.phase, self.operation = (
            0,
            "gradient" if random_seed is None else "hypothesis",
            None,
        )
        self.metric = None

    def __call__(self, session, turn, observation):
        ids = list(range(len(session.dataset.train)))
        if self.phase == "gradient":
            self.phase = "hypothesis"
            return {
                "action": "inspect_gradients",
                "name": self.target,
                "ids": ids,
                "weights": "edited",
                "limit": 256,
            }
        if self.phase == "hypothesis":
            original = session.copy.original(self.target).double().numpy().reshape(-1)
            current = session.copy.current(self.target).double().numpy().reshape(-1)
            if self.random_seed is None:
                sensitivity = session.sensitivities[-1]
                indices = np.array([row["index"] for row in sensitivity["entries"]], dtype=int)
                direction = -np.array([row["gradient"] for row in sensitivity["entries"]])
                dose = (0.01, 0.03)[self.attempt] * float(np.linalg.norm(original))
                reason = "HOST positive control, gradient-assisted: descend the measured practice completion-NLL gradient at the fixed final-block norm tensor, with a predeclared small relative dose. Does not predict decoded-answer gain."
            else:
                if not self.references or self.attempt >= len(self.references):
                    return {
                        "action": "stop_and_explain",
                        "reason": "Matched random control unavailable: no corresponding valid host-directed trial. No unmatched fallback or autonomous success was substituted.",
                    }
                reference = self.references[self.attempt]
                indices = np.asarray(reference["indices"], dtype=int)
                direction = np.random.default_rng(self.random_seed + self.attempt).normal(
                    size=len(indices)
                )
                dose = reference["delta_norm"]
                reason = "HOST random-direction null control: same tensor, support and actual delta norm as the host-directed practice trial, independent seeded direction. Test the improvement prediction rather than assuming randomness helps. Support is gradient-informed, direction is not."
            norm = float(np.linalg.norm(direction))
            if not norm:
                return {
                    "action": "stop_and_explain",
                    "reason": "Zero practice sensitivity/control direction. The predeclared mechanism probe cannot form a nonzero direction; no invented edit.",
                }
            values = current[indices] + dose * direction / norm
            self.operation = {
                "op": "set_entries",
                "name": self.target,
                "indices": indices.tolist(),
                "values": values.tolist(),
            }
            self.phase = "edit"
            return {
                "action": "hypothesize",
                "problem": "Practice completion/reference mismatch; probe local loss sensitivity, not a semantic circuit claim.",
                "targets": [self.target],
                "reason": reason,
                "ids": ids,
                "metric": "nll",
                "expected_delta": -0.0001,
                "falsification": "Falsified if actual paired practice NLL does not decrease; confirmed only if decrease reaches 0.0001 nats. Accuracy/development/reserved significance gates remain unchanged.",
            }
        if self.phase == "edit":
            self.phase = "review"
            return {
                "action": "edit",
                "hypothesis_id": session.hypotheses[-1]["id"],
                "operations": [self.operation],
            }
        if self.phase == "review":
            self.phase = "record"
            self.metric = session.experiments[-1].get("outcome")
            return {
                "action": "analyze_failure",
                "experiment_id": session.experiments[-1]["id"],
                "assessment": self.metric["effect"] if self.metric else "unmeasured",
                "evidence": "HOST reads the actual paired practice result archived beside this statement; no hidden split or model-authorship inference.",
                "revision": "Predeclared two-trial host policy: rollback harmful trials; recompute the next practice gradient at the current state. Random-direction trials always rollback between doses for a matched null. Never change acceptance gates.",
            }
        if self.phase == "record":
            self.phase = "compare"
            return {
                "action": "record_experiment",
                "experiment_id": session.experiments[-1]["id"],
                "summary": "HOST-authored mechanism observation; a loss decrease is not decoded-answer accuracy, autonomous discovery or learned future editing ability.",
            }
        if self.phase == "compare":
            self.phase = (
                "rollback"
                if (self.metric and self.metric["effect"] == "harmful")
                or (self.random_seed is not None and self.attempt == 0)
                else "next"
            )
            return {"action": "compare_checkpoints", "left": "original", "right": "edited"}
        if self.phase == "rollback":
            self.phase = "next"
            return {"action": "rollback", "to": "previous"}
        if self.phase == "next":
            self.attempt += 1
            self.phase = (
                ("gradient" if self.random_seed is None else "hypothesis")
                if self.attempt < 2
                else "evaluate"
            )
            return self(session, turn, observation)
        if self.phase == "evaluate":
            self.phase = "submit"
            return {"action": "evaluate_candidate", "ids": ids}
        if self.phase == "submit":
            self.phase = "confirm"
            return {"action": "submit"}
        return {"action": "confirm"}


def random_references(journal, target):
    rows = []
    for experiment in journal["experiments"]:
        if experiment["operation"].get("action") != "edit" or not experiment.get(
            "validation_result", {}
        ).get("passed"):
            continue
        tensors = [
            row
            for row in experiment["changes"]["tensors"]
            if row["name"] == target and row["changed_entries"]
        ]
        if tensors:
            rows.append(
                {
                    "source_experiment": experiment["id"],
                    "indices": tensors[0]["indices"],
                    "delta_norm": tensors[0]["delta_norm"],
                    "source_checkpoint_sha256": experiment["original_checkpoint_sha256"],
                }
            )
    return rows
