"""Evidence levels and exact attempted-edit deltas; no scores are inferred from syntax."""

import hashlib
import json

import numpy as np

from ..core.weight_trials import _tensor_hash


def candidate_hash(candidate):
    rows = {name: _tensor_hash(value) for name, value in sorted(candidate.items())}
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def compare_candidates(copy, left, right, *, full=False):
    """Actual stored-dtype changes between overlays on the SAME starting target."""
    rows, changed, squared = [], 0, 0.0
    for name in sorted(set(left) | set(right)):
        before, after = copy.current(name, left), copy.current(name, right)
        a, b = before.double().numpy().reshape(-1), after.double().numpy().reshape(-1)
        indices = np.flatnonzero(a != b)
        delta = b[indices] - a[indices]
        norm = float(np.linalg.norm(delta))
        row = {
            "name": name,
            "shape": list(before.shape),
            "dtype": str(before.dtype),
            "original_parameter_sha256": _tensor_hash(copy.original(name)),
            "before_parameter_sha256": _tensor_hash(before),
            "after_parameter_sha256": _tensor_hash(after),
            "changed_entries": len(indices),
            "delta_norm": norm,
        }
        selected = indices if full else indices[:16]
        row.update(
            indices=selected.tolist(),
            before_values=a[selected].tolist(),
            after_values=b[selected].tolist(),
            values_complete=bool(full or len(indices) <= 16),
        )
        rows.append(row)
        changed += len(indices)
        squared += norm**2
    return {
        "changed_entries": changed,
        "delta_norm": squared**0.5,
        "tensors": rows,
        "left_candidate_sha256": candidate_hash(left),
        "right_candidate_sha256": candidate_hash(right),
    }


def verify_roundtrip(copy, left, right, changes):
    """Replay exact audited values and reverse on detached tensors, never resident weights."""
    import torch

    for row in changes["tensors"]:
        if not row["values_complete"]:
            raise ValueError("Cannot replay an incomplete numerical ledger")
        before, after = copy.current(row["name"], left), copy.current(row["name"], right)
        if (
            _tensor_hash(before) != row["before_parameter_sha256"]
            or _tensor_hash(after) != row["after_parameter_sha256"]
        ):
            raise ValueError("Ledger tensor hash does not match source overlays")
        replay = before.clone()
        indices = row["indices"]
        replay.reshape(-1)[indices] = torch.tensor(row["after_values"], dtype=replay.dtype)
        if _tensor_hash(replay) != _tensor_hash(after):
            raise ValueError("Numerical edit replay failed")
        replay.reshape(-1)[indices] = torch.tensor(row["before_values"], dtype=replay.dtype)
        if _tensor_hash(replay) != _tensor_hash(before):
            raise ValueError("Numerical edit reversal failed")
    return {
        "passed": True,
        "scope": "complete stored-dtype tensor hashes, detached forward/reverse replay; no resident write",
    }


def outcome(hypothesis, paired):
    delta = paired["accuracy_delta"]["mean"]
    if hypothesis["metric"] == "nll":
        delta = (
            paired["candidate"]["loss"]["problem_mean_nll"]
            - paired["baseline"]["loss"]["problem_mean_nll"]
        )
    predicted = hypothesis["expected_delta"]
    signed = delta if hypothesis["metric"] == "accuracy" else -delta
    return {
        "metric": hypothesis["metric"],
        "actual_delta": delta,
        "expected_delta": predicted,
        "prediction_squared_error": (delta - predicted) ** 2,
        "prediction_supported": signed >= abs(predicted),
        "prediction_falsified": signed <= 0,
        "effect": "helpful" if signed > 0 else "harmful" if signed < 0 else "unchanged",
        "practice_only": True,
    }


def selection_diagnostics(changes, sensitivities, hypothesis):
    rows = []
    for tensor in changes["tensors"]:
        matched = [
            s
            for s in sensitivities
            if s["name"] == tensor["name"]
            and s["candidate_sha256"] == changes["left_candidate_sha256"]
            and sorted(s["practice_ids"]) == sorted(hypothesis["ids"])
        ]
        if not matched:
            rows.append({"name": tensor["name"], "matched_practice_sensitivity": False})
            continue
        gradients = {entry["index"]: entry["gradient"] for entry in matched[-1]["entries"]}
        dot, count = 0.0, 0
        for index, a, b in zip(tensor["indices"], tensor["before_values"], tensor["after_values"]):
            if index in gradients:
                dot += gradients[index] * (b - a)
                count += 1
        rows.append(
            {
                "name": tensor["name"],
                "matched_practice_sensitivity": True,
                "inspected_changed_entries": count,
                "changed_entries": tensor["changed_entries"],
                "partial_first_order_nll_delta": dot,
                "interpretation": "Local first-order prediction on inspected entries only; uninspected entries and nonlinear effects are not accounted for. Not a semantic/relevance or causal-understanding certificate.",
            }
        )
    return rows


def adaptive_summary(hypotheses, experiments, analyses):
    edits = [row for row in experiments if row["operation"].get("action") in ("edit", "restore")]
    observed = [row for row in edits if row.get("outcome")]
    effects = [row["outcome"]["actual_delta"] for row in observed]
    predictions = [row["outcome"]["prediction_squared_error"] for row in observed]
    by_id = {row["id"]: row for row in experiments}
    checked = [
        row
        for row in analyses
        if row["experiment_id"] in by_id and by_id[row["experiment_id"]].get("outcome")
    ]
    return {
        "hypotheses": len(hypotheses),
        "attempted_edits": len(edits),
        "explicit_reversal_actions": len(experiments) - len(edits),
        "valid_nonempty_edits": sum(
            row.get("changes", {}).get("changed_entries", 0) > 0
            and row.get("validation_result", {}).get("passed", False)
            for row in edits
        ),
        "measured_attempts": len(observed),
        "actual_deltas_in_order": effects,
        "metrics_in_order": [row["outcome"]["metric"] for row in observed],
        "effects_in_order": [row["outcome"]["effect"] for row in observed],
        "practice_helpful_attempts": sum(row["outcome"]["effect"] == "helpful" for row in observed),
        "practice_harmful_attempts": sum(row["outcome"]["effect"] == "harmful" for row in observed),
        "correctly_identified_harmful_attempts": len(
            {
                row["experiment_id"]
                for row in checked
                if row["assessment"] == "harmful"
                and by_id[row["experiment_id"]]["outcome"]["effect"] == "harmful"
            }
        ),
        "repeated_numerical_programs": len(edits)
        - len(
            {
                json.dumps(
                    row["operation"].get("operations", {"restore": row["operation"].get("name")}),
                    sort_keys=True,
                )
                for row in edits
            }
        ),
        "prediction_squared_errors_in_order": predictions,
        "harm_classifications": len(checked),
        "correct_harm_classifications": sum(
            row["assessment"] == by_id[row["experiment_id"]]["outcome"]["effect"] for row in checked
        ),
        "in_context_adaptation_demonstrated": None,
        "improved_future_editing_ability_demonstrated": False,
        "interpretation": "Ordered observations/calibration only. A trend or remembered feedback is not learned weight-level meta-improvement or a causal learning claim; use matched feedback-off arms and unseen editing tasks.",
    }


SUCCESS_LEVELS = {
    "command_validity": "Parsed command satisfies host interface; not an edit.",
    "edit_validity": "Nonempty actual stored-dtype parameter delta passes unchanged protection/count/norm/byte checks.",
    "edit_usefulness": "Strict predefined development accuracy/macro gain with original/incumbent per-domain retention, then a sealed independent reserved confirmation (configured significance unchanged). Loss alone never passes this gate.",
    "generalization": "Winner frozen before once-only final evaluation; report strict gain, uncertainty and domains, without selecting again from final outcomes.",
    "retention": "After selection, test a predefined subsequent intervention on a separate retention split; preserve both original and selected-model domain floors. Unmeasured is not passed.",
    "meta_improvement": "Requires a separate independently evaluated editing-policy checkpoint and unseen editing tasks; successful tool use or within-session feedback is insufficient.",
}
