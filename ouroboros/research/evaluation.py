"""Independent measured gates. Loss, model claims and practice cannot rescue a rejection."""

import math
from collections import Counter, defaultdict

from ..data import normalize
from ..lab.metrics import paired_summary, summarize
from .contracts import identity


def diagnosis(evaluation):
    """Observable symptoms only; hidden/weight patterns do not establish mechanisms."""
    rows, failures = evaluation["records"], []
    categories = Counter()
    for row in rows:
        if row["correct"]:
            continue
        category = (
            "token_budget_exhausted"
            if row["cut"]
            else ("empty_response" if not row["text"].strip() else "incorrect_answer")
        )
        categories[category] += 1
        failures.append({"id": row["id"], "domain": row["domain"], "category": category})
    evidence = [
        {
            "id": "practice:" + identity(evaluation["summary"]),
            "measurement": evaluation["summary"],
            "authority": "independent-executor",
        }
    ]
    if "nll" in evaluation:
        evidence.append(
            {
                "id": "practice_nll:" + identity(evaluation["nll"]),
                "measurement": evaluation["nll"],
                "authority": "independent-executor",
            }
        )
    return {
        "evidence": evidence,
        "symptoms": dict(categories),
        "failures": failures,
        "causal_claim": False,
        "uncertainty": "Answer mismatch cannot distinguish data, capacity, formatting and reasoning failures.",
    }


def validate_evaluation(value):
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("records"), list)
        or not value["records"]
    ):
        raise ValueError("Nonempty independent evaluation records required")
    seen = set()
    for row in value["records"]:
        if set(
            (
                "id",
                "sample",
                "seed",
                "domain",
                "correct",
                "cut",
                "thinking_closed",
                "text",
                "new_tokens",
            )
        ) - set(row):
            raise ValueError("Incomplete measured answer record")
        if (
            type(row["correct"]) is not bool
            or type(row["cut"]) is not bool
            or type(row["thinking_closed"]) is not bool
        ):
            raise ValueError("Evaluation booleans must be actual booleans")
        if type(row["new_tokens"]) is not int or row["new_tokens"] < 0:
            raise ValueError("Invalid measured token count")
        key = (row["id"], row["sample"])
        if key in seen:
            raise ValueError("Repeated problem/sample in measured evaluation")
        seen.add(key)
        if row["correct"] and (row["cut"] or not row["thinking_closed"]):
            raise ValueError("Cut/unclosed answer cannot be correct")
    if summarize(value["records"]) != value["summary"]:
        raise ValueError("Evaluation summary disagrees with executor records")


def gate(baseline, candidates, *, alpha, seed, require_improvement=True):
    """A fixed recipe must reproduce its improvement at every predeclared training seed.

    Each seed is tested separately, and the pooled test clusters all seeds by problem. The
    baseline's repeated predictions are NOT independent observations. No loss-based fallback.
    """
    validate_evaluation(baseline)
    if not candidates:
        raise ValueError("Candidate replications required")
    pooled_before, pooled_after, repeats = [], [], []
    for index, candidate in enumerate(candidates):
        validate_evaluation(candidate)
        paired = paired_summary(baseline, candidate, seed=seed)
        floors = {
            domain: candidate["summary"]["scores"][domain] >= score
            for domain, score in baseline["summary"]["scores"].items()
        }
        strict = candidate["summary"]["macro_accuracy"] > baseline["summary"]["macro_accuracy"]
        significant = paired["problem_sign_flip"]["p_value"] <= alpha
        passed = all(floors.values()) and (not require_improvement or (strict and significant))
        repeats.append(
            {
                "replicate": index,
                "passed": passed,
                "domain_floors": floors,
                "strict_accuracy_improvement": strict,
                "significant": significant,
                "paired": paired,
            }
        )
        for left, right in zip(baseline["records"], candidate["records"]):
            pooled_before.append({**left, "sample": index})
            pooled_after.append({**right, "sample": index})
    pooled = paired_summary(
        {"records": pooled_before, "summary": summarize(pooled_before)},
        {"records": pooled_after, "summary": summarize(pooled_after)},
        seed=seed,
    )
    passed = all(r["passed"] for r in repeats) and (
        not require_improvement or pooled["problem_sign_flip"]["p_value"] <= alpha
    )
    return {
        "passed": passed,
        "alpha": alpha,
        "strict_accuracy_only": True,
        "zero_domain_forgetting": True,
        "require_improvement": require_improvement,
        "replications": repeats,
        "problem_clustered": pooled,
        "reason": "all fixed gates passed"
        if passed
        else "strict gain/significance or zero-domain-forgetting failed; no practice/loss rescue",
    }


def choose_practice(baseline, candidates):
    """Pick a recipe, never a lucky seed, before any selection/held-out inference."""
    eligible = []
    for index, values in enumerate(candidates):
        for value in values:
            validate_evaluation(value)
        if values and all(
            v["summary"]["macro_accuracy"] > baseline["summary"]["macro_accuracy"]
            and all(
                v["summary"]["scores"][d] >= score
                for d, score in baseline["summary"]["scores"].items()
            )
            for v in values
        ):
            # Worst-seed first; deterministic recipe index breaks ties. No NLL tie-breaking.
            eligible.append((min(v["summary"]["macro_accuracy"] for v in values), -index, index))
    return max(eligible)[2] if eligible else None


def retention_gate(
    before_baseline, before_candidates, after_baseline, after_candidates, *, alpha, seed
):
    if len(before_candidates) != len(after_candidates) or len(after_baseline) != len(
        after_candidates
    ):
        raise ValueError("Matched fixed follow-up replications required")
    pre = gate(
        before_baseline, before_candidates, alpha=alpha, seed=seed, require_improvement=False
    )
    posts = [
        gate(base, [candidate], alpha=alpha, seed=seed, require_improvement=False)
        for base, candidate in zip(after_baseline, after_candidates)
    ]
    retained = [
        gate(before, [after], alpha=alpha, seed=seed, require_improvement=False)
        for before, after in zip(before_candidates, after_candidates)
    ]
    return {
        "passed": pre["passed"] and all(v["passed"] for v in posts + retained),
        "pre_intervention": pre,
        "matched_post_followup_controls": posts,
        "candidate_pre_post_followup": retained,
        "note": "Real fixed follow-up SFT. No-domain-regression floors, not a new useful-edit claim or a researcher-training experiment.",
    }


def audit_rewards(rows, reference):
    """Independent exact-answer verifier for training-only rejection/distillation data.

    A match verifies the final answer, not a rationale/proof. Never execute generated code.
    """
    if not isinstance(rows, list) or not 1 <= len(rows) <= 8:
        raise ValueError("Expected 1–8 bounded training rollouts")
    accepted = []
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("text"), str)
            or len(row["text"]) > 32000
        ):
            raise ValueError("Invalid rollout text")
        if type(row.get("cut")) is not bool or type(row.get("thinking_closed")) is not bool:
            raise ValueError("Explicit cut/thinking flags required")
        correct = (
            not row["cut"]
            and row["thinking_closed"]
            and normalize(row["text"]) == normalize(reference)
        )
        accepted.append(
            {
                "rollout": index,
                "correct": correct,
                "reward": float(correct),
                "text_sha256": identity(row["text"]),
            }
        )
    rewards = [r["reward"] for r in accepted]
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((v - mean) ** 2 for v in rewards) / len(rewards))
    return {
        "records": accepted,
        "group_relative_advantages": [(r - mean) / std if std else 0.0 for r in rewards],
        "uninformative_group": std == 0,
        "note": "DeepSeekMath-style group standardization diagnostic only. Not GRPO optimization; no reward for length/format/self-rating.",
    }


def aggregate_errors(evaluations):
    totals = defaultdict(int)
    for evaluation in evaluations:
        for key, value in diagnosis(evaluation)["symptoms"].items():
            totals[key] += value
    return dict(totals)
