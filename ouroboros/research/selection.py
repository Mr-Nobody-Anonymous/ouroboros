"""Predeclared offline comparison of experiment-selection policies.

The controller ranks proposed recipes with a transparent posterior/cost heuristic. Does that
heuristic actually choose better experiments than a null policy? This module answers only that
narrow, offline question, and only from **counterfactually logged** candidates:

* every episode is one recorded experiment whose *entire* candidate set was measured,
* a policy selects one candidate using only outcomes from earlier episodes (no leakage),
* the score is the measured practice improvement of the selected candidate,
* the design (policies, ordering, metric, alpha, decision rule, minimum episodes) is frozen
  before looking at any score and written once.

It is not online learning, not a trained researcher, and not a claim that the repository
improves its own future experiments. A null result is reported as a null result.
"""

from pathlib import Path

from ..lab.metrics import sign_flip_test
from .contracts import identity
from .provenance import write_once

POLICIES = ("proposal_order", "posterior_optimistic", "random_fixed_seed")


def design(
    *, alpha=0.05, min_episodes=20, seed=17, primary="posterior_optimistic", null="proposal_order"
):
    if primary == null or primary not in POLICIES or null not in POLICIES:
        raise ValueError("Primary and null policies must be distinct known policies")
    if not 0 < alpha < 1 or min_episodes < 6:
        raise ValueError("Predeclare a sane alpha and at least six episodes")
    return {
        "schema_version": 1,
        "kind": "predeclared-selection-policy-comparison",
        "policies": list(POLICIES),
        "primary": primary,
        "null": null,
        "episode_order": "recorded experiment creation order (ascending)",
        "metric": "measured practice macro-accuracy improvement of the selected candidate",
        "information_set": "only outcomes from strictly earlier episodes",
        "pairing": "the same episode sequence is scored by every policy",
        "decision_rule": "one-sided paired problem sign-flip test over episodes",
        "alpha": alpha,
        "min_episodes": min_episodes,
        "seed": seed,
        "frozen_before_analysis": True,
        "online_learning": False,
        "researcher_trained": False,
        "hidden_data_used": False,
    }


def _select(policy, candidates, history, *, seed, episode_index):
    """Return the chosen candidate index using `history` only (random handled by the caller)."""
    if policy == "proposal_order":
        return 0
    # posterior_optimistic: mean verified success + one standard deviation, cost ignored here
    # because the offline log has no comparable per-family cost for unselected candidates.
    scores = []
    for index, candidate in enumerate(candidates):
        trials = [row for row in history if row["family"] == candidate["family"]]
        successes = sum(row["improved"] for row in trials)
        n = len(trials)
        mean = (1 + successes) / (2 + n)
        variance = ((1 + successes) * (1 + n - successes)) / ((2 + n) ** 2 * (3 + n))
        scores.append((mean + variance**0.5, -index))
    return max(range(len(scores)), key=lambda i: scores[i])


def _random_index(seed, episode_index, count):
    raw = int(identity({"seed": seed, "episode": episode_index})[:8], 16)
    return raw % count


def compare(episodes, *, frozen, seed=None):
    """Replay every policy over the same episodes; never look ahead."""
    episodes = [episode for episode in episodes if len(episode["candidates"]) >= 2]
    seed = frozen["seed"] if seed is None else seed
    if len(episodes) < frozen["min_episodes"]:
        return {
            "status": "insufficient_data",
            "episodes": len(episodes),
            "required": frozen["min_episodes"],
            "design_sha256": identity(frozen),
            "improved_future_selection_demonstrated": False,
            "reason": "Not enough episodes with a logged candidate set; no policy claim is made.",
        }
    results = {policy: [] for policy in frozen["policies"]}
    selections = []
    history = []
    for episode_index, episode in enumerate(episodes):
        candidates = episode["candidates"]
        row = {"experiment_id": episode["experiment_id"], "choices": {}}
        for policy in frozen["policies"]:
            if policy == "random_fixed_seed":
                index = _random_index(seed, episode_index, len(candidates))
            else:
                index = _select(policy, candidates, history, seed=seed, episode_index=episode_index)
            chosen = candidates[index]
            delta = chosen["practice"] - chosen["baseline"]
            results[policy].append(delta)
            row["choices"][policy] = {
                "ordinal": chosen["ordinal"],
                "family": chosen["family"],
                "practice_delta": delta,
            }
        selections.append(row)
        # Update history only AFTER every policy has chosen for this episode.
        history.extend(
            {
                "family": candidate["family"],
                "improved": candidate["practice"] > candidate["baseline"],
            }
            for candidate in candidates
        )
    primary, null = frozen["primary"], frozen["null"]
    differences = [a - b for a, b in zip(results[primary], results[null])]
    mean_primary = sum(results[primary]) / len(results[primary])
    mean_null = sum(results[null]) / len(results[null])
    test = sign_flip_test(differences, seed=seed, draws=10_000)
    decisive = test["p_value"] <= frozen["alpha"] and mean_primary > mean_null
    return {
        "status": "estimated",
        "episodes": len(episodes),
        "design_sha256": identity(frozen),
        "mean_practice_delta": results,
        "mean_by_policy": {policy: sum(v) / len(v) for policy, v in results.items()},
        "primary_minus_null": {
            "primary": primary,
            "null": null,
            "mean_difference": mean_primary - mean_null,
            "episodes_primary_better": sum(d > 0 for d in differences),
            "episodes_null_better": sum(d < 0 for d in differences),
            "ties": sum(d == 0 for d in differences),
            "p_value": test["p_value"],
            "method": test["method"],
        },
        "decision": "policy_better_on_replay" if decisive else "not_established",
        "improved_future_selection_demonstrated": decisive,
        "scope": "offline replay of recorded counterfactual candidates; not online learning",
        "selections": selections,
    }


def analyse(memory, *, alpha=0.05, min_episodes=20, seed=17, directory=None):
    """Freeze the design, then replay; optionally persist the frozen design first."""
    frozen = design(alpha=alpha, min_episodes=min_episodes, seed=seed)
    binding = None
    if directory is not None:
        binding = write_once(Path(directory) / "selection_design.json", frozen)
    episodes = memory.candidate_episodes()
    # One context at a time: the posterior is context-specific.
    contexts = {}
    for episode in episodes:
        contexts.setdefault(episode["context"], []).append(episode)
    context, rows = max(contexts.items(), key=lambda item: len(item[1])) if contexts else ("", [])
    result = compare(rows, frozen=frozen, seed=seed)
    return {
        **result,
        "design": frozen,
        "design_binding": binding,
        "context": context,
        "contexts_observed": len(contexts),
        "episodes_logged": sum(len(value) for value in contexts.values()),
        "note": (
            "A positive result here would mean only that, on this replay of recorded outcomes, "
            "one selection rule picked better-measured candidates than another. It is not "
            "online improvement, not a trained researcher, and not a capability claim."
        ),
    }


def fixture_episodes(*, episodes=24, candidates=2, seed=17, advantage=0.6):
    """Deterministic synthetic episodes for exercising the machinery. Never evidence."""
    rows = []
    for episode in range(episodes):
        # The strong family moves around in the proposal order, so "pick the first proposal"
        # is only right by chance while a family-aware policy can learn it.
        strong = int(identity({"fixture": seed, "episode": episode})[:8], 16) % candidates
        log = []
        for ordinal in range(candidates):
            draw = int(
                identity({"fixture": seed, "episode": episode, "candidate": ordinal})[:8], 16
            )
            probability = advantage if ordinal == strong else 1 - advantage
            improvement = 0.2 if (draw % 1000) / 1000 < probability else 0.0
            log.append(
                {
                    "ordinal": ordinal,
                    "family": "strong" if ordinal == strong else "weak",
                    "recipe_sha256": identity(
                        {"recipe": seed, "episode": episode, "ordinal": ordinal}
                    ),
                    "practice": 0.5 + improvement,
                    "baseline": 0.5,
                    "gate_passed": False,
                    "selected": False,
                }
            )
        rows.append(
            {
                "experiment_id": f"fixture-{episode:03d}",
                "context": "fixture-context",
                "created": float(episode),
                "candidates": log,
            }
        )
    return rows


def compare_fixture(*, episodes=48, candidates=2, seed=17, min_episodes=20, advantage=0.85):
    """Run the replay on synthetic episodes and label the output as non-evidence."""
    frozen = design(min_episodes=min_episodes, seed=seed)
    result = compare(
        fixture_episodes(episodes=episodes, candidates=candidates, seed=seed, advantage=advantage),
        frozen=frozen,
        seed=seed,
    )
    return {
        **result,
        "design": frozen,
        "fixture": True,
        "scientific_evidence": False,
        "note": (
            "Generated episodes with a known advantage. This exercises the comparison code "
            "path only; it says nothing about this repository's recorded experiments."
        ),
    }


# --- magnitude_v2 (G6) -------------------------------------------------------------------
# The v1 design scores a proposal with the binary event "did the candidate beat baseline?".
# Both families beat it in every one of the 20 recorded episodes, so v1 tied in 20/20 and had
# no resolution left. v2 keeps the same arms and the same exact, distribution-free test but
# changes BOTH the statistic (signed magnitude of the practice gain) and the arm policy (UCB on
# observed magnitudes with forced exploration). That makes it a *new* predeclared design with a
# new hash; the v1 result stays frozen and is never re-tuned.

POLICIES_V2 = ("proposal_order", "magnitude_ucb", "random_fixed_seed")


def design_magnitude(
    *,
    alpha=0.05,
    min_episodes=20,
    seed=17,
    primary="magnitude_ucb",
    null="proposal_order",
    noise_floor=0.02,
):
    if primary == null or primary not in POLICIES_V2 or null not in POLICIES_V2:
        raise ValueError("Primary and null policies must be distinct known v2 policies")
    if not 0 < alpha < 1 or min_episodes < 6:
        raise ValueError("Predeclare a sane alpha and at least six episodes")
    if not 0 <= noise_floor < 0.5:
        raise ValueError("The magnitude noise floor must be small and declared up front")
    return {
        "schema_version": 1,
        "kind": "predeclared-magnitude-selection-comparison",
        "policies": list(POLICIES_V2),
        "primary": primary,
        "null": null,
        "episode_order": "recorded experiment creation order (ascending)",
        "metric": "measured practice macro-accuracy gain of the selected candidate (magnitude)",
        "arm_policy": "mean observed magnitude + UCB exploration bonus, unvisited arms explored first",
        "information_set": "only outcomes from strictly earlier episodes",
        "pairing": "the same episode sequence is scored by every policy",
        "decision_rule": (
            "one-sided paired sign-flip test on signed magnitude differences, "
            "requiring a positive median; |difference| < noise_floor counts as a tie"
        ),
        "noise_floor": noise_floor,
        "alpha": alpha,
        "min_episodes": min_episodes,
        "seed": seed,
        "frozen_before_analysis": True,
        "supersedes": "binary win/loss statistic of selection-v1 (frozen, never re-tuned)",
        "online_learning": False,
        "researcher_trained": False,
        "hidden_data_used": False,
    }


def _magnitude_index(candidates, history, *, episode_index):
    """UCB over observed magnitudes; every arm is tried once before any arm is preferred."""
    totals, counts = {}, {}
    for row in history:
        totals[row["family"]] = totals.get(row["family"], 0.0) + row["delta"]
        counts[row["family"]] = counts.get(row["family"], 0) + 1
    seen = sum(counts.values())
    scores = []
    for index, candidate in enumerate(candidates):
        family = candidate["family"]
        n = counts.get(family, 0)
        if n == 0:
            scores.append((float("inf"), -index))
            continue
        mean = totals[family] / n
        bonus = (2.0 * (seen + 1) / n) ** 0.5
        scores.append((mean + bonus, -index))
    return max(range(len(scores)), key=lambda i: scores[i])


def _select_v2(policy, candidates, history, *, seed, episode_index):
    if policy == "proposal_order":
        return 0
    if policy == "magnitude_ucb":
        return _magnitude_index(candidates, history, episode_index=episode_index)
    raise ValueError("random_fixed_seed is handled by the caller")


def compare_magnitude(episodes, *, frozen, seed=None):
    """Replay the magnitude design over recorded episodes; never look ahead."""
    episodes = [episode for episode in episodes if len(episode["candidates"]) >= 2]
    seed = frozen["seed"] if seed is None else seed
    if len(episodes) < frozen["min_episodes"]:
        return {
            "status": "insufficient_data",
            "episodes": len(episodes),
            "required": frozen["min_episodes"],
            "design_sha256": identity(frozen),
            "improved_future_selection_demonstrated": False,
            "reason": "Not enough episodes with a logged candidate set; no policy claim is made.",
        }
    results = {policy: [] for policy in frozen["policies"]}
    selections, history = [], []
    for episode_index, episode in enumerate(episodes):
        candidates = episode["candidates"]
        row = {"experiment_id": episode["experiment_id"], "choices": {}}
        for policy in frozen["policies"]:
            if policy == "random_fixed_seed":
                index = _random_index(seed, episode_index, len(candidates))
            else:
                index = _select_v2(
                    policy, candidates, history, seed=seed, episode_index=episode_index
                )
            chosen = candidates[index]
            delta = chosen["practice"] - chosen["baseline"]
            results[policy].append(delta)
            row["choices"][policy] = {
                "ordinal": chosen["ordinal"],
                "family": chosen["family"],
                "practice_delta": delta,
            }
        selections.append(row)
        history.extend(
            {
                "family": candidate["family"],
                "delta": candidate["practice"] - candidate["baseline"],
            }
            for candidate in candidates
        )
    primary, null = frozen["primary"], frozen["null"]
    floor = frozen["noise_floor"]
    raw = [a - b for a, b in zip(results[primary], results[null])]
    differences = [0.0 if abs(value) < floor else value for value in raw]
    ordered = sorted(differences)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    mean_primary = sum(results[primary]) / len(results[primary])
    mean_null = sum(results[null]) / len(results[null])
    test = sign_flip_test(differences, seed=seed, draws=10_000)
    decisive = test["p_value"] <= frozen["alpha"] and median > 0
    return {
        "status": "estimated",
        "episodes": len(episodes),
        "design_sha256": identity(frozen),
        "mean_practice_delta": results,
        "mean_by_policy": {policy: sum(v) / len(v) for policy, v in results.items()},
        "primary_minus_null": {
            "primary": primary,
            "null": null,
            "mean_difference": mean_primary - mean_null,
            "median_difference": median,
            "episodes_primary_better": sum(d > 0 for d in differences),
            "episodes_null_better": sum(d < 0 for d in differences),
            "ties": sum(d == 0 for d in differences),
            "raw_mean_difference": sum(raw) / len(raw),
            "noise_floor": floor,
            "p_value": test["p_value"],
            "method": test["method"],
        },
        "decision": "policy_better_on_replay" if decisive else "not_established",
        "improved_future_selection_demonstrated": decisive,
        "scope": "offline replay of recorded counterfactual candidates; not online learning",
        "selections": selections,
    }


def analyse_magnitude(memory, *, alpha=0.05, min_episodes=20, seed=17, directory=None):
    """Freeze the magnitude design, then replay; persist the frozen design first."""
    frozen = design_magnitude(alpha=alpha, min_episodes=min_episodes, seed=seed)
    binding = None
    if directory is not None:
        binding = write_once(Path(directory) / "selection_design_magnitude.json", frozen)
    episodes = memory.candidate_episodes()
    contexts = {}
    for episode in episodes:
        contexts.setdefault(episode["context"], []).append(episode)
    context, rows = max(contexts.items(), key=lambda item: len(item[1])) if contexts else ("", [])
    result = compare_magnitude(rows, frozen=frozen, seed=seed)
    return {
        **result,
        "design": frozen,
        "design_binding": binding,
        "context": context,
        "contexts_observed": len(contexts),
        "episodes_logged": sum(len(value) for value in contexts.values()),
        "note": (
            "A replay over an already-observed log is exploratory re-analysis. Confirmation "
            "needs episodes collected after this design was frozen."
        ),
    }
