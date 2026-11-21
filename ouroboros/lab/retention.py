"""After-search fixed-intervention retention probe; never an editor observation or selector."""

from ..core.evaluator import SelfEvaluator
from ..self_edit import model_fingerprint
from .journal import candidate_hash, compare_candidates, verify_roundtrip
from .metrics import paired_summary
from .program import build_program


def domain_floors(original, selected, subsequent):
    a, b, c = (value["summary"]["scores"] for value in (original, selected, subsequent))
    floors = {key: max(a[key], b[key]) for key in a}
    return {
        "passed": all(c[key] + 1e-12 >= floor for key, floor in floors.items()),
        "floors": floors,
        "scores": c,
        "rule": "Every domain >= both original and pre-followup candidate, zero tolerance; equality is retention, never improvement.",
    }


def final_assessment(paired, policy):
    passed, reason = SelfEvaluator(paired["baseline"]["scores"]).verify_improvement(
        paired["candidate"]["scores"],
        forgetting_threshold=policy.forgetting_threshold,
        improvement_threshold=policy.improvement_threshold,
    )
    if passed and paired["problem_sign_flip"]["p_value"] >= policy.alpha:
        passed, reason = False, "Final problem-clustered significance threshold not met"
    return {"strict_independent_gain": passed, "reason": reason, "used_for_promotion": False}


def retention_probe(session, candidate, *, candidate_kind):
    """Four-cell probe: base, edit, subsequent-only, edit+subsequent. Same fixed edit in both cells."""
    spec, copy = session.discovery, session.copy
    resident = model_fingerprint(copy.model)
    prior = copy.candidate
    counters_before = (
        session.generated_sequences,
        session.loss_evaluated_problems,
        session.cache_hits,
    )
    report = {
        "status": "running",
        "candidate_kind": candidate_kind,
        "original_checkpoint_sha256": session.target_sha256,
        "candidate_sha256": candidate_hash(candidate),
        "used_for_promotion": False,
        "post_test_reselection_allowed": False,
        "intervention": {
            "authorship": "preregistered host-designed subsequent edit, not autonomous training",
            "operations": spec["followup_operations"],
            "hypothesis": "A useful edit should survive this independently fixed subsequent perturbation without a category falling below original or pre-followup performance.",
            "falsification": "Any retention category below either floor. No assessment of learned future editing ability.",
        },
    }
    try:
        copy.candidate = {}
        only, only_validation = build_program(copy, spec["followup_operations"])
        report["subsequent_only_changes"] = compare_candidates(copy, {}, only, full=True)
        report["subsequent_only_roundtrip_replay"] = verify_roundtrip(
            copy, {}, only, report["subsequent_only_changes"]
        )
        report["subsequent_only_validation"] = only_validation
        copy.candidate = candidate
        combined, combined_validation = build_program(copy, spec["followup_operations"])
        report["changes"] = compare_candidates(copy, candidate, combined, full=True)
        report["roundtrip_replay"] = verify_roundtrip(copy, candidate, combined, report["changes"])
        samples = spec["retention"]
        baseline = session.measure(samples, {}, seed=session.policy.test_seed + 1)
        selected = (
            session.measure(samples, candidate, seed=session.policy.test_seed + 1)
            if candidate
            else baseline
        )
        subsequent_only = session.measure(samples, only, seed=session.policy.test_seed + 1)
        subsequent_combined = session.measure(samples, combined, seed=session.policy.test_seed + 1)
        changes = compare_candidates(copy, candidate, combined, full=True)
        report.update(
            status="completed",
            validation={"subsequent_only": only_validation, "combined": combined_validation},
            changes=changes,
            roundtrip_replay=verify_roundtrip(copy, candidate, combined, changes),
            baseline_vs_candidate=paired_summary(
                baseline,
                selected,
                seed=session.policy.test_seed + 1,
                draws=session.policy.permutation_draws,
            ),
            candidate_vs_after=paired_summary(
                selected,
                subsequent_combined,
                seed=session.policy.test_seed + 1,
                draws=session.policy.permutation_draws,
            ),
            subsequent_only_vs_combined=paired_summary(
                subsequent_only,
                subsequent_combined,
                seed=session.policy.test_seed + 1,
                draws=session.policy.permutation_draws,
            ),
            category_retention=domain_floors(baseline, selected, subsequent_combined),
            measurement={
                "baseline": baseline,
                "candidate": selected,
                "subsequent_only": subsequent_only,
                "combined": subsequent_combined,
            },
        )
        loss_floors = {
            key: min(
                baseline["summary"]["loss"]["by_domain"][key],
                selected["summary"]["loss"]["by_domain"][key],
            )
            for key in baseline["summary"]["scores"]
        }
        after_losses = subsequent_combined["summary"]["loss"]["by_domain"]
        report["completion_nll_retention"] = {
            "floors_lower_is_better": loss_floors,
            "after": after_losses,
            "passed": all(after_losses[key] <= floor for key, floor in loss_floors.items()),
            "used_for_promotion": False,
            "note": "Descriptive loss retention is distinct from the primary accuracy/category gate; a tiny loss regression remains visible.",
        }
        report["retention_of_validated_useful_edit"] = (
            report["category_retention"]["passed"] if candidate_kind == "accepted" else None
        )
    except ValueError as exc:
        # A combined norm/count violation is a measured refusal, not license to shrink the
        # followup dose until it fits. Other failures propagate to the enclosing run.
        report.update(
            status="refused",
            reason=f"{type(exc).__name__}: {exc}",
            retention_of_validated_useful_edit=None,
        )
    finally:
        copy.candidate = prior
        report["base_restored"] = model_fingerprint(copy.model) == resident
        report["actual_generated_sequences"] = session.generated_sequences - counters_before[0]
        report["actual_loss_evaluated_problems"] = (
            session.loss_evaluated_problems - counters_before[1]
        )
        report["measurement_cache_hits"] = session.cache_hits - counters_before[2]
    return report
