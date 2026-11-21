"""Practice-only experiment readiness. These diagnostics never change a grade or a target."""

from .metrics import problem_id


def text_preview(text, limit=800):
    if len(text) <= limit:
        return text
    # Keep the end as well as the beginning: a prefix-only preview can hide a final answer.
    marker = " [preview clipped; full text archived] "
    head = (limit - len(marker)) * 2 // 3
    tail = limit - len(marker) - head
    return text[:head] + marker + text[-tail:]


def practice_diagnosis(samples, measurement, *, answer_tokens):
    """Report actual graded completions and poor-signal warnings; no host-authored contrasts."""
    rows, summary = measurement["records"], measurement["summary"]
    correct = sum(row["correct"] for row in rows)
    failed = len(rows) - correct
    cut = sum(row["cut"] for row in rows)
    open_thinking = sum(not row["thinking_closed"] for row in rows)
    warnings = []
    if len(samples) < 8:
        warnings.append(
            "Tiny practice fixture: fewer than eight problems cannot support a broad capability claim."
        )
    if cut:
        warnings.append(
            "Some answers hit the completion limit. Inspect them and set an adequate FIXED answer budget before a new experiment; cuts still score incorrect."
        )
    if open_thinking:
        warnings.append("Some answers never closed thinking; they still score incorrect.")
    if not correct:
        warnings.append(
            "All practice answers failed the selected strict grader. Inspect completion/formatting and task suitability; there are no measured successful outputs for a practice contrast."
        )
    if not failed:
        warnings.append(
            "Practice is at ceiling. There are no measured failures to diagnose; do not manufacture a gain by changing targets or consulting held-out results."
        )
    problems = []
    for index, sample in enumerate(samples):
        answers = [row for row in rows if row["id"] == problem_id(sample)]
        problems.append(
            {
                "id": index,
                "instruction": text_preview(sample.instruction),
                "reference": text_preview(sample.target),
                "domain": sample.domain,
                "answers": [
                    {
                        **{
                            key: row[key]
                            for key in (
                                "sample",
                                "seed",
                                "correct",
                                "cut",
                                "thinking_closed",
                                "new_tokens",
                                "repetitive",
                            )
                        },
                        "text": text_preview(row["text"]),
                    }
                    for row in answers[:4]
                ],
                "omitted_answers": max(0, len(answers) - 4),
            }
        )
    return {
        "split": "practice_only",
        "answer_max_new_tokens": answer_tokens,
        "summary": summary,
        "readiness": {
            "signal": "mixed_correct_and_incorrect"
            if correct and failed
            else "all_incorrect"
            if failed
            else "all_correct",
            "correct_answers": correct,
            "incorrect_answers": failed,
            "cut_answers": cut,
            "open_thinking_answers": open_thinking,
            "measured_contrast_available": bool(correct and failed),
            "warnings": warnings,
            "note": "Practice readiness is a diagnostic, not a promotion gate, causal interpretation or benchmark. No grader/target/budget is adjusted during the run.",
        },
        "practice": problems,
    }


def diagnosis_preview(diagnosis):
    """Keep both observed outcome types when a small editor cannot hold every example."""
    problems = diagnosis["practice"]
    failed = [p for p in problems if any(not row["correct"] for row in p["answers"])]
    correct = [p for p in problems if any(row["correct"] for row in p["answers"])]
    chosen = []
    for problem in failed[:2] + correct[:2] + problems:
        if problem["id"] not in {p["id"] for p in chosen}:
            chosen.append(problem)
        if len(chosen) >= 6:
            break
    return {
        **diagnosis,
        "practice": chosen,
        "omitted_problems": len(problems) - len(chosen),
    }
