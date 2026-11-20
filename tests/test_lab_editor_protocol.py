"""Frozen-editor feedback/diagnosis/real-practice-contrast boundaries, not LLM gain evidence."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ouroboros.lab.config import Sampling, normalize_editor
from ouroboros.lab.runner import _editor_prompt, run_lab, run_practice_diagnosis
from ouroboros.lab.session import BudgetExceeded, LabSession
from ouroboros.lab.state import WorkingCopy
from ouroboros.self_edit import model_fingerprint

pytestmark = pytest.mark.ml


def _run(backend, dataset, policy, tmp_path, *, editor=None, sampling=None):
    return run_lab(
        backend,
        dataset,
        policy,
        tmp_path,
        sampling=sampling or Sampling(),
        editor={"mode": "model", **(editor or {})},
        provenance="Torch classifier / scripted editor fixture, not a pretrained capability result",
        license_name="MIT",
    )


def test_parse_error_reaches_next_editor_reply_and_every_failed_reply_is_metered(
    lab_backend, lab_dataset, lab_policy, tmp_path
):
    lab_backend.invalid_reply = True
    report = _run(lab_backend, lab_dataset, replace(lab_policy, max_invalid_actions=2), tmp_path)
    session = report["sessions"][0]
    assert session["counters"]["turns"] == session["counters"]["invalid_actions"] == 2
    assert session["status"] == "budget_exhausted" and report["base_restored"]
    turn = json.loads(
        (Path(report["report_path"]).parent / session["events"][0]["artifact"]).read_text()
    )
    assert "Invalid editor reply:" in turn["observation"]["error"]
    assert "Unknown research action" not in turn["observation"]["error"]
    assert turn["action"] is None and not session["events"][0]["parsed_json"]
    feedback = json.loads(lab_backend.editor_prompts[1])["recent_conversation"][-1]["observation"]
    assert feedback["error"] == turn["observation"]["error"]
    assert feedback["repair"]["minimal_example"] == {"action": "withdraw"}
    assert "RESERVED-QUESTION" not in json.dumps(turn) and "FINAL-QUESTION" not in json.dumps(turn)
    assert turn["editor_generation"]["new_tokens"] == 1


def test_semantic_feedback_names_missing_fields_and_does_not_replace_actions(
    lab_backend, lab_dataset, lab_policy, tmp_path
):
    lab_backend.editor_actions = [{"action": "test"}, {"action": "withdraw"}]
    report = _run(lab_backend, lab_dataset, lab_policy, tmp_path)
    conversation = json.loads(lab_backend.editor_prompts[1])["recent_conversation"][-1]
    assert "missing=['ids']" in conversation["observation"]["error"]
    assert conversation["observation"]["repair"]["required_fields"] == ["action", "ids"]
    assert report["sessions"][0]["status"] == "withdrawn"
    assert report["sessions"][0]["counters"]["edit_actions"] == 0


def test_editor_sampling_grammar_and_target_sampling_are_separate_with_truthful_metadata(
    lab_backend, lab_dataset, lab_policy, tmp_path, monkeypatch
):
    pytest.importorskip("lmformatenforcer")
    calls, original = [], lab_backend.generate_records

    def generation(requests, **kwargs):
        calls.append((requests[0]["prompt"], dict(kwargs)))
        kwargs.pop(
            "json_schema", None
        )  # stub does not enforce syntax; real decoder tested separately
        return original(requests, **kwargs)

    monkeypatch.setattr(lab_backend, "generate_records", generation)
    lab_backend.editor_actions = [{"action": "withdraw"}]
    report = _run(
        lab_backend,
        lab_dataset,
        lab_policy,
        tmp_path,
        sampling=Sampling(temperature=0.6),
        editor={"decoding": "json_schema", "sampling": {"temperature": 0.0}},
    )
    editor_calls = [kwargs for prompt, kwargs in calls if '"recent_conversation"' in prompt]
    answer_calls = [kwargs for prompt, kwargs in calls if '"recent_conversation"' not in prompt]
    assert len(editor_calls) == 1
    assert editor_calls[0]["json_schema"]["anyOf"]
    assert editor_calls[0]["sampling"].temperature == 0
    assert all(
        kwargs["sampling"].temperature == 0.6 and "json_schema" not in kwargs
        for kwargs in answer_calls
    )
    assert report["editor_decoding"] == "json_schema" and report["model_authored_actions"]
    assert report["evaluation_plan"]["editor"]["sampling"]["temperature"] == 0
    assert (
        report["model"]["is_llm"] is False
    )  # classifier with scripted replies, never called an LLM


def test_repeat_warning_and_latest_error_survive_bounded_history(
    lab_backend, lab_dataset, lab_policy, tmp_path
):
    lab_backend.editor_actions = [{"action": "inventory", "limit": 1}] * 2 + [
        {"action": "withdraw"}
    ]
    _run(lab_backend, lab_dataset, lab_policy, tmp_path)
    latest = json.loads(lab_backend.editor_prompts[2])["recent_conversation"][-1]
    assert "not new evidence" in latest["observation"]["repeat_warning"]
    session = LabSession(
        lab_backend,
        WorkingCopy(lab_backend.model, replace(lab_policy, max_history_chars=2000)),
        lab_dataset,
        sampling=Sampling(),
        grader="exact",
    )
    diagnosis = {"practice": [], "summary": {}, "readiness": {}}
    history = [{"action": {"action": "test"}, "observation": {"error": "important " * 500}}]
    with pytest.raises(ValueError, match="Latest feedback was not silently removed"):
        _editor_prompt(lab_backend, session, diagnosis, history, normalize_editor(), False)


def test_practice_diagnosis_never_opens_reserved_or_final_inference_or_edits(
    lab_backend, lab_dataset, lab_policy, tmp_path
):
    import torch

    before = model_fingerprint(lab_backend.model)
    rng = torch.get_rng_state().clone()
    flags = [module.training for module in lab_backend.model.modules()]
    report = run_practice_diagnosis(
        lab_backend,
        lab_dataset,
        lab_policy,
        tmp_path,
        sampling=Sampling(),
    )
    assert report["status"] == "completed" and report["diagnosis"]["summary"]["accuracy"] == 0.5
    assert report["diagnosis"]["readiness"]["measured_contrast_available"]
    assert report["reserved_generated_sequences"] == report["final_generated_sequences"] == 0
    assert report["editor_generations"] == report["staged_edits"] == 0
    assert len(lab_backend.requests) == 2 and all(
        "practice" in row["prompt"] for row in lab_backend.requests
    )
    assert "RESERVED-QUESTION" not in json.dumps(report) and "FINAL-QUESTION" not in json.dumps(
        report
    )
    assert report["base_restored"] and before == model_fingerprint(lab_backend.model)
    assert torch.equal(rng, torch.get_rng_state())
    assert flags == [module.training for module in lab_backend.model.modules()]
    assert not list(tmp_path.rglob("final_measurement.json"))


def test_practice_contrast_uses_only_actual_graded_completions_and_is_metered(
    lab_backend, lab_dataset, lab_policy, monkeypatch
):
    captured = {}

    def measurement(copy, backend, **kwargs):
        captured.update(kwargs)
        assert model_fingerprint(backend.model) == before
        copy.directions["d0"] = [1.0, 0.0]
        return {"direction_id": "d0", "measurement": {"kind": "fixture stub"}}

    monkeypatch.setattr("ouroboros.lab.session.measure_direction", measurement)
    before = model_fingerprint(lab_backend.model)
    lab = LabSession(
        lab_backend,
        WorkingCopy(lab_backend.model, lab_policy),
        lab_dataset,
        sampling=Sampling(),
        grader="exact",
    )
    result = lab.act({"action": "contrast_practice", "module": "a", "ids": [0, 1]})
    assert result["available"] and result["source"] == "actual_graded_practice_generations"
    assert captured["failing"] == [
        "practice warm\ncool"
    ]  # actual wrong classifier answer, not reference
    assert captured["successful"] == ["practice cool\ncool"]
    assert result["practice_summary"]["accuracy"] == 0.5
    assert [row["text"] for row in result["practice_records"]] == ["cool", "cool"]
    assert (
        result["counters"]["forward_calls"] == 2 and result["counters"]["generated_sequences"] == 2
    )
    assert "RESERVED-QUESTION" not in json.dumps(result) and "FINAL-QUESTION" not in json.dumps(
        result
    )
    assert all(row["correct"] for row in result["source_records"]["successful"])
    assert all(not row["correct"] for row in result["source_records"]["failing"])
    assert model_fingerprint(lab_backend.model) == before


def test_practice_contrast_does_not_invent_a_missing_cohort_or_use_unmetered_reads(
    lab_backend, lab_dataset, lab_policy, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("No hidden read without two measured cohorts")

    monkeypatch.setattr("ouroboros.lab.session.measure_direction", forbidden)
    lab = LabSession(
        lab_backend,
        WorkingCopy(lab_backend.model, lab_policy),
        lab_dataset,
        sampling=Sampling(),
        grader="exact",
    )
    result = lab.act({"action": "contrast_practice", "module": "a", "ids": [1]})
    assert not result["available"] and "No synthetic" in result["reason"]
    assert not lab.copy.directions and result["counters"]["forward_calls"] == 1
    limited = LabSession(
        lab_backend,
        WorkingCopy(lab_backend.model, replace(lab_policy, max_forward_calls=1)),
        lab_dataset,
        sampling=Sampling(),
        grader="exact",
    )
    with pytest.raises(BudgetExceeded, match="forward"):
        limited.act({"action": "contrast_practice", "module": "a", "ids": [0, 1]})
    assert limited.forward_calls == 1 and not limited.copy.directions


def test_no_progress_stop_is_not_a_host_written_edit_or_model_withdrawal(
    lab_backend, lab_dataset, lab_policy, tmp_path
):
    lab_backend.editor_actions = [
        {"action": "inventory", "limit": 1, "rationale": str(i)} for i in range(5)
    ]
    report = _run(lab_backend, lab_dataset, replace(lab_policy, max_repeated_actions=3), tmp_path)
    record = report["sessions"][0]
    assert record["status"] == "stalled" and record["editor_generations"] == 3
    assert record["counters"]["invalid_actions"] == 0 and record["counters"]["edit_actions"] == 0
    assert record["counters"]["turns"] == 3 and not record["patch"]
    assert "host stopped" in record["reason"] and report["selected_session"] is None
    assert report["base_restored"] and report["final_test"]["candidate_is_baseline"]
    archived = json.loads(Path(report["report_path"]).read_text())
    assert len(archived["sessions"][0]["events"]) == 3


def test_generated_command_is_archived_even_when_a_tool_request_budget_is_exhausted(
    lab_backend, lab_dataset, lab_policy, tmp_path
):
    lab_backend.editor_actions = [
        {"action": "test", "ids": [0]},
        {"action": "test", "ids": [1]},
    ]
    report = _run(lab_backend, lab_dataset, replace(lab_policy, max_forward_calls=1), tmp_path)
    session = report["sessions"][0]
    assert session["status"] == "budget_exhausted"
    assert session["editor_generations"] == len(session["events"]) == 2
    last = session["events"][-1]
    assert last["action"] == "test" and last["parsed_json"] and last["status"] == "budget_exhausted"
    turn = json.loads((Path(report["report_path"]).parent / last["artifact"]).read_text())
    assert turn["raw_editor_reply"] and "forward-request budget" in turn["observation"]["error"]
    assert session["counters"]["invalid_actions"] == 0 and report["base_restored"]


def test_schema_context_exposes_only_public_names_practice_counts_and_existing_ids(
    lab_backend, lab_dataset, lab_policy
):
    session = LabSession(
        lab_backend,
        WorkingCopy(lab_backend.model, lab_policy),
        lab_dataset,
        sampling=Sampling(),
        grader="exact",
    )
    context = session.schema_context()
    assert context["practice_count"] == 2 and context["parameters"]
    assert context["directions"] == context["saved"] == context["edited"] == []
    text = json.dumps(context)
    assert (
        "RESERVED-QUESTION" not in text
        and "FINAL-QUESTION" not in text
        and "practice warm" not in text
    )


def test_editor_context_excludes_wallclock_counters_for_reproducible_comparisons(
    lab_backend, lab_dataset, lab_policy
):
    from ouroboros.lab.runner import _preview

    lab = LabSession(
        lab_backend,
        WorkingCopy(lab_backend.model, lab_policy),
        lab_dataset,
        sampling=Sampling(),
        grader="exact",
    )
    diagnosis = {"practice": [], "summary": {}, "readiness": {}}
    history = [
        {
            "action": {"action": "inventory"},
            "observation": _preview({"counters": {"turns": 1, "elapsed_seconds": 100.5}}),
        }
    ]
    first = _editor_prompt(lab_backend, lab, diagnosis, history, normalize_editor(), False)
    lab.started -= 150
    second = _editor_prompt(lab_backend, lab, diagnosis, history, normalize_editor(), False)
    assert first == second and "elapsed_seconds" not in first
    assert "elapsed_seconds" not in json.dumps(
        _preview({"counters": {"elapsed_seconds": 1.5, "turns": 1}})
    )
    assert "elapsed_seconds" in lab.counters()  # real timing remains archived/host-enforced
