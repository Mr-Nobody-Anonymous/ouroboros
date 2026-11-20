"""Lightweight research schema/statistics tests: no pretrained data, Torch or model download."""

import json
import math
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from ouroboros.cli import main
from ouroboros.lab.config import (
    LabPolicy,
    Sampling,
    input_fingerprint,
    lab_config_summary,
    load_lab_config,
    strict_json,
)
from ouroboros.lab.metrics import (
    grade,
    last_boxed,
    paired_summary,
    parse_action,
    repetition,
    sample_seed,
    sign_flip_test,
    summarize,
)
from ouroboros.project import init_project


def test_packaged_research_project_check_cli_and_tools_do_not_import_torch(tmp_path, capsys):
    root = tmp_path / "lab"
    init_project(root)
    path = root / "research_lab.json"
    config = load_lab_config(path)
    summary = lab_config_summary(config)
    assert summary["valid"] and not summary["model_loaded"]
    assert summary["counts"] == {"train": 2, "validation": 4, "test": 2}
    assert summary["editor_mode"] == "model"
    assert "Rome" not in json.dumps(summary) and "capital of" not in json.dumps(summary)
    assert main(["lab", "run", "--config", str(path), "--check-only"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    code = "from ouroboros.cli import main; from ouroboros.lab.patches import verify_bundle; import sys; assert main(['lab','run','--config',sys.argv[1],'--check-only']) == 0; assert main(['lab','tools']) == 0; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    subprocess.run([sys.executable, "-c", code, str(path)], check=True, capture_output=True)


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_turns", 161),
        ("max_sessions", 9),
        ("max_forward_calls", True),
        ("max_edited_fraction", 0),
        ("max_relative_norm", float("nan")),
        ("seed", -1),
        ("seed", 20261119),
        ("attention_maps", "true"),
        ("max_changed_entries", 0),
        ("max_patch_bytes", 1023),
        ("permutation_draws", 99),
    ],
)
def test_lab_policy_is_bounded_and_rejects_boolean_numbers(key, value):
    with pytest.raises(ValueError):
        LabPolicy(**{key: value})


def test_explicit_research_resource_profile_can_exceed_smoke_limits_without_loading():
    policy = LabPolicy(
        max_model_parameters=30_000_000_000,
        max_model_bytes=80 * 1024**3,
        max_turns=160,
        max_edit_actions=200,
        max_seconds=10800,
        max_read_tokens=8192,
    )
    assert policy.max_model_parameters > 27_000_000_000
    assert replace(policy, max_sessions=8).max_sessions == 8
    with pytest.raises(ValueError):
        Sampling(temperature=float("inf"))
    with pytest.raises(ValueError):
        Sampling(top_k=True)


def test_lab_config_approvals_bind_splits_actions_and_workspace_paths(tmp_path):
    root = tmp_path / "lab"
    init_project(root)
    path = root / "research_lab.json"
    original = load_lab_config(path, workspace=root)
    digest = input_fingerprint(original)
    (root / "practice.json").write_text((root / "practice.json").read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        input_fingerprint(original)
    assert input_fingerprint(load_lab_config(path, workspace=root)) != digest
    values = json.loads(path.read_text())
    values["output_dir"] = "../outside"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="escapes"):
        load_lab_config(path, workspace=root)
    values["output_dir"] = "outputs/research"
    values["editor"] = {"mode": "scripted", "actions": "actions.json"}
    (root / "actions.json").write_text('[{"action":"withdraw"}]')
    path.write_text(json.dumps(values))
    config = load_lab_config(path, workspace=root)
    assert config["actions"] == [{"action": "withdraw"}]
    assert str(root / "actions.json") in config["input_hashes"]
    values["editor"]["mode"] = "model"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="scripted actions"):
        load_lab_config(path)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', "[Infinity]"])
def test_json_rejects_duplicate_keys_nonfinite_and_overflow(text):
    with pytest.raises(ValueError):
        strict_json(text)


def test_action_parser_is_data_only_not_a_python_parser():
    assert parse_action('```json\n{"action":"withdraw"}\n```') == {"action": "withdraw"}
    for text in (
        "import os",
        '{"action":"notes","text":NaN}',
        "[]",
        "{}",
        'prefix {"action":"withdraw"}',
    ):
        with pytest.raises(ValueError):
            parse_action(text)
    assert parse_action('{"action":"notes","text":"import os is plain data"}')["text"].startswith(
        "import os"
    )


def _records(grades, repeats):
    return [
        {
            "id": f"problem-{i // repeats}",
            "sample": i % repeats,
            "seed": i + 7,
            "domain": "math",
            "correct": bool(g),
            "cut": not bool(g),
            "repetitive": False,
            "new_tokens": 2,
            "text": str(g),
            "thinking_closed": True,
        }
        for i, g in enumerate(grades)
    ]


def _evaluation(grades, repeats):
    rows = _records(grades, repeats)
    return {"summary": summarize(rows), "records": rows}


def test_paired_equation_and_statistics_cluster_repeated_answers_by_problem():
    before = _evaluation([0, 0, 0, 0, 1, 1], 2)
    after = _evaluation([1, 1, 1, 1, 0, 0], 2)
    result = paired_summary(before, after)
    assert result["accuracy_delta"]["mean"] == pytest.approx(1 / 3)
    assert result["accuracy_delta"]["standard_error"] == pytest.approx(2 / 3)
    assert result["wrong_to_right"] == 4 and result["right_to_wrong"] == 2
    assert result["problem_sign_flip"]["p_value"] == 0.5
    assert "paired_sign_test_one_sided" not in result
    assert result["cohorts"]["originally_cut"]["candidate_correct"] == 4
    after["records"][0]["seed"] += 1
    with pytest.raises(ValueError, match="mismatch"):
        paired_summary(before, after)


def test_exact_and_monte_carlo_sign_flips_have_no_global_rng_side_effects():
    assert sign_flip_test([1, 1, 1])["p_value"] == 0.125
    assert sign_flip_test([0, 0])["p_value"] == 1
    state = np.random.get_state()
    first = sign_flip_test([0.5] * 20, seed=3, draws=1000)
    assert first == sign_flip_test([0.5] * 20, seed=3, draws=1000)
    assert first["p_value"] >= 1 / 1001
    assert np.array_equal(state[1], np.random.get_state()[1])
    singleton = paired_summary(_evaluation([0], 1), _evaluation([1], 1))
    assert singleton["accuracy_delta"]["standard_error"] is None
    assert singleton["paired_sign_test_one_sided"] == pytest.approx(0.5)
    assert sample_seed("x", 0, 42) == sample_seed("x", 0, 42)
    assert sample_seed("x", 0, 42) != sample_seed("x", 0, 43)


def test_grading_and_loop_diagnostics_never_relabel_a_cut_or_open_thinking_as_correct():
    sampling = Sampling(thinking=True)
    record = {"text": "reasoning</think> YES", "cut": False, "thinking_closed": True}
    assert grade(record, "yes", "exact", sampling)
    assert not grade({**record, "cut": True}, "yes", "exact", sampling)
    assert not grade({**record, "thinking_closed": False}, "yes", "exact", sampling)
    assert last_boxed("first \\boxed{1}, last \\boxed{\\frac{1}{2}}") == r"\frac{1}{2}"
    assert last_boxed(r"\boxed{unfinished") is None
    assert repetition("a" * 6000, True)[0]
    assert not repetition("a" * 6000, False)[0]
    assert not repetition("x", True)[0]
    assert math.isfinite(repetition("", True)[1])


def test_optional_math_verify_is_an_explicit_real_grader():
    pytest.importorskip("math_verify")
    record = {"text": r"Think. \boxed{\frac{1}{2}}", "cut": False, "thinking_closed": True}
    assert grade(record, "0.5", "boxed_math", Sampling())
    assert not grade({**record, "text": "0.5"}, "0.5", "boxed_math", Sampling())
    assert not grade(record, "2", "boxed_math", Sampling())


def test_reviewed_scripted_source_example_is_check_only_and_explicitly_labelled():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples/research_scripted.json"
    config = load_lab_config(path)
    assert config["editor"]["mode"] == "scripted" and len(config["actions"]) == 16
    assert lab_config_summary(config)["counts"] == {"train": 2, "validation": 4, "test": 2}
    assert config["actions"][-2:] == [{"action": "submit"}, {"action": "confirm"}]


def test_research_github_notice_excludes_paths_prompts_and_extra_nested_fields(monkeypatch, capsys):
    from scripts.run_research_smoke import github_evidence

    summary = {
        "parameters": 134515008,
        "editor_state_sha256": "a" * 64,
        "base_restored": True,
        "source_files_unchanged": True,
        "capability_gain_claimed": False,
        "model_authored": {
            "editor_generations": 2,
            "valid_tool_actions": 0,
            "invalid_actions": 2,
            "status": "budget_exhausted",
            "accepted": False,
            "private_prompt": "do not emit",
        },
        "scripted_tools": {
            "actions": 16,
            "forward_requests": 10,
            "changed_entries": 1152,
            "edited_tensors": 2,
            "status": "rejected",
            "accepted": False,
            "reserved_baseline_accuracy": 0.0,
            "reserved_candidate_accuracy": 0.0,
            "max_control_norm_rounding_error": 1e-8,
            "private_path": "/private/do-not-emit",
        },
        "actual_report": "/private/labels/lab.json",
        "scripted_report": "/private/labels/lab.json",
    }
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    evidence = github_evidence(summary)
    text = capsys.readouterr().out
    assert text.startswith("::notice title=Ouroboros research evidence::")
    assert "private" not in text and "prompt" not in text and "lab.json" not in text
    assert not evidence["capability_gain_claimed"] and evidence["scripted_tools"]["actions"] == 16
    assert "private_prompt" in summary["model_authored"]  # caller evidence was not mutated
