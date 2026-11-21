"""Research protocol fixtures: frozen editor, sealed checks, independent restarts and cleanup."""

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ouroboros.lab.config import Sampling
from ouroboros.lab.runner import run_lab
from ouroboros.lab.session import BudgetExceeded, LabSession
from ouroboros.lab.state import WorkingCopy
from ouroboros.self_edit import model_fingerprint

pytestmark = pytest.mark.ml


def edits():
    return {
        "action": "edit",
        "operations": [
            {"op": "set_entries", "name": "a.weight", "indices": [0], "values": [1]},
            {"op": "add", "name": "b.weight", "indices": [1], "value": 0.05},
        ],
    }


def session(backend, dataset, policy):
    return LabSession(
        backend, WorkingCopy(backend.model, policy), dataset, sampling=Sampling(), grader="exact"
    )


def execute(backend, dataset, policy, tmp_path, *, mode="scripted", actions=None):
    return run_lab(
        backend,
        dataset,
        policy,
        tmp_path,
        sampling=Sampling(),
        editor={"mode": mode, "max_new_tokens": 128},
        actions=actions,
        provenance="unit fixture with actual Torch classifier and scripted editor replies, NOT an LLM benchmark",
        license_name="MIT",
    )


def test_reserved_check_is_once_only_sealed_and_does_not_leak_raw_data(
    lab_backend, lab_dataset, lab_policy
):
    lab = session(lab_backend, lab_dataset, replace(lab_policy, max_invalid_actions=5))
    before = model_fingerprint(lab_backend.model)
    lab.act(edits())
    checked = lab.act({"action": "submit"})
    assert checked["sealed"] and checked["eligible_for_host_promotion"]
    assert lab.reserved_checks == 1
    text = json.dumps(checked)
    assert "RESERVED-QUESTION" not in text and "FINAL-QUESTION" not in text
    assert "records" not in text and "seed" not in checked
    for action in ({"action": "submit"}, {"action": "drop"}, edits()):
        with pytest.raises(ValueError, match="sealed"):
            lab.act(action)
    assert lab.act({"action": "confirm"})["decision"] == "accepted"
    assert model_fingerprint(lab_backend.model) == before


def test_independent_sessions_select_without_merging_and_final_test_runs_only_afterward(
    lab_backend, lab_dataset, lab_policy, tmp_path
):
    import torch

    before = model_fingerprint(lab_backend.model)
    modes = [module.training for module in lab_backend.model.modules()]
    torch_rng, python_rng, numpy_rng = (
        torch.get_rng_state().clone(),
        random.getstate(),
        np.random.get_state(),
    )
    actions = [
        edits(),
        {"action": "test", "ids": [0, 1]},
        {"action": "submit"},
        {"action": "confirm"},
    ]
    report = execute(
        lab_backend, lab_dataset, replace(lab_policy, max_sessions=2), tmp_path, actions=actions
    )
    assert [entry["status"] for entry in report["sessions"]] == ["accepted", "rejected"]
    assert report["selected_session"] == 1 and report["base_restored"]
    assert not report["model_authored_actions"]
    assert len({entry["starting_target_state_sha256"] for entry in report["sessions"]}) == 1
    assert all(entry["counters"]["reserved_checks"] == 1 for entry in report["sessions"])
    assert report["sessions"][0]["patch"] and not report["sessions"][1]["patch"]
    final = report["final_test"]
    assert final["status"] == "completed" and not final["used_for_promotion"]
    assert final["paired"]["baseline"]["accuracy"] == 0.5
    assert final["paired"]["candidate"]["accuracy"] == 1
    assert final["paired"]["wrong_to_right"] == 1
    final_indices = [
        i for i, request in enumerate(lab_backend.requests) if "FINAL-QUESTION" in request["prompt"]
    ]
    assert final_indices == list(range(len(lab_backend.requests) - 4, len(lab_backend.requests)))
    assert model_fingerprint(lab_backend.model) == before
    assert [module.training for module in lab_backend.model.modules()] == modes
    assert torch.equal(torch_rng, torch.get_rng_state()) and random.getstate() == python_rng
    assert np.array_equal(numpy_rng[1], np.random.get_state()[1])
    assert Path(report["evaluation_plan"]["path"]).is_file()


def test_control_sweep_saved_variants_are_paired_and_do_not_replace_current_edit(
    lab_backend, lab_dataset, lab_policy
):
    import torch

    lab = session(lab_backend, lab_dataset, lab_policy)
    lab.act(edits())
    candidate = {name: value.clone() for name, value in lab.copy.candidate.items()}
    before = model_fingerprint(lab_backend.model)
    noop = lab.act({"action": "control", "kind": "no_op", "ids": [0, 1]})
    assert noop["variants"]["control"]["paired"]["accuracy_delta"]["mean"] == 0
    lab.act({"action": "save", "name": "candidate1"})
    paired = lab.act({"action": "test", "ids": [0, 1], "saved": ["candidate1"]})
    assert set(paired["variants"]) == {"edited", "candidate1"}
    isolated = lab.act(
        {"action": "control", "kind": "layer_isolation", "ids": [0, 1], "names": ["a.weight"]}
    )
    assert isolated["variants"]["control"]["paired"]["candidate"]["accuracy"] == 1
    sweep = lab.act(
        {
            "action": "sweep",
            "ids": [0, 1],
            "variants": [
                {
                    "label": "weak",
                    "operations": [{"op": "add", "name": "a.weight", "indices": [0], "value": 0.1}],
                },
                {"label": "fix", "operations": edits()["operations"]},
            ],
        }
    )
    assert set(sweep["variants"]) == {"weak", "fix"}
    assert all(torch.equal(candidate[name], lab.copy.candidate[name]) for name in candidate)
    assert model_fingerprint(lab_backend.model) == before
    for result in paired["variants"].values():
        assert [row["seed"] for row in paired["baseline_records"]] == [
            row["seed"] for row in result["records"]
        ]


def test_confirmation_cannot_promote_plateau_or_override_significance(
    lab_backend, lab_dataset, lab_policy
):
    lab = session(lab_backend, lab_dataset, replace(lab_policy, require_significance=True))
    lab.act(edits())
    assert not lab.act({"action": "submit"})["eligible_for_host_promotion"]
    assert lab.act({"action": "confirm"})["decision"] == "rejected"
    lab = session(lab_backend, lab_dataset, lab_policy)
    lab.act(
        {
            "action": "edit",
            "operations": [{"op": "add", "name": "b.weight", "indices": [1], "value": 0.01}],
        }
    )
    assert not lab.act({"action": "submit"})["eligible_for_host_promotion"]
    assert lab.act({"action": "confirm"})["decision"] == "rejected"
    with pytest.raises(ValueError, match="ended"):
        lab.act({"action": "withdraw"})


def test_model_reply_path_checks_frozen_editor_and_never_shows_held_out_or_damage_spec(
    lab_backend, lab_dataset, lab_policy, tmp_path
):
    lab_backend.editor_actions = [
        edits(),
        {"action": "test", "ids": [0, 1]},
        {"action": "submit"},
        {"action": "confirm"},
    ]
    report = execute(lab_backend, lab_dataset, lab_policy, tmp_path, mode="model")
    checks = report["sessions"][0]["editor_hash_checks"]
    assert checks == [report["editor_state_sha256"]] * 4
    assert report["sessions"][0]["status"] == "accepted"
    assert all(
        "RESERVED-QUESTION" not in prompt and "FINAL-QUESTION" not in prompt
        for prompt in lab_backend.editor_prompts
    )
    assert len(lab_backend.editor_prompts) == 4
    # The editor can use bounded raw PRACTICE answers from its own requested forward test.
    after_practice = json.loads(lab_backend.editor_prompts[2])
    practice_result = after_practice["recent_conversation"][-1]["observation"]
    assert practice_result["baseline_records"][0]["text"]
    assert practice_result["variants"]["edited"]["records"][0]["text"]
    # This is a backend-stub flow test. Metadata must not mislabel the classifier as an LLM.
    assert not report["model"]["is_llm"]


def test_invalid_json_stops_without_forcing_any_edit_and_withdrawal_does_not_export(
    lab_backend, lab_dataset, lab_policy, tmp_path
):
    lab_backend.invalid_reply = True
    report = execute(
        lab_backend, lab_dataset, replace(lab_policy, max_invalid_actions=2), tmp_path, mode="model"
    )
    assert len(report["sessions"][0]["events"]) == 2
    assert report["sessions"][0]["status"] == "budget_exhausted"
    assert report["sessions"][0]["counters"]["invalid_actions"] == 2
    assert report["selected_session"] is None and report["base_restored"]
    assert report["final_test"]["candidate_is_baseline"]
    assert not list(tmp_path.rglob("changes.safetensors"))
    lab_backend.invalid_reply = False
    report = execute(
        lab_backend,
        lab_dataset,
        lab_policy,
        tmp_path / "withdraw",
        actions=[edits(), {"action": "submit"}, {"action": "withdraw"}],
    )
    assert report["sessions"][0]["status"] == "withdrawn" and not report["sessions"][0]["patch"]


@pytest.mark.parametrize(
    "failure,exception",
    [("candidate", RuntimeError), ("interrupt", KeyboardInterrupt), ("final", RuntimeError)],
)
def test_failed_or_interrupted_inference_always_restores_and_reports_honestly(
    lab_backend, lab_dataset, lab_policy, tmp_path, failure, exception
):
    before = model_fingerprint(lab_backend.model)
    lab_backend.failure = failure
    with pytest.raises(exception):
        execute(
            lab_backend,
            lab_dataset,
            lab_policy,
            tmp_path,
            actions=[
                edits(),
                {"action": "test", "ids": [0, 1]},
                {"action": "submit"},
                {"action": "confirm"},
            ],
        )
    report = json.loads(next(tmp_path.glob("*/lab.json")).read_text())
    assert report["base_restored"] and model_fingerprint(lab_backend.model) == before
    assert report["status"] == ("interrupted" if exception is KeyboardInterrupt else "failed")
    assert report["final_test"]["status"] == (
        "failed" if failure == "final" else "skipped_after_failure"
    )


def test_forward_edit_and_turn_limits_are_enforced_without_a_model_change(
    lab_backend, lab_dataset, lab_policy
):
    lab = session(lab_backend, lab_dataset, replace(lab_policy, max_forward_calls=1))
    lab.act({"action": "test", "ids": [0]})
    with pytest.raises(BudgetExceeded, match="forward"):
        lab.act({"action": "test", "ids": [0]})
    lab = session(lab_backend, lab_dataset, replace(lab_policy, max_edit_actions=1))
    lab.act(edits())
    with pytest.raises(BudgetExceeded, match="Edit-action"):
        lab.act(edits())
    with pytest.raises(ValueError, match="PRACTICE"):
        session(lab_backend, lab_dataset, lab_policy).act({"action": "test", "ids": [-1]})
    with pytest.raises(ValueError, match="Unexpected"):
        session(lab_backend, lab_dataset, lab_policy).act({"action": "weights", "file": ".env"})
