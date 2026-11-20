"""Plan/condition/evidence tests use an explicit stub runner, never pretrained inference."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_research_experiment as experiment


@pytest.mark.parametrize("failure", [None, "prompt", "baseline", "implementation"])
def test_fixed_comparison_checks_initial_prompts_baselines_and_implementation(
    tmp_path, monkeypatch, capsys, failure
):
    calls = []
    source_hashes = {"stub.py": "0" * 64}
    monkeypatch.setattr(experiment, "require_structured_decoder", lambda: None)
    monkeypatch.setattr(experiment, "verify_model_files", lambda path: {"verified": True})
    monkeypatch.setattr(experiment, "preflight_model", lambda config: {"fixture": True})
    monkeypatch.setattr(experiment, "implementation_hashes", lambda: dict(source_hashes))

    def stub_run(backend, dataset, policy, output_dir, *, editor, **kwargs):
        # Registration must exist BEFORE either condition. No response from condition one is
        # passed into condition two. This boundary test intentionally does no generation/edit.
        plan = json.loads((tmp_path / "project/interface_plan.json").read_text())
        assert plan["conditions_in_order"] == ["unconstrained", "json_schema"]
        assert plan["implementation_files_sha256"] == {"stub.py": "0" * 64}
        assert plan["editor"]["max_new_tokens"] == editor["max_new_tokens"]
        calls.append((dataset.fingerprint(), policy, editor, kwargs))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True)
        prompt = "fixed stub prompt"
        if failure == "prompt" and len(calls) == 2:
            prompt = "different stub prompt"
        event_path = output_dir / "turn.json"
        event_path.write_text(json.dumps({"editor_prompt": prompt}))
        report = {
            "editor_decoding": editor["decoding"],
            "model_authored_actions": False,  # explicit boundary stub, not an actual editor
            "base_restored": True,
            "source_files_unchanged": True,
            "editor_state_sha256": "1" * 64,
            "report_path": str(output_dir / "lab.json"),
            "selected_session": None,
            "final_test": {"used_for_promotion": False},
            "sessions": [
                {
                    "events": [
                        {
                            "artifact": "turn.json",
                            "action": "inventory",
                            "status": "ok",
                            "parsed_json": True,
                        }
                    ],
                    "editor_generations": 1,
                    "counters": {"invalid_actions": 0},
                    "status": "stalled",
                    "practice_baseline": {
                        "accuracy": 1.0 if failure == "baseline" and len(calls) == 2 else 0.5,
                        "cut_share": 0.0,
                    },
                    "practice_readiness": {"signal": "mixed_correct_and_incorrect"},
                }
            ],
        }
        if failure == "implementation" and len(calls) == 2:
            source_hashes["stub.py"] = "2" * 64
        return report

    class StubBackend:
        def describe(self):
            return {"parameters": 0}

    monkeypatch.setattr(experiment, "ResearchCausalLM", lambda config: StubBackend())
    monkeypatch.setattr(experiment, "run_lab", stub_run)
    args = [
        "--workdir",
        str(tmp_path / "project"),
        "--model-dir",
        str(tmp_path / "model"),
        "--compare-unconstrained",
        "--turns",
        "16",
        "--max-seconds",
        "600",
    ]
    messages = {
        "prompt": "initial editor prompts did not match",
        "baseline": "model/baseline equality failed",
        "implementation": "files changed during the run",
    }
    if failure:
        with pytest.raises(RuntimeError, match=messages[failure]):
            experiment.main(args)
        assert not (tmp_path / "project/experiment_summary.json").exists()
    else:
        assert experiment.main(args) == 0
        summary = json.loads((tmp_path / "project/experiment_summary.json").read_text())
        assert summary["matched_initial_editor_prompt"]
        assert (
            summary["initial_editor_prompt_sha256"]
            == hashlib.sha256(b"fixed stub prompt").hexdigest()
        )
        assert not summary["capability_gain_claimed"]
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        summary["host_path"] = "DO_NOT_PUBLISH_PATH"
        summary["conditions"][0]["raw_editor_reply"] = "DO_NOT_PUBLISH_REPLY"
        safe = experiment.emit_evidence(summary)
        printed = capsys.readouterr().out
        assert "::notice title=Ouroboros structured editor evidence::" in printed
        assert "DO_NOT_PUBLISH" not in printed and "DO_NOT_PUBLISH" not in json.dumps(safe)
    assert len(calls) == 2
    first, second = calls
    assert first[:2] == second[:2] and first[3] == second[3]
    assert {k: v for k, v in first[2].items() if k != "decoding"} == {
        k: v for k, v in second[2].items() if k != "decoding"
    }
