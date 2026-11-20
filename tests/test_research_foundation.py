"""Foundation safety/protocol tests. Scripted measurements below are NOT capability evidence."""

import json
import os
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from ouroboros.data import Sample
from ouroboros.lab.metrics import summarize
from ouroboros.research.contracts import (
    ROLES,
    Budget,
    Hypothesis,
    Proposal,
    Recipe,
    ResearchData,
    commands,
    identity,
    load_config,
)
from ouroboros.research.evaluation import (
    audit_rewards,
    choose_practice,
    diagnosis,
    gate,
    retention_gate,
)
from ouroboros.research.memory import ExperimentMemory
from ouroboros.research.project import init_foundation, verify_run
from ouroboros.research.provenance import write_once
from ouroboros.research.researcher import rank_recipes


def config_fixture(root):
    model = root / "model"
    model.mkdir()
    save_file({"a": np.zeros((2, 2), dtype=np.float32)}, str(model / "model.safetensors"))
    (model / "config.json").write_text("{}")
    result = init_foundation(root / "project", model)
    return Path(result["config"])


def evaluation(samples, correct=True, *, wrong_domain=None, cut=False):
    rows = []
    for sample in samples:
        is_correct = correct and sample.domain != wrong_domain and not cut
        rows.append(
            {
                "id": identity(sample.instruction),
                "sample": 0,
                "seed": 42,
                "domain": sample.domain,
                "text": sample.target if is_correct else "wrong",
                "new_tokens": 1,
                "correct": is_correct,
                "cut": cut,
                "thinking_closed": True,
                "repetitive": False,
                "tail_compression_ratio": 1.0,
            }
        )
    return {"records": rows, "summary": summarize(rows)}


def finish(memory, experiment, *, useful=False, family="lora_sft", context="context"):
    for state in (
        "observing",
        "diagnosing",
        "researching",
        "designed",
        "executing",
        "validating",
        "reserved",
        "final",
        "robustness",
        "retention",
    ):
        memory.transition(experiment, state)
    summary = {"full_gate_passed": useful}
    record = {
        "context": context,
        "family": family,
        "useful": useful,
        "cost": 2.0,
        "summary": summary,
    }
    memory.transition(
        experiment, "completed" if useful else "rejected", {"result_sha256": identity(record)}
    )
    memory.record_result(experiment, **record)


@pytest.mark.parametrize(
    "patch",
    [
        {"max_steps": True},
        {"max_steps": 0},
        {"family_size": 1},
        {"repeats": 1},
        {"max_rss_bytes": float("inf")},
        {"max_total_steps": -1},
        {"max_candidates": 5},
    ],
)
def test_fixed_budgets_cannot_be_relaxed_by_coercion(patch):
    with pytest.raises(ValueError):
        Budget(**patch)


def hypothesis():
    return Hypothesis(
        "practice failures",
        "unverified possibility",
        0.3,
        ["measured:1"],
        [],
        ["capacity rather than data"],
        ["a controlled intervention"],
        0.1,
        "No independent improvement disproves this candidate prediction.",
    )


def test_proposals_require_evidence_uncertainty_and_actual_types():
    proposal = Proposal(1, "experiment", "bounded", [asdict(hypothesis())], [asdict(Recipe())])
    proposal.validate_context(["measured:1"], Budget(), ["general"])
    with pytest.raises(ValueError, match="absent"):
        proposal.validate_context([], Budget(), ["general"])
    with pytest.raises(ValueError, match="step budget"):
        proposal.validate_context(["measured:1"], Budget(max_steps=1), ["general"])
    for value in [
        {**asdict(proposal), "python": "import os"},
        {**asdict(proposal), "schema_version": True},
        {**asdict(proposal), "action": "declare_improved"},
        {**asdict(proposal), "action": "abstain"},
    ]:
        with pytest.raises(ValueError):
            Proposal.parse(value)
    with pytest.raises(ValueError, match="Duplicate"):
        Proposal.parse('{"action":"abstain","action":"experiment"}')
    with pytest.raises(ValueError):
        replace(hypothesis(), confidence=float("nan"))
    with pytest.raises(ValueError, match="competing"):
        replace(hypothesis(), alternatives=[])
    with pytest.raises(ValueError):
        Recipe(optimizer="execute_code")
    with pytest.raises(ValueError):
        Recipe(target_modules=['__import__("os")'])
    with pytest.raises(ValueError):
        Recipe(family="data_reweight")


def test_normalized_disjoint_seven_role_data_and_verified_training():
    roles = {role: [Sample(role + " example", "yes", verified=True)] for role in ROLES}
    data = ResearchData(roles)
    assert len(data.fingerprint()) == 64
    roles["reserved"] = [Sample("TRAIN  EXAMPLE", "different", verified=True)]
    with pytest.raises(ValueError, match="contaminated"):
        ResearchData(roles)
    roles["reserved"] = [Sample("new hidden prompt", "yes", verified=True)]
    roles["train"] = [Sample("train example", "yes")]
    with pytest.raises(ValueError, match="verification"):
        ResearchData(roles)


def test_check_only_scaffold_no_torch_no_download_no_database(tmp_path, monkeypatch):
    path = config_fixture(tmp_path)
    from ouroboros.research.controller import run_research

    def forbidden(*args, **kwargs):
        raise AssertionError("check-only must not execute a worker")

    monkeypatch.setattr("ouroboros.research.controller.WorkerExecutor.run", forbidden)
    result = run_research(path, check_only=True)
    assert result["stored_parameters"] == 4 and result["alpha"] == 0.05 / 44
    assert result["checked_only"] and not result["loads_model"]
    assert not (path.parent / "experiments.sqlite3").exists()
    assert not (path.parent / "runs").exists()
    with pytest.raises(FileExistsError):
        init_foundation(path.parent, tmp_path / "model")
    raw = json.loads(path.read_text())
    raw["budget"]["family_size"] = 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_config(path)


def test_symlink_inputs_and_output_inside_model_are_rejected(tmp_path):
    path = config_fixture(tmp_path)
    original = path.parent / "data/train.json"
    source = path.parent / "data/original.json"
    original.rename(source)
    original.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        load_config(path)
    original.unlink()
    original.write_bytes(source.read_bytes())
    value = json.loads(path.read_text())
    value["output_dir"] = str(tmp_path / "model" / "outputs")
    path.write_text(json.dumps(value))
    from ouroboros.research.controller import run_research

    with pytest.raises(ValueError, match="immutable"):
        run_research(path, check_only=True)


def test_strict_paired_gates_no_loss_rescue_regression_or_seed_cherry_pick():
    samples = [Sample(str(i), "yes", "one" if i < 12 else "two") for i in range(24)]
    base = evaluation(samples, False)
    good, failed = evaluation(samples), evaluation(samples, False)
    passed = gate(base, [good, good], alpha=0.05 / 44, seed=17)
    assert passed["passed"]
    assert passed["problem_clustered"]["unit_of_analysis"].startswith("problem")
    assert not gate(base, [good, failed], alpha=0.05 / 44, seed=17)["passed"]
    assert not gate(good, [good, good], alpha=0.05 / 44, seed=17)["passed"]
    regression = evaluation(samples, wrong_domain="two")
    assert not gate(good, [regression, regression], alpha=0.05 / 44, seed=17)["passed"]
    assert choose_practice(base, [[good, failed], [good, good]]) == 1
    assert choose_practice(good, [[good, good]]) is None
    bad_summary = {**good, "summary": {**good["summary"], "accuracy": 0.123}}
    with pytest.raises(ValueError, match="disagrees"):
        gate(base, [bad_summary], alpha=0.05, seed=17)


def test_practice_loss_and_confidence_are_not_accuracy_evidence():
    samples = [Sample(str(i), "yes") for i in range(12)]
    base, plateau = evaluation(samples, False), evaluation(samples, False)
    plateau["nll"] = {"problem_mean_nll": 0.0001}
    assert not gate(base, [plateau], alpha=0.05 / 44, seed=17)["passed"]
    symptoms = diagnosis(evaluation(samples, cut=True))
    assert symptoms["symptoms"]["token_budget_exhausted"] == 12
    assert not symptoms["causal_claim"]


def test_retention_requires_actual_matched_controls_and_no_new_regression():
    samples = [Sample(str(i), "yes") for i in range(12)]
    bad, good = evaluation(samples, False), evaluation(samples)
    assert retention_gate(bad, [good, good], [bad, bad], [good, good], alpha=0.001, seed=17)[
        "passed"
    ]
    assert not retention_gate(bad, [good, good], [bad, bad], [bad, good], alpha=0.001, seed=17)[
        "passed"
    ]


def test_rule_verified_distillation_never_accepts_self_rating_or_format():
    rows = [
        {"text": "wrong but very confident", "cut": False, "thinking_closed": True, "score": 1},
        {"text": "Yes", "cut": False, "thinking_closed": True},
        {"text": "yes", "cut": True, "thinking_closed": True},
    ]
    audit = audit_rewards(rows, "yes")
    assert [r["correct"] for r in audit["records"]] == [False, True, False]
    assert abs(sum(audit["group_relative_advantages"])) < 1e-12
    assert not audit["uninformative_group"]
    equal = audit_rewards([rows[0]], "yes")
    assert equal["uninformative_group"] and equal["group_relative_advantages"] == [0]


def test_append_only_memory_hashes_claims_holdout_consumption_and_roles(tmp_path):
    memory = ExperimentMemory(tmp_path / "history.sqlite3")
    baseline = memory.register_baseline({"model_files": {"model": "a" * 64}})
    memory.register("one", {"baseline_id": baseline})
    memory.register_roles("one", {"train": ["public"], "reserved": ["private"]})
    memory.claim("one", "I improved myself", "model-written unverified claim")
    assert memory.posterior("context", "lora_sft")["observations"] == 0
    memory.consume_holdout("one", "reserved", ["private"])
    with pytest.raises(ValueError, match="already consumed"):
        memory.consume_holdout("one", "reserved", ["private"])
    with pytest.raises(ValueError, match="changed dataset role"):
        memory.register_roles("one", {"train": ["private"]})
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        memory.db.execute("UPDATE experiments SET plan=?", ("{}",))
    finish(memory, "one", useful=True)
    assert memory.verify()
    assert memory.search("full_gate", limit=1)[0]["useful"] == 1
    posterior = memory.posterior("context", "lora_sft")
    assert posterior["alpha"] == 2 and posterior["beta"] == 1
    memory.close()
    readonly = ExperimentMemory(tmp_path / "history.sqlite3", readonly=True)
    with pytest.raises(sqlite3.OperationalError):
        readonly.register_baseline({"write": "denied"})
    readonly.close()


def test_memory_state_machine_and_forged_result_do_not_update_strategy(tmp_path):
    memory = ExperimentMemory(tmp_path / "m.sqlite3")
    memory.register("one", {})
    with pytest.raises(ValueError, match="transition"):
        memory.transition("one", "completed")
    memory.transition("one", "failed")
    with pytest.raises(ValueError, match="bound"):
        memory.record_result(
            "one", context="x", family="lora_sft", useful=False, cost=1, summary={}
        )
    assert memory.posterior("x", "lora_sft")["observations"] == 0
    memory.close()


def test_tamper_detection_including_results(tmp_path):
    path = tmp_path / "m.sqlite3"
    memory = ExperimentMemory(path)
    memory.register("one", {})
    finish(memory, "one", useful=True)
    memory.db.execute("DROP TRIGGER no_results_UPDATE")
    memory.db.execute("UPDATE results SET useful=0")
    memory.close()
    with pytest.raises(ValueError, match="event binding"):
        ExperimentMemory(path, readonly=True)


def test_recover_refuses_live_or_terminal_controller_and_never_resumes(tmp_path, monkeypatch):
    path = tmp_path / "m.sqlite3"
    memory = ExperimentMemory(path)
    memory.register("one", {})
    with pytest.raises(ValueError, match="still alive"):
        memory.recover_interrupted("one")
    monkeypatch.setattr("ouroboros.research.memory.process_token", lambda pid: None)
    memory.recover_interrupted("one")
    assert memory.state("one") == "interrupted"
    with pytest.raises(ValueError, match="unfinished"):
        memory.recover_interrupted("one")
    assert memory.verify()
    memory.close()


def test_verified_outcomes_change_future_ranking_without_claiming_selection_gain(tmp_path):
    memory = ExperimentMemory(tmp_path / "m.sqlite3")
    recipes = [Recipe(family="lora_sft", steps=2), Recipe(family="full_sft", steps=2)]
    assert rank_recipes(recipes, memory, "context")[0]["proposal_index"] == 0
    for i in range(3):
        memory.register(str(i), {})
        finish(memory, str(i), useful=True, family="full_sft")
    ranked = rank_recipes(recipes, memory, "context")
    assert ranked[0]["proposal_index"] == 1
    assert "not a trained researcher" in ranked[0]["posterior"]["interpretation"]
    memory.close()


class ScriptedExecutor:
    """Protocol fixture, explicitly scripted, NOT a scientific executor or model gain."""

    mode = "pass"
    calls = []

    def __init__(
        self,
        config,
        bindings,
        files,
        software,
        root,
        run_id,
        deadline,
        *,
        auxiliary_model_files=None,
    ):
        self.config, self.bindings, self.root, self.run_id = config, bindings, root, run_id
        self.receipts, self.planned_steps = [], 0
        self.count = 0

    def run(
        self,
        operation,
        *,
        inputs=None,
        recipe=None,
        checkpoint=None,
        seed=None,
        extra=None,
        model=None,
    ):
        self.count += 1
        self.calls.append((operation, list((inputs or {}).keys()), bool(checkpoint)))
        if self.mode == "interrupt" and operation == "train":
            raise KeyboardInterrupt()
        if operation in ("train", "followup"):
            from ouroboros.core.checkpoints import CheckpointStore

            original = recipe or Recipe(**extra["original_recipe"])

            class Backend:
                def save(self, path):
                    (Path(path) / "fixture.txt").write_text(
                        "Not model weights; scripted controller fixture only"
                    )

                def describe(self):
                    return {"is_llm": False}

            store = CheckpointStore(self.root / f"fixture-{self.count}", self.run_id)
            name = store.save(Backend(), 1, scores={}, metadata={"recipe": asdict(original)})
            value = {"checkpoint": str(store.root / name), "steps": 1}
            self.planned_steps += 1
            return value if operation == "train" else {"training": value}
        if operation == "research":
            return {
                "valid": False,
                "error": "invalid fixture JSON",
                "researcher": {"base_model_files": {}},
                "fallback_used": False,
            }
        binding = next(iter(inputs.values()))
        samples = [Sample.from_dict(s) for s in json.loads(Path(binding["file"]).read_text())]
        value = evaluation(samples, bool(checkpoint) and self.mode != "plateau")
        if operation == "observe":
            return {
                "evaluation": value,
                "introspection": {
                    "description": {"parameters": 4},
                    "parameter_metadata": [
                        {"name": "model.layers.0.q_proj.weight"},
                        {"name": "model.layers.0.v_proj.weight"},
                    ],
                    "architecture": {"model_type": "scripted-fixture"},
                },
            }
        return value


@pytest.mark.parametrize(
    "mode,expected", [("pass", "completed"), ("plateau", "rejected"), ("interrupt", "interrupted")]
)
def test_complete_state_machine_and_rejection_recovery_paths(tmp_path, monkeypatch, mode, expected):
    path = config_fixture(tmp_path)
    from ouroboros.research.controller import run_research

    ScriptedExecutor.mode, ScriptedExecutor.calls = mode, []
    monkeypatch.setattr("ouroboros.research.controller.WorkerExecutor", ScriptedExecutor)
    report = run_research(path)
    assert report["state"] == expected and report["source_files_unchanged"]
    assert not report["autonomous_improvement_demonstrated"]
    assert report["full_gate_passed"] == (expected == "completed")
    assert verify_run(report["report_file"])["report_bound_to_database"]
    if expected == "completed":
        assert set(report["gates"]) == {
            "validation",
            "reserved",
            "final",
            "robustness",
            "retention",
        }
        assert report["promoted_checkpoints"]
        # Every post-followup score is produced by a separate evaluate operation.
        calls = ScriptedExecutor.calls
        assert all(
            calls[i + 1][0] == "evaluate" for i, call in enumerate(calls) if call[0] == "followup"
        )
    elif mode == "plateau":
        assert not report["gates"] and not report["promoted_checkpoints"]
    assert json.loads(Path(report["plan"]["file"]).read_text())["gates"]["alpha"] == 0.05 / 44
    # Tampering with the top-level report is detected via the authoritative DB binding.
    raw = Path(report["report_file"])
    raw.write_text(raw.read_text() + "\n")
    with pytest.raises(ValueError, match="authoritative"):
        verify_run(raw)


def test_invalid_local_researcher_never_falls_back_to_host_training(tmp_path, monkeypatch):
    path = config_fixture(tmp_path)
    value = json.loads(path.read_text())
    value["researcher"] = {"kind": "local_model", "model_path": str(tmp_path / "model")}
    path.write_text(json.dumps(value))
    from ouroboros.research.controller import run_research

    ScriptedExecutor.mode, ScriptedExecutor.calls = "pass", []
    monkeypatch.setattr("ouroboros.research.controller.WorkerExecutor", ScriptedExecutor)
    report = run_research(path)
    assert report["state"] == "abstained"
    assert report["invalid_model_proposal"]
    assert all(call[0] != "train" for call in ScriptedExecutor.calls)


def test_second_run_cannot_reopen_consumed_hidden_set(tmp_path, monkeypatch):
    path = config_fixture(tmp_path)
    from ouroboros.research.controller import run_research

    ScriptedExecutor.mode, ScriptedExecutor.calls = "pass", []
    monkeypatch.setattr("ouroboros.research.controller.WorkerExecutor", ScriptedExecutor)
    assert run_research(path)["state"] == "completed"
    second = run_research(path)
    assert second["state"] == "failed" and "already consumed" in second["reason"]
    assert not second["full_gate_passed"] and not second["promoted_checkpoints"]


def test_campaign_rejects_reused_hidden_data_before_execution(tmp_path, monkeypatch):
    path = config_fixture(tmp_path)
    from ouroboros.research.controller import run_campaign

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid campaign must not start workers")

    monkeypatch.setattr("ouroboros.research.controller.WorkerExecutor.run", forbidden)
    with pytest.raises(ValueError, match="fresh"):
        run_campaign([path, path], max_cycles=2)
    with pytest.raises(ValueError, match="1–8"):
        run_campaign([path], max_cycles=100)


def test_cli_commands_are_read_only_by_default_and_versioned(tmp_path, capsys):
    from ouroboros.cli import main

    assert main(["research", "tools"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == commands() and not result["arbitrary_python"]
    assert {"design_experiment", "rollback_candidate", "record_result"} <= result["commands"].keys()
    path = config_fixture(tmp_path)
    assert main(["research", "run", "--config", str(path), "--check-only"]) == 0
    capsys.readouterr()
    assert main(["research", "history", "--database", str(tmp_path / "missing.sqlite3")]) == 2
    capsys.readouterr()


def test_append_only_json_is_not_overwritten(tmp_path):
    write_once(tmp_path / "immutable.json", {"fixed": True})
    with pytest.raises(FileExistsError):
        write_once(tmp_path / "immutable.json", {"fixed": False})
    assert os.path.isfile(tmp_path / "immutable.json")


def test_deepseek_matrix_is_source_linked_and_does_not_claim_reproduction():
    from ouroboros.research.applicability import STATUSES, report

    root = Path(__file__).resolve().parents[1]
    result = report(root)
    assert not result["missing_local_evidence"]
    assert not result["source_reported_gains_reproduced_locally"]
    assert not result["review_is_exhaustive"]
    assert sum(result["counts"].values()) == len(result["entries"])
    assert set(result["counts"]) == set(STATUSES)
    assert result["counts"]["implemented"] and result["counts"]["deferred"]
    assert result["documented_repositories"] == 19
    assert result["repositories_reviewed"] == 19
    for entry in result["entries"]:
        assert entry["status"] in STATUSES
        assert entry["claims_not_made"] and entry["differences"]
        assert entry["repo"].startswith("deepseek-ai/")
    # The two deliberately refused substitutions must stay visible, not silently "done".
    by_id = {entry["id"]: entry for entry in result["entries"]}
    assert by_id["deepseekmath-grpo"]["status"] == "partial"
    assert by_id["esft-expert-specialization"]["status"] == "deferred"
    assert by_id["engram-conditional-memory"]["status"] == "deferred"


def test_every_reviewed_deepseek_repository_is_present_in_the_repo_writing():
    """The review must live in the repository's own research writing, not only in a matrix."""
    from ouroboros.research.applicability import ENTRIES

    root = Path(__file__).resolve().parents[1]
    catalogue = json.loads((root / "papers/research.json").read_text(encoding="utf-8"))["papers"]
    mapped = {paper["deepseek_repo"] for paper in catalogue if paper.get("deepseek_repo")}
    assert mapped == {entry["repo"] for entry in ENTRIES}
    assert len(mapped) == 19
    bibliography = (root / "papers/references.bib").read_text(encoding="utf-8")
    index = (root / "papers/PAPERS_INDEX.md").read_text(encoding="utf-8")
    for paper in catalogue:
        if not paper.get("deepseek_repo"):
            continue
        assert paper["url"].startswith("https://")
        assert (root / paper["local_path"]).exists()
        assert paper["reproduced_here"] is False
        assert f"@misc{{{paper['id']}," in bibliography
        assert paper["title"] in index
    # The paper that introduces GRPO must stay explicitly labelled as unimplemented RL.
    math = next(paper for paper in catalogue if paper["id"] == "deepseekmath")
    assert "GRPO is not implemented" in math["summary"]
