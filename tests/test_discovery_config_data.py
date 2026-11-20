"""Discovery plans/data/interface checks are useful without Torch, models or network."""

import json
import subprocess
import sys
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import pytest

from ouroboros.cli import main
from ouroboros.core.checkpoints import file_sha256
from ouroboros.data import DatasetBundle, Sample
from ouroboros.lab.action_schema import action_schema
from ouroboros.lab.config import input_fingerprint, lab_config_summary, load_lab_config
from ouroboros.lab.discovery_config import validate_discovery
from ouroboros.lab.discovery_data import DOMAINS, GSM8K_SOURCE, prepare_discovery_data
from ouroboros.lab.discovery_session import DiscoverySession
from ouroboros.lab.experiment import CONDITIONS, EFFECTIVE_ALPHA, load_data, policy_for
from ouroboros.lab.metrics import paired_loss_summary
from ouroboros.lab.tools import capabilities


def test_packaged_independently_selected_data_are_hash_bound_balanced_and_disjoint():
    root = Path(str(files("ouroboros.resources").joinpath("discovery")))
    dataset, roles, manifest = load_data(root)
    assert manifest["source"] == GSM8K_SOURCE
    assert manifest["counts"] == {
        "practice": 32,
        "development": 32,
        "reserved": 48,
        "final": 64,
        "retention": 64,
    }
    assert len(dataset.test) == 64 and len(dataset.train) == 32
    assert all({sample.domain for sample in values} == set(DOMAINS) for values in roles.values())
    assert sum(len(rows) for rows in manifest["gsm8k_selected_rows"].values()) == 60
    assert "not semantic or pretrained" in manifest["limitations"]
    spec = validate_discovery(
        {
            "development": roles["development"],
            "retention": roles["retention"],
            "followup_operations": [{"op": "add"}],
        },
        dataset,
        policy_for("135m"),
    )
    assert not spec["gradient_access"] and spec["measure_loss"]
    with pytest.raises(ValueError, match="disjoint"):
        validate_discovery({**spec, "development": dataset.train}, dataset, policy_for("135m"))


def test_discovery_refuses_weaker_gates_or_implicit_gradients():
    rows = [[Sample(f"{role} {d}", "x", d) for d in ("a", "b")] for role in range(5)]
    dataset = DatasetBundle(rows[0], rows[2], test=rows[3])
    spec = {"development": rows[1], "retention": rows[4], "followup_operations": [{"op": "add"}]}
    policy = policy_for("135m")
    assert policy.alpha == EFFECTIVE_ALPHA and policy.alpha < 0.05
    for updates, match in (
        ({"require_significance": False}, "significance"),
        ({"forgetting_threshold": 0.01}, "forgetting"),
        ({"max_sessions": 2}, "one independent"),
    ):
        with pytest.raises(ValueError, match=match):
            validate_discovery(spec, dataset, replace(policy, **updates))
    with pytest.raises(ValueError, match="boolean"):
        validate_discovery({**spec, "gradient_access": "yes"}, dataset, policy)
    with pytest.raises(ValueError, match="required"):
        validate_discovery({**spec, "measure_loss": False}, dataset, policy)
    assert policy_for("360m").max_model_parameters > 360_000_000
    assert policy_for("360m").alpha == policy.alpha
    assert policy_for("360m").max_relative_norm == policy.max_relative_norm


def test_discovery_schema_preserves_model_values_and_state_binds_hypotheses_and_operators():
    context = {
        "parameters": ["a.weight"],
        "unused_hypotheses": [],
        "experiments": [],
        "gradient_access": False,
    }

    def branches(context):
        return {
            row["properties"]["action"]["enum"][0]: row
            for row in action_schema(tools=DiscoverySession.tools, context=context)["anyOf"]
        }

    before = branches(context)
    assert "hypothesize" in before and "edit" not in before and "inspect_gradients" not in before
    after = branches(
        {**context, "unused_hypotheses": ["h0"], "experiments": ["e0"], "operation_scope": "sparse"}
    )
    assert after["edit"]["properties"]["hypothesis_id"]["enum"] == ["h0"]
    assert "hypothesis_id" in after["edit"]["required"]
    operations = after["edit"]["properties"]["operations"]["items"]["anyOf"]
    assert {row["properties"]["op"]["enum"][0] for row in operations} == {"set_entries", "add"}
    assert operations[0]["properties"]["values"]["items"] == {"type": "number"}
    sealed = action_schema(tools=DiscoverySession.tools, sealed=True)
    assert {row["properties"]["action"]["enum"][0] for row in sealed["anyOf"]} == {
        "confirm",
        "withdraw",
        "stop_and_explain",
    }
    assert (
        len(capabilities(discovery=True)["actions"]) == 25
    )  # legacy 19 - 2 implicit variant actions + 8


def test_prepare_data_selection_is_reproducible_not_performance_filtered(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "grade_school_math/data").mkdir(parents=True)
    for split in ("train", "test"):
        rows = [
            {
                "question": f"Independent {split} problem {i}: what is the number?",
                "answer": f"Unexposed rationale.\n#### {i}",
            }
            for i in range(100)
        ]
        (source / f"grade_school_math/data/{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows)
        )
    (source / "LICENSE").write_text(
        "Test-owned source fixture, not a replacement upstream license."
    )
    hashes = {name: file_sha256(source / name) for name in GSM8K_SOURCE["files"]}
    monkeypatch.setitem(GSM8K_SOURCE, "files", hashes)
    first = prepare_discovery_data(source, tmp_path / "one")
    second = prepare_discovery_data(source, tmp_path / "two")
    assert first == second
    assert (tmp_path / "one/final.json").read_bytes() == (tmp_path / "two/final.json").read_bytes()
    assert "Unexposed rationale" not in (tmp_path / "one/practice.json").read_text()
    with pytest.raises(FileExistsError):
        prepare_discovery_data(source, tmp_path / "one")
    (source / "LICENSE").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_discovery_data(source, tmp_path / "three")


def test_loss_delta_is_unclipped_nll_not_an_accuracy_or_promotion_score():
    before = [{"id": str(i), "domain": "d", "nll": 10.0, "tokens": 2} for i in range(2)]
    after = [{**row, "nll": 2.0} for row in before]
    measured = paired_loss_summary(before, after)
    assert measured["mean"] == -8 and measured["normal_95_interval"] == [-8, -8]
    assert not measured["used_for_promotion"]
    with pytest.raises(ValueError, match="mismatch"):
        paired_loss_summary(before, [{**row, "tokens": 3} for row in after])


def test_cli_full_experiment_check_only_and_catalogue_need_no_torch_or_model(tmp_path, capsys):
    root = tmp_path / "not-created"
    assert (
        main(
            ["lab", "experiment", "--model-dir", "missing", "--workdir", str(root), "--check-only"]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert not root.exists() and not result["model_loaded"]
    assert result["conditions"] == list(CONDITIONS) and result["seeds"] == [17, 42]
    assert main(["lab", "tools", "--discovery"]) == 0
    assert "hypothesize" in json.loads(capsys.readouterr().out)["actions"]
    code = "from ouroboros.cli import main; import sys; assert main(['lab','tools','--discovery']) == 0; assert main(['lab','experiment','--model-dir','absent','--workdir',sys.argv[1],'--check-only']) == 0; assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    subprocess.run([sys.executable, "-c", code, str(root)], capture_output=True, check=True)


def test_prepared_data_bytes_cannot_change_without_invalidating_approval(tmp_path):
    from shutil import copytree

    root = Path(str(files("ouroboros.resources").joinpath("discovery")))
    copytree(root, tmp_path / "data")
    path = tmp_path / "data/practice.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="bytes changed"):
        load_data(tmp_path / "data")


def test_discovery_config_and_hidden_roles_are_part_of_mcp_approval(tmp_path):
    from ouroboros.project import init_project

    root = tmp_path / "lab"
    result = init_project(root, discovery=True)
    path = Path(result["discovery_config"])
    config = load_lab_config(path, workspace=root)
    assert lab_config_summary(config)["discovery"]["counts"] == {"development": 32, "retention": 64}
    fingerprint = input_fingerprint(config)
    hidden = root / "discovery/retention.json"
    hidden.write_text(hidden.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        input_fingerprint(config)
    assert input_fingerprint(load_lab_config(path, workspace=root)) != fingerprint
    # Summary/approval never displays hidden questions or labels.
    summary = json.dumps(lab_config_summary(config))
    assert "<item>" not in summary and "Unexposed" not in summary
