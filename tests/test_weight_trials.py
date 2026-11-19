import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ouroboros.core.checkpoints import file_sha256
from ouroboros.core.weight_trials import (
    WeightEditPlan,
    apply_weight_patch,
    run_weight_edit_demo,
    run_weight_edit_trial,
    verify_weight_patch,
)
from ouroboros.data import Sample

pytestmark = pytest.mark.ml


@pytest.fixture
def circuit():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2, bias=False)
    model.probe = torch.nn.Dropout(0.5)
    model.probe.eval()  # deliberately heterogeneous training flags must survive trials
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[-1.0, 0.0], [0.0, 1.0]]))
    features = {"repair": [1.0, -0.5], "retain": [1.0, 1.0]}

    def predict(prompts):
        with torch.inference_mode():
            result = model(torch.tensor([features[p] for p in prompts], dtype=model.weight.dtype))
        return [str(i) for i in result.argmax(dim=-1).tolist()]

    validation = [Sample("repair", "0", "repair"), Sample("retain", "1", "retention")]
    return model, predict, validation


def trial(circuit, root, predict=None, **plan):
    model, original_predict, validation = circuit
    config = {
        "edited_fraction": 0.25,
        "strength": 1.0,
        "max_relative_norm": 0.75,
        "forgetting_threshold": 0.0,
        **plan,
    }
    return run_weight_edit_trial(
        model,
        predict or original_predict,
        validation,
        [1.0, 0.0],
        WeightEditPlan("weight", **config),
        root,
        provenance="unit-test two-feature classifier",
        license_name="MIT",
    )


@pytest.mark.parametrize("dtype_name", ["float16", "float32", "float64", "bfloat16"])
def test_accepted_trial_restores_and_patch_applies_and_reverses_exactly(
    circuit, tmp_path, dtype_name
):
    import torch

    model, predict, _ = circuit
    model.to(dtype=getattr(torch, dtype_name))
    with torch.no_grad():
        model.weight[0, 1] = -0.0  # numeric equality alone would miss a broken patch fingerprint
    original = model.weight.detach().clone()
    result = trial(circuit, tmp_path)
    assert result["passed"] and result["model_restored"]
    assert torch.equal(original.view(torch.uint8), model.weight.detach().view(torch.uint8))
    assert model.training and not model.probe.training
    assert result["baseline"]["scores"] == {"repair": 0.0, "retention": 1.0}
    assert result["candidate"]["scores"] == {"repair": 1.0, "retention": 1.0}
    assert verify_weight_patch(result["patch"])["changed_entries"] == 1
    before = predict(["repair", "retain"])
    apply_weight_patch(model, result["patch"])
    assert predict(["repair", "retain"]) == ["0", "1"]
    with pytest.raises(ValueError, match="fingerprint"):
        apply_weight_patch(model, result["patch"])
    apply_weight_patch(model, result["patch"], reverse=True)
    assert torch.equal(original.view(torch.uint8), model.weight.detach().view(torch.uint8))
    assert predict(["repair", "retain"]) == before
    with pytest.raises(ValueError, match="fingerprint"):
        apply_weight_patch(model, result["patch"], reverse=True)


def test_float64_unedited_entries_do_not_get_rounded_to_float32(circuit, tmp_path):
    import torch

    model = circuit[0].double()
    with torch.no_grad():
        model.weight[1, 1] = 1.000000000123
    original = model.weight.detach().clone()
    result = trial(circuit, tmp_path)
    apply_weight_patch(model, result["patch"])
    assert model.weight[1, 1].item() == original[1, 1].item()
    apply_weight_patch(model, result["patch"], reverse=True)
    assert torch.equal(model.weight, original)


@pytest.mark.parametrize(
    "plan,reason",
    [
        ({"strength": 0}, "No representable"),
        ({"max_relative_norm": 0.1}, "norm budget"),
        ({"max_patch_bytes": 16}, "byte budget"),
    ],
)
def test_unacceptable_edit_budgets_never_export_or_leak_changes(circuit, tmp_path, plan, reason):
    import torch

    original = circuit[0].weight.detach().clone()
    result = trial(circuit, tmp_path, **plan)
    assert not result["passed"] and result["patch"] is None
    assert result["candidate"] is None and reason in result["reason"]
    assert torch.equal(original, circuit[0].weight)
    assert not list(tmp_path.glob("*/patch"))


def test_plateau_is_rejected_even_when_edit_is_numerically_valid(circuit, tmp_path):
    result = trial(circuit, tmp_path, predict=lambda prompts: ["1"] * len(prompts))
    assert result["status"] == "rejected" and result["patch"] is None
    assert result["candidate"] and not result["passed"]


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_inference_failure_or_cancellation_restores_model_and_records_failure(
    circuit, tmp_path, error
):
    import torch

    model, original_predict, _ = circuit
    original = model.weight.detach().clone()
    rng = torch.get_rng_state().clone()
    count = 0

    def predict(prompts):
        nonlocal count
        count += 1
        if count == 2:
            raise error("candidate interrupted")
        return original_predict(prompts)

    with pytest.raises(error):
        trial(circuit, tmp_path, predict=predict)
    assert torch.equal(model.weight, original) and torch.equal(torch.get_rng_state(), rng)
    (report_path,) = tmp_path.glob("*/trial.json")
    report = json.loads(report_path.read_text())
    assert report["model_restored"] and not report["passed"] and report["patch"] is None
    assert report["status"] == ("interrupted" if error is KeyboardInterrupt else "failed")


def test_torch_rng_is_paired_and_restored_for_read_only_callbacks(circuit, tmp_path):
    import torch

    rng = torch.get_rng_state().clone()
    draws = []

    def predict(prompts):
        draws.append(torch.rand(2).tolist())
        return circuit[1](prompts)

    result = trial(circuit, tmp_path, predict=predict)
    assert result["passed"] and draws[0] == draws[1]
    assert torch.equal(rng, torch.get_rng_state())


def test_export_failure_occurs_after_model_restoration(circuit, tmp_path):
    import torch

    original = circuit[0].weight.detach().clone()

    def fail(*args, **kwargs):
        assert torch.equal(circuit[0].weight, original)
        raise OSError("disk full")

    with patch("ouroboros.core.weight_trials.save_file", fail), pytest.raises(OSError):
        trial(circuit, tmp_path)
    assert torch.equal(circuit[0].weight, original)
    (path,) = tmp_path.glob("*/trial.json")
    assert json.loads(path.read_text())["status"] == "failed"
    assert not list(tmp_path.glob("*/patch"))


def test_patch_cannot_apply_to_a_different_target_tensor(circuit, tmp_path):
    import torch

    model = circuit[0]
    result = trial(circuit, tmp_path)
    with torch.no_grad():
        model.weight[1, 1] += 0.1
    modified = model.weight.detach().clone()
    with pytest.raises(ValueError, match="fingerprint"):
        apply_weight_patch(model, result["patch"])
    assert torch.equal(model.weight, modified)


@pytest.mark.parametrize(
    "field,match", [("before_bytes", "input tensor"), ("after_bytes", "fingerprint")]
)
def test_rehashed_corrupt_payload_still_cannot_modify_the_model(circuit, tmp_path, field, match):
    import torch
    from safetensors.numpy import load_file, save_file

    result = trial(circuit, tmp_path)
    directory = Path(result["patch"])
    file = directory / "delta.safetensors"
    arrays = load_file(str(file))
    arrays[field] = np.array([[3.0]], dtype=np.float32).view(np.uint8).reshape(1, 4)
    save_file(arrays, str(file))
    manifest = directory / "patch.json"
    meta = json.loads(manifest.read_text())
    meta["data_sha256"] = file_sha256(file)
    manifest.write_text(json.dumps(meta))
    original = circuit[0].weight.detach().clone()
    with pytest.raises(ValueError, match=match):
        apply_weight_patch(circuit[0], directory)
    assert torch.equal(circuit[0].weight, original)


def test_quantized_model_and_symlink_patch_fail_closed(circuit, tmp_path):
    result = trial(circuit, tmp_path)
    link = tmp_path / "link"
    link.symlink_to(result["patch"], target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        verify_weight_patch(link)
    circuit[0].is_loaded_in_4bit = True
    with pytest.raises(ValueError, match="Quantized"):
        apply_weight_patch(circuit[0], result["patch"])


def test_weight_edit_demo_is_explicit_real_fixture_not_llm(circuit, tmp_path):
    result = run_weight_edit_demo(tmp_path)
    assert result["passed"] and result["model_info"]["is_llm"] is False
    assert result["edit"]["edited_weights"] == 1 and result["model_restored"]


def test_weight_demo_cli_and_patch_evaluation_always_undo_edit(
    circuit, tmp_path, capsys, monkeypatch
):
    from types import SimpleNamespace

    import torch

    from ouroboros.cli import main

    assert main(["weights", "demo", "--output-dir", str(tmp_path / "demo")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] and result["model_info"]["is_llm"] is False
    model, predict, validation = circuit
    original = model.weight.detach().clone()
    # A unit-level loader stub tests CLI cleanup with an actual Torch parameter and valid real patch.
    monkeypatch.setattr(
        "ouroboros.backends.local_model.LocalCausalLM",
        lambda config: SimpleNamespace(model=model, predict=predict),
    )
    (tmp_path / "validation.json").write_text(json.dumps([s.as_dict() for s in validation]))
    (tmp_path / "direction.json").write_text("[1.0, 0.0]")
    config = {
        "model": {"model_name_or_path": "unused-local-model"},
        "edit": {"parameter_name": "weight"},
        "validation": "validation.json",
        "direction_file": "direction.json",
        "output_dir": "unused-output",
        "provenance": "unit fixture",
        "license": "MIT",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert (
        main(
            [
                "weights",
                "evaluate-patch",
                "--config",
                str(path),
                "--patch",
                result["patch"],
                "--data",
                str(tmp_path / "validation.json"),
            ]
        )
        == 0
    )
    response = json.loads(capsys.readouterr().out)
    assert response["patch_reverted"] and response["model_saved"] is False
    assert response["scores"] == {"repair": 1.0, "retention": 1.0}
    assert torch.equal(original, model.weight)


def test_application_rechecks_the_norm_policy_before_copying(circuit, tmp_path):
    import torch

    result = trial(circuit, tmp_path)
    manifest = Path(result["patch"]) / "patch.json"
    meta = json.loads(manifest.read_text())
    meta["max_relative_norm"] = 0.01
    manifest.write_text(json.dumps(meta))
    original = circuit[0].weight.detach().clone()
    with pytest.raises(ValueError, match="norm budget"):
        apply_weight_patch(circuit[0], result["patch"])
    assert torch.equal(original, circuit[0].weight)
