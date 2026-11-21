"""Multi-target arithmetic/transactions/exact patches; tiny fixtures, not research results."""

import json
from dataclasses import replace

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from ouroboros.core.checkpoints import file_sha256
from ouroboros.lab.instrumentation import weight_statistics
from ouroboros.lab.patches import apply_bundle, export_bundle, use_bundle, verify_bundle
from ouroboros.lab.program import build_program, norm_matched_control, validate_candidate
from ouroboros.lab.state import WorkingCopy, make_damage, temporary_weights
from ouroboros.self_edit import model_fingerprint

pytestmark = pytest.mark.ml


def set_entry(name="a.weight", index=0, value=1.0):
    return {"op": "set_entries", "name": name, "indices": [index], "values": [value]}


def test_atomic_program_has_vector_multi_matrix_signed_projection_and_lowrank(lab_copy):
    import torch

    before = model_fingerprint(lab_copy.model)
    operations = [
        set_entry(),
        {"op": "add", "name": "drift", "value": 0.05},
        {"op": "project", "name": "b.weight", "direction": [1, 0], "strength": 6, "columns": [1]},
    ]
    candidate, stats = build_program(lab_copy, operations)
    assert set(candidate) == {"a.weight", "b.weight", "drift"}
    assert stats["changed_entries"] == 4
    assert candidate["b.weight"][0, 1] == pytest.approx(-0.5)
    assert candidate["drift"].tolist() == pytest.approx([0.15, 0.25])
    assert model_fingerprint(lab_copy.model) == before and not lab_copy.candidate
    lab_copy.candidate = candidate
    with lab_copy.use():
        assert float(lab_copy.model.a.weight[0, 0]) == 1
    assert model_fingerprint(lab_copy.model) == before
    lowrank, _ = build_program(
        lab_copy,
        [{"op": "low_rank_add", "name": "b.weight", "left": [[0.1], [0.0]], "right": [[0.0, 0.2]]}],
        replace=True,
    )
    assert torch.allclose(
        lowrank["b.weight"], lab_copy.original("b.weight") + torch.tensor([[0.0, 0.02], [0.0, 0.0]])
    )


def test_program_failure_never_partially_commits_and_global_budget_is_actual(lab_copy):
    before = model_fingerprint(lab_copy.model)
    for operation in (
        [set_entry(), {"op": "exec", "name": "b.weight", "code": "not run"}],
        [set_entry(), set_entry("unknown.weight")],
        [{"op": "set_entries", "name": "a.weight", "indices": [0, 0], "values": [1, 2]}],
        [{"op": "scale_rows", "name": "a.weight", "indices": [0, 1], "factor": 0.5}],
    ):
        with pytest.raises(ValueError):
            build_program(lab_copy, operation)
        assert model_fingerprint(lab_copy.model) == before and not lab_copy.candidate
    tiny = WorkingCopy(lab_copy.model, replace(lab_copy.policy, max_edited_fraction=0.1))
    with pytest.raises(ValueError, match="Global"):
        build_program(tiny, [set_entry(), set_entry("b.weight", 1, 0.2)])
    too_small = WorkingCopy(lab_copy.model, replace(lab_copy.policy, max_relative_norm=0.01))
    with pytest.raises(ValueError, match="norm"):
        build_program(too_small, [set_entry()])
    unchanged, stats = build_program(lab_copy, [set_entry(value=-1.0)])
    assert unchanged == {} and stats["changed_entries"] == 0


def test_top_columns_exact_entry_limit_and_measured_direction_ids(lab_copy):
    lab_copy.directions["d0"] = np.array([1.0, 1.0]) / np.sqrt(2)
    candidate, stats = build_program(
        lab_copy,
        [
            {
                "op": "project",
                "name": "a.weight",
                "direction": "d0",
                "strength": 0.2,
                "top_columns": 1,
                "max_entries": 1,
            }
        ],
    )
    assert stats["changed_entries"] == 1
    assert len(stats["operations"][0]["columns"]) == 1
    assert set(candidate) == {"a.weight"}
    with pytest.raises(ValueError, match="Unknown measured"):
        build_program(
            lab_copy,
            [{"op": "project", "name": "a.weight", "direction": "missing", "strength": 0.5}],
        )


def test_transaction_validates_all_targets_before_mutation_and_restores_on_cancel(lab_copy):
    import torch

    before = model_fingerprint(lab_copy.model)
    modes = [m.training for m in lab_copy.model.modules()]
    candidate, _ = build_program(lab_copy, [set_entry(), set_entry("b.weight", 1, 0.2)])
    with pytest.raises(ValueError, match="shape/dtype"):
        with temporary_weights(lab_copy.model, {**candidate, "drift": torch.zeros(3)}):
            pytest.fail("Invalid overlay entered")
    assert model_fingerprint(lab_copy.model) == before
    with pytest.raises(KeyboardInterrupt):
        with temporary_weights(lab_copy.model, candidate):
            assert model_fingerprint(lab_copy.model) != before
            raise KeyboardInterrupt()
    assert model_fingerprint(lab_copy.model) == before
    assert [m.training for m in lab_copy.model.modules()] == modes


def test_protected_tied_embedding_alias_and_final_norm_cannot_be_edited(lab_backend, lab_policy):
    import torch

    model = lab_backend.model
    model.embed_tokens = torch.nn.Embedding(2, 2)
    model.alias = model.embed_tokens
    model.norm = torch.nn.LayerNorm(2)
    model.get_input_embeddings = lambda: model.embed_tokens
    copy = WorkingCopy(model, lab_policy)
    for name in ("embed_tokens.weight", "norm.weight"):
        with pytest.raises(ValueError, match="protected"):
            build_program(copy, [set_entry(name, 0, 0.2)])


def test_damage_original_tools_and_clip_never_read_clean_editor_values(lab_backend, lab_policy):
    before = model_fingerprint(lab_backend.model)
    spec = {
        "parameter_name": "a.weight",
        "first_row": 0,
        "rows": 1,
        "std_multiplier": 200,
        "seed": 5,
    }
    initial = make_damage(lab_backend.model, lab_policy, spec)
    copy = WorkingCopy(lab_backend.model, lab_policy, initial=initial)
    damaged = weight_statistics(copy, names=["a.weight"], values=4, weights="original")
    assert damaged["tensors"][0]["flat_prefix"] != [-1.0, 0.1, 0.2, 1.0]
    # A larger matrix gives the median-based repair a well-defined intact majority.
    import torch

    lab_backend.model.wide = torch.nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        lab_backend.model.wide.weight.copy_(torch.arange(1, 65).reshape(8, 8) / 100)
    spec.update(parameter_name="wide.weight")
    copy = WorkingCopy(
        lab_backend.model, lab_policy, initial=make_damage(lab_backend.model, lab_policy, spec)
    )
    candidate, stats = build_program(
        copy, [{"op": "clip_outliers", "name": "wide.weight", "factor": 15}]
    )
    assert stats["changed_entries"] > 0
    assert candidate["wide.weight"].abs().max() < copy.original("wide.weight").abs().max()
    assert (
        model_fingerprint(lab_backend.model) != before
    )  # Only the explicit test-added wide module changed.
    assert float(lab_backend.model.a.weight[0, 0]) == -1.0


def test_norm_matched_controls_are_reproducible_and_leave_candidate_untouched(lab_copy):
    import torch

    lab_copy.candidate, _ = build_program(
        lab_copy, [{"op": "project", "name": "a.weight", "direction": [1, 1], "strength": 0.1}]
    )
    saved = {name: value.clone() for name, value in lab_copy.candidate.items()}
    first, stats = norm_matched_control(lab_copy, seed=7)
    second, _ = norm_matched_control(lab_copy, seed=7)
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert all(torch.equal(saved[name], lab_copy.candidate[name]) for name in saved)
    assert stats["norm_matches"][0]["relative_rounding_error"] < 1e-5
    for name, value in first.items():
        original = lab_copy.original(name)
        assert torch.equal(value != original, saved[name] != original)


def _bundle(copy, tmp_path):
    copy.candidate, _ = build_program(
        copy,
        [set_entry(), set_entry("b.weight", 1, 0.2), {"op": "add", "name": "drift", "value": 0.05}],
    )
    return export_bundle(
        copy, tmp_path / "bundle", provenance="mechanical fixture", license_name="MIT"
    )


def test_full_model_bound_multi_tensor_bundle_roundtrip_and_dense_budget(lab_copy, tmp_path):
    import torch

    before = model_fingerprint(lab_copy.model)
    directory = _bundle(lab_copy, tmp_path)
    info = verify_bundle(directory)
    assert info["valid"] and len(info["tensors"]) == 3
    assert info["before_state_sha256"] == before
    with pytest.raises(ValueError, match="snapshot"):
        apply_bundle(lab_copy.model, directory, max_snapshot_elements=1)
    assert model_fingerprint(lab_copy.model) == before
    apply_bundle(lab_copy.model, directory)
    assert model_fingerprint(lab_copy.model) == info["after_state_sha256"]
    with pytest.raises(ValueError, match="Complete model"):
        apply_bundle(lab_copy.model, directory)
    apply_bundle(lab_copy.model, directory, reverse=True)
    assert model_fingerprint(lab_copy.model) == before
    with pytest.raises(RuntimeError):
        with use_bundle(lab_copy.model, directory):
            assert float(lab_copy.model.a.weight[0, 0]) == 1
            raise RuntimeError("grader failed")
    assert model_fingerprint(lab_copy.model) == before
    with torch.no_grad():
        lab_copy.model.scalar.add_(1)
    with pytest.raises(ValueError, match="Complete model"):
        apply_bundle(lab_copy.model, directory)
    assert float(lab_copy.model.a.weight[0, 0]) == -1


def test_bundle_rehashed_malformation_and_wrong_output_state_fail_without_partial_mutation(
    lab_copy, tmp_path
):
    from pathlib import Path

    directory = Path(_bundle(lab_copy, tmp_path))
    before = model_fingerprint(lab_copy.model)
    path = directory / "bundle.json"
    manifest = json.loads(path.read_text())
    manifest["after_state_sha256"] = "a" * 64
    path.write_text(json.dumps(manifest))
    assert verify_bundle(directory)["valid"]  # Integrity reader cannot reconstruct a full model.
    with pytest.raises(ValueError, match="output full-state"):
        apply_bundle(lab_copy.model, directory)
    assert model_fingerprint(lab_copy.model) == before
    data = directory / "changes.safetensors"
    arrays = load_file(data)
    arrays["t0_indices"] = np.array([99999], dtype=np.int64)
    save_file(arrays, str(data))
    manifest["data_sha256"] = file_sha256(data)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="indices"):
        verify_bundle(directory)
    assert model_fingerprint(lab_copy.model) == before


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_bundles_preserve_dtype_and_signed_zero_exactly(lab_backend, lab_policy, tmp_path, dtype):
    import torch

    lab_backend.model.to(dtype=getattr(torch, dtype))
    with torch.no_grad():
        lab_backend.model.a.weight[0, 1] = -0.0
    copy = WorkingCopy(lab_backend.model, lab_policy)
    before = model_fingerprint(copy.model)
    directory = _bundle(copy, tmp_path)
    apply_bundle(copy.model, directory)
    assert torch.signbit(copy.model.a.weight[0, 1])
    apply_bundle(copy.model, directory, reverse=True)
    assert model_fingerprint(copy.model) == before
    with pytest.raises(ValueError):
        validate_candidate(
            copy, {"drift": torch.full((2,), float("nan"), dtype=getattr(torch, dtype))}
        )


@pytest.mark.parametrize("direction", [[True, 1], ["1", "0"], [1e308, 1e308], [0, 0], [1j, 0]])
def test_numeric_program_directions_refuse_coercion_and_overflow(lab_copy, direction):
    before = model_fingerprint(lab_copy.model)
    with pytest.raises(ValueError):
        build_program(
            lab_copy,
            [{"op": "project", "name": "a.weight", "direction": direction, "strength": 0.1}],
        )
    assert not lab_copy.candidate and model_fingerprint(lab_copy.model) == before
