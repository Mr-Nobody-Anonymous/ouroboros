"""Higher-rank flat edit/patch support and storage-alias boundaries; not model capability tests."""

import pytest

from ouroboros.lab.instrumentation import weight_statistics
from ouroboros.lab.patches import apply_bundle, export_bundle, verify_bundle
from ouroboros.lab.program import build_program, norm_matched_control
from ouroboros.lab.state import WorkingCopy, protected_names, temporary_weights
from ouroboros.self_edit import model_fingerprint

pytestmark = pytest.mark.ml


@pytest.mark.parametrize("shape", [(2, 2, 2), (2, 2, 2, 2)])
def test_higher_rank_flat_edits_controls_and_exact_bundle_roundtrip(
    lab_backend, lab_policy, tmp_path, shape
):
    import math

    import torch

    model = lab_backend.model
    model.kernel = torch.nn.Parameter(
        torch.arange(1, math.prod(shape) + 1, dtype=torch.float32).reshape(shape)
    )
    copy = WorkingCopy(model, lab_policy)
    before = model_fingerprint(model)
    copy.candidate, stats = build_program(
        copy, [{"op": "add", "name": "kernel", "indices": [1, 3], "value": 0.1}]
    )
    assert stats["changed_entries"] == 2 and tuple(copy.candidate["kernel"].shape) == shape
    assert weight_statistics(copy, names=["kernel"])["tensors"][0]["shape"] == list(shape)
    with copy.use():
        assert model_fingerprint(model) != before
    assert model_fingerprint(model) == before
    control, info = norm_matched_control(copy, kind="random_delta", seed=4)
    assert tuple(control["kernel"].shape) == shape
    assert info["norm_matches"][0]["relative_rounding_error"] < 1e-5
    with pytest.raises(ValueError, match="matrix"):
        norm_matched_control(copy, kind="random_direction")
    with pytest.raises(ValueError, match="matrix"):
        build_program(
            copy, [{"op": "project", "name": "kernel", "direction": [1, 1], "strength": 0.1}]
        )
    bundle = export_bundle(
        copy, tmp_path / "bundle", provenance="higher-rank mechanical test", license_name="MIT"
    )
    assert verify_bundle(bundle)["tensors"][0]["shape"] == list(shape)
    apply_bundle(model, bundle)
    assert torch.equal(model.kernel.detach().cpu(), copy.candidate["kernel"])
    apply_bundle(model, bundle, reverse=True)
    assert model_fingerprint(model) == before


def test_distinct_shared_storage_refused_before_any_mutation_and_protected_storage_propagates(
    lab_backend, lab_policy
):
    import torch

    model = lab_backend.model
    model.alias = torch.nn.Parameter(model.a.weight.detach().reshape(-1))
    copy = WorkingCopy(model, lab_policy)
    before = model_fingerprint(model)
    for name in ("a.weight", "alias"):
        with pytest.raises(ValueError, match="sharing storage"):
            build_program(copy, [{"op": "add", "name": name, "indices": [0], "value": 0.1}])
    with pytest.raises(ValueError, match="sharing storage"):
        with temporary_weights(
            model,
            {
                "b.weight": model.b.weight.detach().clone() + 0.01,
                "alias": model.alias.detach().clone(),
            },
        ):
            pytest.fail("shared-storage overlay entered")
    assert model_fingerprint(model) == before
    model.embed_tokens = torch.nn.Embedding(2, 2)
    model.embed_view = torch.nn.Parameter(model.embed_tokens.weight.detach().view(-1))
    assert "embed_view" in protected_names(model)
    with pytest.raises(ValueError, match="protected"):
        build_program(
            WorkingCopy(model, lab_policy),
            [{"op": "add", "name": "embed_view", "indices": [0], "value": 0.1}],
        )
