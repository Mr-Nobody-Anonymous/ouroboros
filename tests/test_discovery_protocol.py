"""Real tiny neural token classifier controls, explicitly NOT pretrained/autonomous LLM evidence."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.config import ModelConfig
from ouroboros.data import DatasetBundle, Sample
from ouroboros.lab.backend import ResearchCausalLM, frozen_eval
from ouroboros.lab.config import Sampling
from ouroboros.lab.discovery_config import validate_discovery
from ouroboros.lab.discovery_session import DiscoverySession
from ouroboros.lab.experiment import policy_for
from ouroboros.lab.journal import compare_candidates, verify_roundtrip
from ouroboros.lab.runner import run_lab
from ouroboros.lab.sensitivity import completion_loss, inspect_gradients, measure_loss
from ouroboros.lab.state import WorkingCopy
from ouroboros.self_edit import model_fingerprint

pytestmark = pytest.mark.ml


@pytest.fixture
def token_classifier():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = ["[PAD]", "[UNK]", "[EOS]", "warm", "cool", "yes", "no", "Instruction", "Response", ":"]
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )

    class CausalClassifier(torch.nn.Module):
        """Hand-designed two-feature causal classifier + fixed EOS gate, not a language model."""

        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(len(words), 2)
            self.bridge = torch.nn.Linear(2, 2, bias=False)
            self.head = torch.nn.Linear(2, len(words), bias=False)
            self.config = SimpleNamespace(vocab_size=len(words))
            with torch.no_grad():
                self.embedding.weight.zero_()
                self.embedding.weight[3, 0] = 1
                self.embedding.weight[4, 1] = 1
                self.bridge.weight.copy_(torch.tensor([[-0.001, 0.0001], [0, 0.1]]))
                self.head.weight.zero_()
                self.head.weight[5, 0] = 1
                self.head.weight[6, 1] = 1
            self.calls = 0
            self.requires_grad_(False)

        def get_input_embeddings(self):
            return self.embedding

        def get_output_embeddings(self):
            return self.head

        def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=None):
            self.calls += 1
            feature = self.embedding(input_ids).cumsum(1)
            hidden = self.bridge(feature)
            logits = self.head(hidden)
            bias = torch.full((len(words),), -20.0, device=logits.device)
            bias[5:7] = 0
            logits = logits + bias
            phase = ((input_ids == 5) | (input_ids == 6)).cumsum(1)
            logits[:, :, 2] += phase * 100
            return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)

        def generate(self, input_ids, max_new_tokens, **kwargs):
            result = input_ids
            for _ in range(max_new_tokens):
                token = self(input_ids=result).logits[:, -1].argmax(-1, keepdim=True)
                result = torch.cat((result, token), dim=1)
                if int(token[0, 0]) == 2:
                    break
            return result

    class Backend(ResearchCausalLM):
        def __init__(self):
            self.model, self.tokenizer, self.device = CausalClassifier(), fast, torch.device("cpu")
            self.config = ModelConfig(max_new_tokens=4)
            self.context_size = 4096
            self.generation_tokens = {"eos_token_ids": [2]}

        def describe(self):
            return {
                "backend": "two-feature-causal-token-classifier-test-fixture",
                "is_llm": False,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "model_config": asdict(self.config),
            }

    return Backend()


def setup_session(backend, *, gradient=True, feedback=True, development_wrong=False, cache=None):
    roles = {}
    for role in ("practice", "development", "reserved", "final", "retention"):
        roles[role] = [
            Sample(
                f"{role}-{i} {word}",
                "no" if development_wrong and role == "development" else target,
                word,
            )
            for i in range(16)
            for word, target in (("warm", "yes"), ("cool", "no"))
        ]
    dataset = DatasetBundle(roles["practice"], roles["reserved"], test=roles["final"])
    policy = replace(
        policy_for("135m"),
        max_edited_fraction=0.1,
        max_turns=32,
        max_invalid_actions=10,
        cpu_threads=1,
    )
    spec = validate_discovery(
        {
            "development": roles["development"],
            "retention": roles["retention"],
            "gradient_access": gradient,
            "feedback": feedback,
            "followup_operations": [
                {"op": "add", "name": "bridge.weight", "indices": [3], "value": 0.00001}
            ],
        },
        dataset,
        policy,
    )
    session = DiscoverySession(
        backend,
        WorkingCopy(backend.model, policy),
        dataset,
        sampling=Sampling(),
        grader="exact",
        discovery=spec,
        measurement_cache=cache,
    )
    return session, dataset, policy, spec


def hypothesis(**updates):
    return {
        "action": "hypothesize",
        "problem": "Warm practice answers have the wrong class.",
        "targets": ["bridge.weight"],
        "reason": "HOST fixture control: reverse the small wrong feature coefficient while preserving the other category.",
        "ids": [0, 1],
        "metric": "accuracy",
        "expected_delta": 0.5,
        "falsification": "No practice accuracy increase; independent hidden accuracy gates are unchanged.",
        **updates,
    }


def edit(identifier="h0", value=0.001):
    return {
        "action": "edit",
        "hypothesis_id": identifier,
        "operations": [
            {"op": "set_entries", "name": "bridge.weight", "indices": [0], "values": [value]}
        ],
    }


def test_loss_and_opt_in_autograd_are_real_no_updates_even_inside_inference_mode(token_classifier):
    import torch

    backend = token_classifier
    session, _, policy, _ = setup_session(backend)
    before = model_fingerprint(backend.model)
    backend.model.train()
    modes = [m.training for m in backend.model.modules()]
    flags = [p.requires_grad for p in backend.model.parameters()]
    rng = torch.get_rng_state().clone()
    with frozen_eval(backend.model):
        result = session.act(
            {"action": "inspect_gradients", "name": "bridge.weight", "ids": [0, 1], "limit": 4}
        )
    assert result["gradient_norm"] > 0 and result["updates_performed"] == 0
    assert len(result["entries"]) == 4 and session.gradient_evaluated_problems == 2
    assert [m.training for m in backend.model.modules()] == modes
    assert [p.requires_grad for p in backend.model.parameters()] == flags
    assert all(p.grad is None for p in backend.model.parameters())
    assert torch.equal(rng, torch.get_rng_state()) and model_fingerprint(backend.model) == before
    measured = measure_loss(backend, session.dataset.train[:2], policy)
    assert measured["completion_tokens"] == 4  # one reference token AND one model EOS per problem
    assert measured["problem_mean_nll"] > 0
    with pytest.raises(ValueError, match="protected"):
        inspect_gradients(backend, session.copy, session.dataset.train[:2], "head.weight")
    blocked, *_ = setup_session(backend, gradient=False)
    with pytest.raises(ValueError, match="disabled"):
        blocked.act({"action": "inspect_gradients", "name": "bridge.weight", "ids": [0]})
    assert blocked.forward_calls == 0


def test_loss_refuses_context_and_logits_before_forward_and_preserves_gradient_flags(
    token_classifier,
):
    backend = token_classifier
    session, _, policy, _ = setup_session(backend)
    sample = session.dataset.train[0]
    calls = backend.model.calls
    with pytest.raises(ValueError, match="no truncation"):
        completion_loss(backend, sample, replace(policy, max_read_tokens=2))
    with pytest.raises(ValueError, match="logits"):
        completion_loss(backend, sample, replace(policy, max_logit_elements=1))
    assert backend.model.calls == calls
    before = [p.requires_grad for p in backend.model.parameters()]
    session.copy.policy = replace(policy, max_read_tokens=2)
    with pytest.raises(ValueError):
        inspect_gradients(backend, session.copy, [sample], "bridge.weight")
    assert [p.requires_grad for p in backend.model.parameters()] == before


def test_every_attempt_including_missing_hypothesis_and_reversals_is_audited(token_classifier):
    session, *_ = setup_session(token_classifier)
    before = model_fingerprint(token_classifier.model)
    missing = edit()
    missing.pop("hypothesis_id")
    with pytest.raises(ValueError, match="missing"):
        session.act(missing)
    failed = session.experiments[-1]
    assert not failed["validation_result"]["passed"] and failed["changes"]["changed_entries"] == 0
    assert failed["operation"] == missing
    session.act(hypothesis())
    measured = session.act(edit())
    trial = session.experiments[-1]
    assert measured["paired_practice"]["baseline"]["accuracy"] == 0.5
    assert measured["paired_practice"]["candidate"]["accuracy"] == 1
    assert measured["outcome"]["prediction_supported"] and trial["roundtrip_replay"]["passed"]
    tensor = trial["changes"]["tensors"][0]
    assert tensor["indices"] == [0] and len(tensor["before_values"]) == 1
    assert trial["original_checkpoint_sha256"] == before
    assert (
        trial["restored_to_resident_base"] and model_fingerprint(token_classifier.model) == before
    )
    assert session.act({"action": "compare_checkpoints"})["changed_entries"] == 1
    session.act({"action": "rollback", "to": "previous"})
    assert not session.copy.candidate and session.experiments[-1]["actual_nonempty_change"]
    assert model_fingerprint(token_classifier.model) == before


def test_invalid_numeric_edit_is_atomic_and_roundtrip_detects_tampering(token_classifier):
    session, *_ = setup_session(token_classifier)
    session.act(hypothesis())
    with pytest.raises(ValueError, match="norm"):
        session.act(edit(value=100))
    assert not session.copy.candidate and not session.experiments[-1]["validation_result"]["passed"]
    assert session.experiments[-1]["operation"]["operations"][0]["values"] == [100]
    session.act(hypothesis())
    session.act(edit("h1"))
    changes = compare_candidates(session.copy, {}, session.copy.candidate, full=True)
    changes["tensors"][0]["after_values"][0] += 0.001
    with pytest.raises(ValueError, match="replay"):
        verify_roundtrip(session.copy, {}, session.copy.candidate, changes)


def test_feedback_ablation_hides_outcomes_and_forbids_candidate_queries_but_archives_them(
    token_classifier,
):
    session, *_ = setup_session(token_classifier, feedback=False)
    session.act(hypothesis())
    measured = session.act(edit())
    visible = session.editor_observation(measured)
    assert "paired_practice" not in visible and "outcome" not in visible
    assert session.experiments[-1]["practice_evaluation"]["paired"]["accuracy_delta"]["mean"] == 0.5
    with pytest.raises(ValueError, match="feedback-off"):
        session.act({"action": "evaluate_candidate", "ids": [0]})
    with pytest.raises(ValueError, match="feedback-off"):
        session.act({"action": "test", "ids": [0], "weights": "edited"})
    original = session.act({"action": "test", "ids": [0], "weights": "original"})
    assert original["summary"]["accuracy"] == 0


def test_failed_development_look_is_sealed_and_cannot_become_adaptive_practice(token_classifier):
    session, *_ = setup_session(token_classifier, development_wrong=True)
    session.act(hypothesis())
    session.act(edit())
    result = session.act({"action": "submit"})
    assert result["sealed"] and not result["eligible_for_host_promotion"]
    assert result["reserved_check"] == "not_opened" and session.reserved_checks == 0
    for action in (edit("h1"), {"action": "rollback"}, hypothesis(), {"action": "practice"}):
        with pytest.raises(ValueError, match="sealed"):
            session.act(action)
    assert session.act({"action": "confirm"})["decision"] == "rejected"
    assert "development-" not in json.dumps(result) and "records" not in json.dumps(result)


def test_host_positive_control_reaches_strict_reserved_final_and_subsequent_retention(
    token_classifier, tmp_path
):
    _, dataset, policy, spec = setup_session(token_classifier)
    report = run_lab(
        token_classifier,
        dataset,
        policy,
        tmp_path,
        sampling=Sampling(),
        editor={"mode": "scripted"},
        actions=[hypothesis(), edit(), {"action": "submit"}, {"action": "confirm"}],
        discovery=spec,
        provenance="Explicit HOST-designed two-feature token-classifier positive control, NOT an LLM/autonomous gain.",
        license_name="MIT",
    )
    assert report["sessions"][0]["status"] == "accepted"
    assert report["sessions"][0]["counters"]["reserved_checks"] == 1
    assert report["final_test"]["paired"]["baseline"]["accuracy"] == 0.5
    assert report["final_test"]["paired"]["candidate"]["accuracy"] == 1
    assert report["final_test"]["generalization_assessment"]["strict_independent_gain"]
    assert report["retention"]["retention_of_validated_useful_edit"]
    assert report["retention"]["category_retention"]["passed"]
    assert (
        report["base_restored"]
        and not report["model_authored_actions"]
        and not report["model"]["is_llm"]
    )
    journal = json.loads(Path(report["sessions"][0]["edit_journal"]["path"]).read_text())
    assert journal["experiments"][0]["reserved_result"]
    assert journal["experiments"][0]["roundtrip_replay"]["passed"]
    from ouroboros.lab.experiment import summarize_run

    summary = summarize_run({**report, "source": None}, "human_directed", 42)
    assert summary["accepted"] and not summary["milestones"]["validated_autonomous_useful_edit"]


def test_cache_is_state_sample_seed_bound_and_never_mutates_a_shared_measurement(token_classifier):
    cache = {}
    session, *_ = setup_session(token_classifier, cache=cache)
    samples = session.dataset.train[:2]
    first = session.measure(samples, {})
    calls, sequences = token_classifier.model.calls, session.generated_sequences
    again = session.measure(samples, {})
    assert again["cache_receipt"]["reused"] and token_classifier.model.calls == calls
    assert session.generated_sequences == sequences and session.cache_hits == 1
    again["records"][0]["text"] = "tampered"
    assert session.measure(samples, {})["records"][0]["text"] == first["records"][0]["text"]
    session.measure(samples, {}, seed=43)
    assert token_classifier.model.calls > calls
    session.act(hypothesis())
    session.act(edit())
    altered = session.measure(samples, session.copy.candidate)
    assert altered["summary"]["accuracy"] == 1
    assert altered["cache_receipt"]["key"] != first["cache_receipt"]["key"]


def test_directional_predictions_and_effect_classification_are_scored_not_assumed(token_classifier):
    session, *_ = setup_session(token_classifier)
    session.act(hypothesis(metric="nll", expected_delta=-0.00001))
    session.act(edit())
    result = session.act(
        {
            "action": "analyze_failure",
            "experiment_id": "e0",
            "assessment": "harmful",
            "evidence": "An intentionally incorrect HOST fixture classification.",
            "revision": "Check real evidence rather than trust my claim.",
        }
    )
    assert result["actual_outcome"]["effect"] == "helpful"
    adaptive = session.journal()["adaptive_summary"]
    assert adaptive["harm_classifications"] == 1 and adaptive["correct_harm_classifications"] == 0
    assert not adaptive["improved_future_editing_ability_demonstrated"]


def test_noop_does_not_become_a_valid_nonempty_edit(token_classifier):
    session, *_ = setup_session(token_classifier)
    value = float(session.copy.original("bridge.weight").reshape(-1)[0])
    session.act(hypothesis())
    session.act(edit(value=value))
    assert not session.experiments[-1]["actual_nonempty_change"]
    assert session.experiments[-1]["changes"]["delta_norm"] == 0


def test_staged_arm_changes_workflow_not_numeric_values_or_acceptance(token_classifier):
    session, *_ = setup_session(token_classifier)
    session.discovery["workflow"] = "staged"
    assert "inventory" not in session.available_actions()
    with pytest.raises(ValueError, match="host-staged"):
        session.act({"action": "inventory"})
    session.act(hypothesis())
    assert (
        "edit" in session.available_actions() and "hypothesize" not in session.available_actions()
    )
    session.act(edit())
    assert (
        "hypothesize" in session.available_actions()
        and "analyze_failure" in session.available_actions()
    )
    assert "submit" in session.available_actions()
    assert session.act({"action": "submit"})["eligible_for_host_promotion"]
    assert session.available_actions() == ["confirm", "stop_and_explain", "withdraw"]
    assert session.act({"action": "confirm"})["decision"] == "accepted"


def test_sensitivity_alignment_is_measured_and_implicit_control_edits_are_unavailable(
    token_classifier,
):
    session, *_ = setup_session(token_classifier)
    session.act({"action": "inspect_gradients", "name": "bridge.weight", "ids": [0, 1], "limit": 4})
    session.act(hypothesis())
    session.act(edit())
    diagnostic = session.experiments[-1]["selection_diagnostics"][0]
    assert (
        diagnostic["matched_practice_sensitivity"] and diagnostic["inspected_changed_entries"] == 1
    )
    assert diagnostic["partial_first_order_nll_delta"] < 0
    with pytest.raises(ValueError, match="Unknown"):
        session.act({"action": "control", "kind": "random_delta", "ids": [0]})


def test_actual_host_and_matched_random_policies_use_the_same_audited_tools(
    token_classifier, tmp_path, monkeypatch, capsys
):
    from ouroboros.lab.controls import HostControl, random_references
    from ouroboros.lab.experiment import github_evidence, summarize_run

    _, dataset, policy, spec = setup_session(token_classifier)
    kwargs = dict(
        sampling=Sampling(),
        editor={"mode": "scripted"},
        discovery=spec,
        provenance="HOST-authored fixture controllers; not an autonomous LLM.",
        license_name="MIT",
        measurement_cache={},
    )
    human = run_lab(
        token_classifier,
        dataset,
        policy,
        tmp_path / "human",
        controller=HostControl("bridge.weight"),
        **kwargs,
    )
    journal = json.loads(Path(human["sessions"][0]["edit_journal"]["path"]).read_text())
    refs = random_references(journal, "bridge.weight")
    assert len(refs) == 2
    random = run_lab(
        token_classifier,
        dataset,
        policy,
        tmp_path / "random",
        controller=HostControl("bridge.weight", random_seed=7, references=refs),
        **kwargs,
    )
    row = summarize_run(random, "random_valid", 42)
    assert row["valid_nonempty_edits"] == 2 and not row["model_authored_actions"]
    assert "rollback" in row["valid_action_names"]
    assert "evaluate_candidate" in row["valid_action_names"]
    assert random["base_restored"]
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "step.md"))
    envelope = {
        "profile": "fixture",
        "model": token_classifier.describe(),
        "state_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "plan_sha256": "c" * 64,
    }
    evidence = github_evidence(envelope, row)
    text = capsys.readouterr().out
    assert text.startswith("::notice title=Ouroboros discovery evidence::")
    assert str(tmp_path) not in text and "report_path" not in text and "practice-0" not in text
    assert (
        evidence["valid_nonempty_edits"] == 2
        and not evidence["milestones"]["validated_autonomous_useful_edit"]
    )
