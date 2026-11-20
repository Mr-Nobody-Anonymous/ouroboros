"""Mechanical control-flow fixtures plus a real (random, not pretrained) local LM integration."""

import json
from pathlib import Path

import pytest

from ouroboros.data import DatasetBundle, Sample
from ouroboros.self_edit import SelfEditPolicy, model_fingerprint, run_self_edit

pytestmark = pytest.mark.ml


@pytest.fixture
def backend():
    torch = pytest.importorskip("torch")

    class Backend:
        def __init__(self):
            self.model = torch.nn.Module()
            self.model.mlp = torch.nn.Module()
            self.model.mlp.down_proj = torch.nn.Linear(32, 32, bias=False)
            self.model.register_buffer("scalar", torch.tensor(1))
            with torch.no_grad():
                self.model.mlp.down_proj.weight.fill_(1)
            self.model.train()
            self.model.mlp.eval()  # Mixed module flags must restore exactly.
            self.prompts, self.proposals = [], []
            self.failure, self.drift, self.abstain = None, False, False

        def describe(self):
            return {"backend": "mechanical-test-fixture", "is_llm": False, "parameters": 1024}

        def predict(self, prompts):
            self.prompts.append(list(prompts))
            torch.rand(1)  # The loop owns and restores caller Torch RNG state.
            delta = 1024 - float(self.model.mlp.down_proj.weight.detach().sum())
            if any("FINAL" in p for p in prompts) and self.failure == "final":
                raise RuntimeError("final evaluation failed")
            if delta and self.failure in ("candidate", "interrupt"):
                if self.failure == "interrupt":
                    raise KeyboardInterrupt()
                raise RuntimeError("candidate evaluation failed")
            output = []
            for prompt in prompts:
                if prompt.startswith("validation-"):
                    if self.drift:
                        # First edit improves mean but loses 1/4 memory; second loses 2/4.
                        phase = 2 if delta > 2 else 1 if delta > 0.5 else 0
                        index = int(prompt.rsplit("-", 1)[1])
                        correct = (index < 4 - phase) if "memory" in prompt else (index < phase * 2)
                    else:
                        index = int(prompt.rsplit("-", 1)[1])
                        correct = delta > (0.5 if index == 0 else 2)
                    output.append("good" if correct else "bad")
                else:
                    output.append("good")
            return output

        def choose(self, instruction, labels):
            self.proposals.append(instruction)
            torch.rand(1)
            choice = "A" if self.abstain else "B"
            if self.failure == "proposal":
                choice = "outside-menu"
            return {"choice": choice, "method": "unit-test-stub"}

    return Backend()


def dataset():
    return DatasetBundle(
        [Sample("practice", "good")],
        [Sample("validation-0", "good"), Sample("validation-1", "good")],
        test=[Sample("FINAL-UNSEEN-PROMPT", "FINAL-SECRET-LABEL")],
    )


def execute(backend, tmp_path, **policy):
    return run_self_edit(
        backend,
        dataset(),
        SelfEditPolicy(edited_fraction=0.01, max_relative_norm=0.1, require_attempt=True, **policy),
        tmp_path,
        provenance="mechanical unit fixture, not a pretrained LLM",
        license_name="MIT",
    )


def test_accepted_chain_is_used_then_reversed_with_all_state_flags_and_rng_restored(
    backend, tmp_path
):
    import torch

    before = model_fingerprint(backend.model)
    modes = [m.training for m in backend.model.modules()]
    rng = torch.get_rng_state().clone()
    report = execute(backend, tmp_path)
    assert [r["trial"]["passed"] for r in report["rounds"]] == [True, True]
    assert len(report["accepted_chain"]) == 2
    assert report["selected_state_sha256"] != before
    assert report["base_restored"] and model_fingerprint(backend.model) == before
    assert torch.equal(rng, torch.get_rng_state())
    assert [m.training for m in backend.model.modules()] == modes
    assert sum(any("FINAL" in p for p in prompts) for prompts in backend.prompts) == 1
    assert not report["final_test"]["used_for_promotion"]
    assert all(
        "FINAL-SECRET-LABEL" not in text
        and "FINAL-UNSEEN" not in text
        and "validation-" not in text
        for text in backend.proposals
    )
    assert all(Path(path, "delta.safetensors").is_file() for path in report["accepted_chain"])


def test_original_baseline_prevents_incremental_retention_drift(backend, tmp_path):
    backend.drift = True
    bundle = DatasetBundle(
        [Sample("practice-memory", "good", "memory"), Sample("practice-skill", "good", "skill")],
        [
            Sample(f"validation-{domain}-{i}", "good", domain)
            for domain in ("memory", "skill")
            for i in range(4)
        ],
        test=[Sample("FINAL-UNSEEN-PROMPT", "good", "skill")],
    )
    report = run_self_edit(
        backend,
        bundle,
        SelfEditPolicy(
            edited_fraction=0.01,
            max_relative_norm=0.1,
            forgetting_threshold=0.25,
            require_attempt=True,
        ),
        tmp_path,
        provenance="mechanical fixture",
        license_name="MIT",
    )
    assert [r["trial"]["passed"] for r in report["rounds"]] == [True, False]
    assert "Forgetting" in report["rounds"][1]["trial"]["reason"]
    assert len(report["accepted_chain"]) == 1 and report["base_restored"]


@pytest.mark.parametrize(
    "failure,exception",
    [
        ("candidate", RuntimeError),
        ("interrupt", KeyboardInterrupt),
        ("proposal", ValueError),
        ("final", RuntimeError),
    ],
)
def test_failures_and_cancellation_restore_model_and_leave_honest_report(
    backend, tmp_path, failure, exception
):
    backend.failure = failure
    before = model_fingerprint(backend.model)
    with pytest.raises(exception):
        execute(backend, tmp_path)
    report = json.loads(next(tmp_path.glob("*/self_edit.json")).read_text())
    assert report["status"] == ("interrupted" if exception is KeyboardInterrupt else "failed")
    assert report["base_restored"] and model_fingerprint(backend.model) == before
    assert report["final_test"]["status"] == (
        "failed" if failure == "final" else "skipped_after_failure"
    )


def test_abstention_and_no_representable_change_export_no_patch(backend, tmp_path):
    backend.abstain = True
    report = run_self_edit(
        backend,
        dataset(),
        SelfEditPolicy(),
        tmp_path / "abstain",
        provenance="fixture",
        license_name="MIT",
    )
    assert report["status"] == "abstained" and report["rounds"] == [] and report["base_restored"]
    backend.abstain = False
    report = execute(backend, tmp_path / "no-change", max_rounds=1, max_matrix_elements=2_000_000)
    assert report["base_restored"]
    # With zero allowed fraction, an actual model-selected trial must be a measured rejection.
    report = run_self_edit(
        backend,
        dataset(),
        SelfEditPolicy(max_rounds=1, edited_fraction=0, require_attempt=True),
        tmp_path / "zero",
        provenance="fixture",
        license_name="MIT",
    )
    assert report["rounds"][0]["trial"]["status"] == "rejected" and not report["accepted_chain"]
    assert not list((tmp_path / "zero").rglob("delta.safetensors"))


def test_failed_reverse_is_not_misreported_as_completed(backend, tmp_path, monkeypatch):
    import ouroboros.self_edit as module

    real = module.apply_weight_patch

    def fail_reverse(model, path, *, reverse=False):
        if reverse:
            raise RuntimeError("simulated corrupt patch during reversal")
        return real(model, path)

    monkeypatch.setattr(module, "apply_weight_patch", fail_reverse)
    with pytest.raises(RuntimeError, match="corrupt patch"):
        execute(backend, tmp_path, max_rounds=1)
    report = json.loads(next(tmp_path.glob("*/self_edit.json")).read_text())
    assert report["status"] == "failed" and not report["base_restored"]
    assert "reversal" in report["restoration_error"]


def test_full_state_fingerprint_includes_scalar_buffers(backend):
    before = model_fingerprint(backend.model)
    backend.model.scalar.fill_(2)
    assert model_fingerprint(backend.model) != before


@pytest.fixture
def tiny_chat_model(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    vocab = {
        word: i
        for i, word in enumerate(
            ["PAD", "UNK", "EOS", "A", "B", "C", "D", "E", "user", "assistant"]
        )
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="UNK"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="PAD", unk_token="UNK", eos_token="EOS"
    )
    fast.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=len(vocab),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=2048,
                bos_token_id=2,
                eos_token_id=2,
                pad_token_id=0,
            )
        )
    directory = tmp_path / "tiny-random-chat-model"
    model.save_pretrained(directory, safe_serialization=True)
    fast.save_pretrained(directory)
    return directory


def test_real_local_llm_selects_a_bounded_edit_and_keeps_disk_and_rng_intact(
    tiny_chat_model, tmp_path
):
    import torch

    from ouroboros.backends.local_model import LocalCausalLM
    from ouroboros.config import ModelConfig
    from ouroboros.project import init_project
    from ouroboros.self_edit import load_self_edit_config, run_configured_self_edit

    backend = LocalCausalLM(
        ModelConfig(model_name_or_path=str(tiny_chat_model), max_new_tokens=1, prompt_format="chat")
    )
    assert backend.render_prompt("hello") == "user hello assistant "
    choice = backend.choose("Choose B or C", ["B", "C"])
    assert choice["choice"] in ("B", "C") and choice["proposer_is_target_model"]
    assert sum(choice["probabilities_within_menu"].values()) == pytest.approx(1)
    for invalid in (["B C", "D"], ["unknown1", "unknown2"], ["B", "B"], []):
        with pytest.raises(ValueError):
            backend.choose("short", invalid)
    init_project(tmp_path / "lab")
    path = tmp_path / "lab/self_edit.json"
    values = json.loads(path.read_text())
    values["model"].update(model_name_or_path=str(tiny_chat_model), max_new_tokens=1)
    values["policy"].update(max_rounds=1, edited_fraction=0.01, max_relative_norm=0.1)
    path.write_text(json.dumps(values))
    threads = torch.get_num_threads()
    report = run_configured_self_edit(load_self_edit_config(path))
    assert report["base_restored"] and report["source_files_unchanged"]
    assert (
        report["rounds"][0]["trial"]["model_restored"]
        and report["rounds"][0]["trial"]["edit"]["edited_weights"] > 0
    )
    assert report["model"]["is_llm"] and report["rounds"][0]["proposal"]["proposer_is_target_model"]
    assert report["final_test"]["status"] == "completed" and torch.get_num_threads() == threads
