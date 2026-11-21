"""Shared optional research fixtures. Importing conftest does not import Torch."""

import json

import pytest


@pytest.fixture
def lab_policy():
    from ouroboros.lab.config import LabPolicy

    return LabPolicy(
        max_edited_fraction=1.0,
        max_relative_norm=2.0,
        max_changed_entries=64,
        max_turns=16,
        max_seconds=60,
        cpu_threads=1,
        permutation_draws=100,
    )


@pytest.fixture
def lab_backend():
    torch = pytest.importorskip("torch")
    from ouroboros.config import ModelConfig

    class TokenizerStub:
        def encode(self, text, **kwargs):
            return list(range(len(text.split())))

    class Backend:
        """Actual tiny Torch classifier, scripted editor replies: NOT a pretrained/autonomous LM."""

        def __init__(self):
            self.model = torch.nn.Module()
            self.model.a = torch.nn.Linear(2, 2, bias=False)
            self.model.b = torch.nn.Linear(2, 2, bias=False)
            self.model.drift = torch.nn.Parameter(torch.tensor([0.1, 0.2]))
            self.model.register_buffer("scalar", torch.tensor(1))
            with torch.no_grad():
                self.model.a.weight.copy_(torch.tensor([[-1.0, 0.1], [0.2, 1.0]]))
                self.model.b.weight.copy_(torch.tensor([[1.0, 0.1], [0.1, 1.0]]))
            self.model.train()
            self.model.b.eval()
            self.config = ModelConfig(max_new_tokens=4)
            self.context_size, self.tokenizer = 100000, TokenizerStub()
            self.editor_actions, self.editor_prompts, self.requests = [], [], []
            self.failure, self.invalid_reply = None, False

        def describe(self):
            return {
                "backend": "torch-classifier-with-scripted-editor-test-fixture",
                "is_llm": False,
                "parameters": 10,
            }

        def render_research_prompt(self, prompt, sampling):
            return prompt

        def generate_records(self, requests, *, sampling, max_new_tokens=None):
            result = []
            for request in requests:
                prompt = request["prompt"]
                self.requests.append(dict(request))
                if '"recent_conversation"' in prompt:
                    self.editor_prompts.append(prompt)
                    text = (
                        "not JSON"
                        if self.invalid_reply
                        else json.dumps(
                            self.editor_actions.pop(0)
                            if self.editor_actions
                            else {"action": "withdraw"}
                        )
                    )
                else:
                    if "FINAL-QUESTION" in prompt and self.failure == "final":
                        raise RuntimeError("final fixture inference failed")
                    if self.model.a.weight[0, 0] > 0 and self.failure in ("candidate", "interrupt"):
                        if self.failure == "interrupt":
                            raise KeyboardInterrupt()
                        raise RuntimeError("candidate fixture inference failed")
                    feature = (
                        torch.tensor([1.0, 0.0]) if "warm" in prompt else torch.tensor([0.0, 1.0])
                    )
                    with torch.inference_mode():
                        index = int(self.model.b(self.model.a(feature)).argmax())
                    text = ["warm", "cool"][index]
                result.append(
                    {"text": text, "new_tokens": 1, "cut": False, "thinking_closed": True}
                )
            return result

    return Backend()


@pytest.fixture
def lab_dataset():
    from ouroboros.data import DatasetBundle, Sample

    return DatasetBundle(
        [Sample(f"practice {label}", label, "labels") for label in ("warm", "cool")],
        [Sample(f"RESERVED-QUESTION {label}", label, "labels") for label in ("warm", "cool")],
        test=[Sample(f"FINAL-QUESTION {label}", label, "labels") for label in ("warm", "cool")],
    )


@pytest.fixture
def lab_copy(lab_backend, lab_policy):
    from ouroboros.lab.state import WorkingCopy

    return WorkingCopy(lab_backend.model, lab_policy)
