"""Explicit practice sensitivity / completion NLL, not an optimizer or a no-gradient claim.

Targets are independently tokenized after the actual inference prompt; terminal model EOS is
included. No truncation, label masking of completion tokens, parameter update or .grad writes.
"""

import inspect
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

from .config import Sampling, integer
from .metrics import problem_id
from .state import editable_parameter


@contextmanager
def gradient_mode(model):
    import torch

    flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        # run_lab's inference context must not accidentally disable this explicit opt-in tool.
        with torch.inference_mode(False), torch.enable_grad(), torch.random.fork_rng():
            yield
    finally:
        for module, flag in flags:
            module.training = flag


def completion_loss(backend, sample, policy, *, sampling=None):
    """Return a differentiable mean NLL and token count, with bounded logits allocation."""
    import torch
    import torch.nn.functional as functional

    if (
        not callable(getattr(backend, "render_research_prompt", None))
        or not getattr(backend, "generation_tokens", {}).get("eos_token_ids")
        or not hasattr(backend.model, "config")
    ):
        raise ValueError(
            "Completion loss/gradients require a supported causal backend with tokenizer, model config and model-declared EOS; no surrogate loss fallback"
        )
    prompt = backend.tokenizer.encode(
        backend.render_research_prompt(sample.instruction, sampling or Sampling()),
        add_special_tokens=backend.config.prompt_format != "chat",
    )
    completion = backend.tokenizer.encode(sample.target, add_special_tokens=False)
    completion += [backend.generation_tokens["eos_token_ids"][0]]
    length = len(prompt) + len(completion)
    if not prompt or not completion or length > min(backend.context_size, policy.max_read_tokens):
        raise ValueError(
            "Complete prompt + reference + EOS exceed loss/read context; no truncation"
        )
    keep = "logits_to_keep" in inspect.signature(backend.model.forward).parameters
    config = getattr(backend.model.config, "text_config", backend.model.config)
    count = len(completion) + 1 if keep else length
    if count * config.vocab_size > policy.max_logit_elements:
        raise ValueError("Completion-loss logits exceed the element budget")
    ids = torch.tensor([prompt + completion], device=backend.device, dtype=torch.long)
    kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "use_cache": False}
    if keep:
        kwargs["logits_to_keep"] = len(completion) + 1
    output = backend.model(**kwargs)
    logits = output.logits[0, -len(completion) - 1 : -1].float()
    targets = torch.tensor(completion, device=backend.device, dtype=torch.long)
    return functional.cross_entropy(logits, targets), len(completion)


def measure_loss(backend, samples, policy, *, sampling=None):
    import torch

    rows, domains = [], defaultdict(list)
    with gradient_mode(backend.model), torch.no_grad():
        for sample in samples:
            loss, count = completion_loss(backend, sample, policy, sampling=sampling)
            value = float(loss.cpu())
            if not np.isfinite(value):
                raise ValueError("Nonfinite completion NLL")
            row = {"id": problem_id(sample), "domain": sample.domain, "nll": value, "tokens": count}
            rows.append(row)
            domains[sample.domain].append(row)
    return {
        "criterion": "teacher-forced independently tokenized reference plus model EOS; natural log, no truncation",
        "problems": len(rows),
        "completion_tokens": sum(row["tokens"] for row in rows),
        "problem_mean_nll": sum(row["nll"] for row in rows) / len(rows),
        "token_mean_nll": sum(row["nll"] * row["tokens"] for row in rows)
        / sum(row["tokens"] for row in rows),
        "by_domain": {
            key: sum(r["nll"] for r in value) / len(value) for key, value in sorted(domains.items())
        },
        "records": rows,
    }


def inspect_gradients(backend, copy, samples, name, *, limit=16, sampling=None):
    """Autograd on one permitted tensor and practice labels. Does not perform gradient descent."""
    import torch

    integer("gradient prefix limit", limit, 1, 256)
    parameter = editable_parameter(backend.model, name, copy.policy)
    flags = [(value, value.requires_grad) for value in backend.model.parameters()]
    total, losses = None, []
    try:
        with gradient_mode(backend.model):
            # Only the selected tensor requires grad; do not build a whole-model training graph.
            for value, _ in flags:
                value.requires_grad_(False)
            parameter.requires_grad_(True)
            for sample in samples:
                loss, _ = completion_loss(backend, sample, copy.policy, sampling=sampling)
                gradient = (
                    torch.autograd.grad(loss, parameter, create_graph=False)[0]
                    .detach()
                    .float()
                    .cpu()
                )
                total = gradient if total is None else total + gradient
                losses.append(float(loss.detach().cpu()))
            total /= len(samples)
    finally:
        for value, flag in flags:
            value.requires_grad_(flag)
    if not torch.isfinite(total).all():
        raise ValueError("Nonfinite practice gradient")
    flat = total.reshape(-1).numpy()
    indices = np.argsort(-np.abs(flat), kind="stable")[: min(limit, len(flat))]
    return {
        "name": name,
        "examples": len(samples),
        "problem_mean_nll": sum(losses) / len(losses),
        "gradient_norm": float(np.linalg.norm(flat.astype(np.float64))),
        "entries": [{"index": int(i), "gradient": float(flat[i])} for i in indices],
        "updates_performed": 0,
        "interpretation": "Local teacher-forced practice sensitivity, not causal weight meaning or a gradient-free experiment. No optimizer, .grad accumulation or parameter update.",
    }
