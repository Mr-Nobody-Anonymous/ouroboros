"""Audited SFT reusing the local loader, PEFT backend, token encoder/collator and managed store.

This separate path refuses prompt truncation, supports controlled optimizer/schedule ablations,
records every real step, and serializes optimizer/RNG tensors without pickle. It does not change
the older generation loop's optimizer-reset contract or silently mix direct base edits into LoRA.
"""

import json
import math
import resource
import time
from dataclasses import asdict
from pathlib import Path

from ..backends.transformers_backend import TransformersLoRABackend
from ..config import TuningConfig
from ..core.checkpoints import CheckpointStore, atomic_json, verify_checkpoint
from ..lab.backend import ResearchCausalLM
from ..lab.config import Sampling
from .contracts import identity


class AuditedLoRA(TransformersLoRABackend, ResearchCausalLM):
    def render_prompt(self, instruction):
        return self.render_research_prompt(instruction, Sampling())


class AuditedFullSFT(ResearchCausalLM):
    # Share the existing completion-label masking and padding implementation, not a surrogate.
    encode_sample = TransformersLoRABackend.encode_sample
    _collate = TransformersLoRABackend._collate

    def render_prompt(self, instruction):
        return self.render_research_prompt(instruction, Sampling())

    def save(self, directory):
        directory = Path(directory)
        self.model.save_pretrained(str(directory), safe_serialization=True)
        self.tokenizer.save_pretrained(str(directory))
        atomic_json(directory / "backend.json", self.describe())


def tuning_for(recipe, budget):
    return TuningConfig(
        lora_r=recipe.rank,
        lora_alpha=recipe.alpha,
        target_modules=recipe.target_modules,
        learning_rate=recipe.learning_rate,
        batch_size=1,
        max_seq_length=budget.max_seq_length,
    )


def followup_effective_recipe(original, fixed):
    """The four-cell retention follow-up, made structure-compatible with the sealed candidate.

    The *fixed* recipe owns the optimizer, schedule, learning rate and step count — those are
    predeclared and identical for both sides of the paired comparison. The intervention family,
    adapter structure and domain weights come from the sealed candidate, because the "after"
    cell has to continue the same intervention the "before" cell measured; a `data_reweight`
    candidate therefore keeps its weights instead of silently becoming an unweighted follow-up.
    """
    from .contracts import Recipe

    return Recipe(
        family=original.family,
        target_modules=list(original.target_modules),
        steps=fixed.steps,
        learning_rate=fixed.learning_rate,
        rank=original.rank,
        alpha=original.alpha,
        optimizer=fixed.optimizer,
        schedule=fixed.schedule,
        weight_decay=fixed.weight_decay,
        domain_weights=dict(original.domain_weights),
    )


def load_target(model_config, budget, *, recipe=None, seed=42, checkpoint=None):
    marker = verify_checkpoint(checkpoint) if checkpoint else None
    if marker:
        recorded = marker["metadata"].get("recipe")
        if not recorded or (recipe and recorded != asdict(recipe)):
            raise ValueError("Checkpoint recipe/provenance mismatch")
        from .contracts import Recipe, from_dict

        recipe = from_dict(Recipe, recorded)
    if recipe is None:
        backend = ResearchCausalLM(model_config)
    elif recipe.family == "full_sft":
        # Full exports are independent weights; nevertheless bind the original base inventory.
        if checkpoint:
            from dataclasses import replace

            backend = AuditedFullSFT(replace(model_config, model_name_or_path=str(checkpoint)))
        else:
            backend = AuditedFullSFT(model_config)
        if sum(p.numel() for p in backend.model.parameters()) > budget.max_full_parameters:
            raise ValueError("Full SFT exceeds the deliberately small-model parameter cap")
    else:
        backend = AuditedLoRA(
            model_config, tuning_for(recipe, budget), seed=seed, adapter_path=checkpoint
        )
    if sum(p.numel() for p in backend.model.parameters()) > budget.max_parameters:
        raise ValueError("Target exceeds the model parameter budget")
    return backend


def strict_rows(backend, samples, budget):
    if not samples or len(samples) > budget.max_train_examples:
        raise ValueError("Training cohort exceeds its fixed example budget")
    eos = backend.generation_tokens["eos_token_ids"][0]
    # Use the aligned model EOS, not stale converted-tokenizer metadata.
    if backend.tokenizer.eos_token_id != eos:
        raise ValueError("Tokenizer/model EOS alignment is required before training")
    for sample in samples:
        prompt = backend.tokenizer.encode(
            backend.render_prompt(sample.instruction),
            add_special_tokens=backend.config.prompt_format != "chat",
        )
        completion = backend.tokenizer.encode(sample.target, add_special_tokens=False) + [eos]
        if not prompt or len(prompt) + len(completion) > min(
            budget.max_seq_length, backend.context_size
        ):
            raise ValueError(
                "Complete training prompt/reference/EOS do not fit; no silent truncation"
            )
    return [
        backend.encode_sample(s, min(budget.max_seq_length, backend.context_size)) for s in samples
    ]


class TrainingExport:
    def __init__(self, backend, optimizer, state):
        self.backend, self.optimizer, self.state = backend, optimizer, state

    def describe(self):
        return self.backend.describe()

    def save(self, directory):
        from safetensors.torch import save_file

        self.backend.save(directory)
        state = self.optimizer.state_dict()
        tensors = {"torch_cpu_rng": self.backend.torch.get_rng_state().clone()}
        encoded = {}
        for index, values in state["state"].items():
            encoded[str(index)] = {}
            for key, value in values.items():
                if hasattr(value, "detach"):
                    name = f"optimizer.{index}.{key}"
                    tensors[name] = value.detach().cpu().contiguous().clone()
                    encoded[str(index)][key] = {"tensor": name}
                elif type(value) in (int, float):
                    encoded[str(index)][key] = value
                else:
                    raise ValueError("Unsupported optimizer-state type; no pickle fallback")
        save_file(tensors, str(Path(directory) / "training_state.safetensors"))
        atomic_json(
            Path(directory) / "training_state.json",
            {
                **self.state,
                "optimizer": {"state": encoded, "param_groups": state["param_groups"]},
                "resume_supported": False,
                "note": "Auditable optimizer/RNG snapshot at a completed step, not automatic mid-step resume. Dead runs require a new plan and new independent confirmation.",
            },
        )


def train_audited(
    backend, samples, recipe, budget, *, seed, root, run_id, source_binding, deadline
):
    import numpy as np
    import torch

    recipe.validate_budget(budget)
    weighted = [s for s in samples for _ in range(recipe.domain_weights.get(s.domain, 1))]
    rows = strict_rows(backend, weighted, budget)
    if recipe.family == "full_sft":
        backend.model.requires_grad_(True)
    parameters = [p for p in backend.model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters; not an intervention")
    optimizer_class = torch.optim.AdamW if recipe.optimizer == "adamw" else torch.optim.SGD
    optimizer = optimizer_class(
        parameters, lr=recipe.learning_rate, weight_decay=recipe.weight_decay
    )
    rng, order, position, losses, paths = np.random.default_rng(seed), [], 0, [], []
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    store = CheckpointStore(root / "checkpoints", run_id, min_free_bytes=16 * 1024**2)
    started = time.monotonic()
    trace_path = root / "steps.jsonl"
    dataset_sha = identity([s.as_dict() for s in weighted])
    torch.manual_seed(seed)
    backend.model.train()
    try:
        with trace_path.open("x") as stream:
            for step in range(recipe.steps):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Training exhausted the frozen wall-clock budget")
                if position >= len(order):
                    order, position = rng.permutation(len(rows)).tolist(), 0
                index = order[position]
                position += 1
                rate = recipe.learning_rate * (
                    1
                    if recipe.schedule == "constant"
                    else 0.5 * (1 + math.cos(math.pi * step / recipe.steps))
                )
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.zero_grad(set_to_none=True)
                # Full-prompt completion-only real causal-LM cross-entropy.
                output = backend.model(**backend._collate([rows[index]]), use_cache=False)
                loss = output.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite SFT loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite SFT gradient norm")
                optimizer.step()
                value = float(loss.detach().cpu())
                losses.append(value)
                row = {
                    "step": step + 1,
                    "loss": value,
                    "gradient_norm_before_clip": float(norm),
                    "gradient_clip": 1.0,
                    "learning_rate": rate,
                    "training_example_index": index,
                    "completion_tokens": sum(v != -100 for v in rows[index][1][1:]),
                    "seconds": time.monotonic() - started,
                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if (step + 1) % budget.checkpoint_every == 0 or step + 1 == recipe.steps:
                    export = TrainingExport(
                        backend,
                        optimizer,
                        {
                            "step": step + 1,
                            "recipe": asdict(recipe),
                            "seed": seed,
                            "numpy_rng": rng.bit_generator.state,
                            "order": order,
                            "position": position,
                            "training_data_sha256": dataset_sha,
                        },
                    )
                    name = store.save(
                        export,
                        step + 1,
                        scores={},
                        parent=paths[-1] if paths else None,
                        metadata={
                            "recipe": asdict(recipe),
                            "seed": seed,
                            "training_data_sha256": dataset_sha,
                            "source_binding": source_binding,
                        },
                    )
                    paths.append(name)
                    backend.model.train()
    finally:
        optimizer.zero_grad(set_to_none=True)
        backend.model.eval()
    return {
        "steps": len(losses),
        "loss_mean": sum(losses) / len(losses),
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "trainable_parameters": sum(p.numel() for p in parameters),
        "checkpoint": str((store.root / paths[-1]).resolve()),
        "checkpoints": paths,
        "training_data_sha256": dataset_sha,
        "trace_file": str(trace_path.resolve()),
        "seconds": time.monotonic() - started,
        "optimizer_reset_at_start": True,
        "mid_step_resume": False,
        "is_real_training": True,
    }
