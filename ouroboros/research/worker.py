"""Fixed offline worker operations, never model-generated code. Invoked only by the controller.

Process separation prevents accidental live-target contamination, NOT filesystem secrecy or
OS sandboxing. Workers contain no shell, plugin/import path, network or arbitrary file-read DSL.
"""

import ctypes
import json
import os
import resource
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from ..config import ModelConfig
from ..core.checkpoints import file_sha256, verify_checkpoint
from ..data import Sample
from ..lab.config import LabPolicy, Sampling, strict_json
from ..lab.instrumentation import read_states
from ..lab.metrics import evaluate
from ..lab.sensitivity import inspect_gradients, measure_loss
from .contracts import Budget, Recipe, from_dict, identity
from .evaluation import audit_rewards
from .provenance import assert_software, model_inventory, verify_binding, write_once


def cohort(binding):
    value = strict_json(verify_binding(binding).read_text())
    if not isinstance(value, list) or not 1 <= len(value) <= 100000:
        raise ValueError("Invalid worker cohort")
    return [Sample.from_dict(row) for row in value]


def measure(backend, samples, budget, seed):
    started = time.monotonic()
    result = evaluate(
        backend, samples, sampling=Sampling(), grader="exact", samples_per_problem=1, seed=seed
    )
    policy = LabPolicy(
        max_read_tokens=budget.max_seq_length,
        max_logit_elements=16_000_000,
        max_model_parameters=budget.max_parameters,
    )
    nll = measure_loss(backend, samples, policy)
    result["loss_records"] = nll["records"]
    result["nll"] = {key: value for key, value in nll.items() if key != "records"}
    result["efficiency"] = {
        "seconds": time.monotonic() - started,
        "generated_tokens": sum(row["new_tokens"] for row in result["records"]),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "unit": "single-process CPU wall-clock, descriptive, not a hardware-normalized speed claim",
    }
    return result


def introspect(backend, samples, budget):
    """Small architecture/parameter/hidden/gradient observations on practice ONLY."""
    from ..lab.state import editable_parameter

    model = backend.model
    params = [
        {"name": name, "shape": list(p.shape), "dtype": str(p.dtype), "trainable": p.requires_grad}
        for name, p in model.named_parameters()
    ]
    modules = dict(model.named_modules())
    layers = [
        name
        for name, module in modules.items()
        if name
        and (
            "decoderlayer" in type(module).__name__.lower()
            or type(module).__name__.lower() in ("gpt2block",)
        )
    ]
    result = {
        "description": backend.describe(),
        "architecture": {
            key: getattr(model.config, key, None)
            for key in (
                "model_type",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "intermediate_size",
                "vocab_size",
                "max_position_embeddings",
                "num_experts",
                "n_routed_experts",
                "kv_lora_rank",
            )
        },
        "parameter_metadata": params[:512],
        "metadata_truncated": len(params) > 512,
        "interpretation": "Bounded numeric observations, not semantics or a causal explanation.",
    }
    policy = LabPolicy(
        max_read_tokens=budget.max_seq_length,
        max_read_texts=1,
        max_read_elements=16000,
        max_matrix_elements=4096,
    )
    if layers:
        raw = read_states(
            backend,
            policy=policy,
            kind="hidden",
            modules=[layers[-1]],
            texts=[backend.render_research_prompt(samples[0].instruction, Sampling())],
            require_full_text=True,
        )
        result["hidden_states"] = raw
    selected = None
    for name, parameter in reversed(list(model.named_parameters())):
        if parameter.ndim == 1 and parameter.numel() <= 4096:
            try:
                editable_parameter(model, name, policy)
                selected = name
                break
            except ValueError:
                continue
    if selected:
        result["practice_gradient_assistance"] = inspect_gradients(
            backend, SimpleNamespace(model=model, policy=policy), samples[:1], selected, limit=8
        )
    else:
        result["practice_gradient_assistance"] = {
            "available": False,
            "reason": "No small permitted non-final tensor",
        }
    return result


def execute(job):
    required = {
        "schema_version",
        "operation",
        "budget",
        "model",
        "model_files",
        "software",
        "seed",
        "inputs",
        "output",
        "checkpoint",
        "recipe",
        "root",
        "run_id",
        "extra",
    }
    if not isinstance(job, dict) or set(job) != required or job["schema_version"] != 1:
        raise ValueError("Invalid fixed-worker request")
    if job["operation"] not in ("observe", "evaluate", "train", "research", "distill", "followup"):
        raise ValueError("Unknown worker operation")
    budget = from_dict(Budget, job["budget"])
    model_config = from_dict(ModelConfig, job["model"])
    assert_software(job["software"])
    if model_inventory(model_config.model_name_or_path) != job["model_files"]:
        raise ValueError("Base model changed after preregistration")
    import torch

    torch.set_num_threads(budget.cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(job["seed"])
    from .training import load_target, train_audited

    operation = job["operation"]
    recipe = from_dict(Recipe, job["recipe"]) if job["recipe"] else None
    if operation == "research":
        from .researcher import local_proposal

        result = local_proposal(job["extra"], model_config, budget, job["seed"])
    elif operation == "distill":
        from ..lab.backend import ResearchCausalLM

        teacher = ResearchCausalLM(model_config)
        samples = cohort(job["inputs"]["train"])
        accepted, audits = [], []
        # Training prompts only. Correctness is host reference matching, never teacher self-rating.
        for sample in samples:
            requests = [
                {"prompt": sample.instruction, "seed": (job["seed"] + i) % (2**31)}
                for i in range(4)
            ]
            rows = teacher.generate_records(requests, sampling=Sampling(temperature=0.7))
            audit = audit_rewards(rows, sample.target)
            audits.append({"id": identity(sample.instruction), **audit})
            good = next((rows[i] for i, r in enumerate(audit["records"]) if r["correct"]), None)
            if good:
                accepted.append(
                    Sample(
                        sample.instruction,
                        good["text"],
                        sample.domain,
                        "verified-teacher:" + identity(job["model_files"]),
                        True,
                    ).as_dict()
                )
        binding = write_once(Path(job["root"]) / "verified_distillation.json", accepted)
        result = {
            "accepted": len(accepted),
            "attempted_prompts": len(samples),
            "corpus": binding,
            "audits": audits,
            "teacher": teacher.describe(),
            "rationale_correctness_verified": False,
        }
    else:
        # Candidate is loaded in a fresh worker, never in the proposer/controller's target.
        checkpoint = job["checkpoint"]
        if checkpoint and identity(verify_checkpoint(checkpoint)) != job["extra"].get(
            "checkpoint_manifest_sha256"
        ):
            raise ValueError("Evaluator checkpoint differs from the sealed candidate")
        if operation == "followup" and checkpoint is None:
            recipe = from_dict(Recipe, job["extra"]["original_recipe"])
        backend = load_target(
            model_config,
            budget,
            recipe=recipe if not checkpoint else None,
            seed=job["seed"],
            checkpoint=checkpoint,
        )
        if checkpoint:
            marker = verify_checkpoint(checkpoint)
            if marker["metadata"]["source_binding"] != identity(job["model_files"]):
                raise ValueError("Candidate checkpoint binds a different base model")
        if operation == "observe":
            samples = cohort(job["inputs"]["practice"])
            result = {
                "evaluation": measure(backend, samples, budget, job["seed"]),
                "introspection": introspect(backend, samples, budget),
            }
        elif operation == "evaluate":
            result = measure(backend, cohort(job["inputs"]["evaluation"]), budget, job["seed"])
        elif operation == "train":
            result = train_audited(
                backend,
                cohort(job["inputs"]["train"]),
                recipe,
                budget,
                seed=job["seed"],
                root=job["root"],
                run_id=job["run_id"],
                source_binding=identity(job["model_files"]),
                deadline=time.monotonic() + budget.max_seconds,
            )
        else:
            # Paired four-cell retention: base/candidate before and after REAL follow-up SFT.
            # Both cells use exactly the candidate's adapter structure and the predeclared
            # follow-up optimizer/data/steps. No merging of hidden-selected weights into baseline.
            from dataclasses import replace

            original = from_dict(Recipe, job["extra"]["original_recipe"])
            fixed = from_dict(Recipe, job["extra"]["followup_recipe"])
            effective = replace(
                fixed,
                family=original.family,
                rank=original.rank,
                alpha=original.alpha,
                target_modules=original.target_modules,
            )
            trained = train_audited(
                backend,
                cohort(job["inputs"]["followup"]),
                effective,
                budget,
                seed=job["seed"],
                root=job["root"],
                run_id=job["run_id"],
                source_binding=identity(job["model_files"]),
                deadline=time.monotonic() + budget.max_seconds,
            )
            result = {
                "training": trained,
                "effective_recipe": asdict(effective),
                "evaluation_requires_fresh_worker": True,
            }
    if job["checkpoint"] and identity(verify_checkpoint(job["checkpoint"])) != job["extra"].get(
        "checkpoint_manifest_sha256"
    ):
        raise ValueError("Candidate checkpoint changed during worker execution")
    if model_inventory(model_config.model_name_or_path) != job["model_files"]:
        raise ValueError("Source model files changed during worker execution")
    assert_software(job["software"])
    return {
        **result,
        "worker": {
            "pid": os.getpid(),
            "operation": operation,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "source_files_unchanged": True,
            "optimizer_output_is_not_a_score": True,
        },
    }


def main(argv=None):
    # Parent death protection is Linux-specific; no security sandbox claim.
    parent = os.getppid()
    if sys.platform.startswith("linux"):
        if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0 or os.getppid() != parent or parent == 1:
            raise RuntimeError("Could not bind worker lifetime to its controller")
    path = Path((sys.argv[1:] if argv is None else argv)[0])
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("Worker job exceeds 4 MiB")
    job = strict_json(path.read_text())
    seconds = job.get("budget", {}).get("max_seconds", 60)
    resource.setrlimit(resource.RLIMIT_CPU, (seconds + 5, seconds + 10))
    result = execute(job)
    write_once(job["output"], result)
    print(json.dumps({"output_sha256": file_sha256(job["output"])}))


if __name__ == "__main__":
    main()
