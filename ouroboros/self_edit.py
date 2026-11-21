"""A bounded SAME-model proposal -> weight trial -> gate loop. No generated code execution.

The model selects from an explicit host-defined menu using its own next-token logits. This is
not unconstrained program synthesis, semantic weight understanding, or a promised improvement.
"""

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import ModelConfig, probability
from .core.checkpoints import atomic_json
from .core.evaluator import evaluate_samples
from .core.weight_trials import (
    WeightEditPlan,
    _tensor_hash,
    apply_weight_patch,
    run_weight_edit_trial,
)
from .data import DatasetBundle, load_samples
from .model_files import RECEIPT, verify_model_files
from .tensor_inspection import inspect_safetensors


@dataclass(frozen=True)
class SelfEditPolicy:
    max_rounds: int = 2
    max_seconds: int = 240
    edited_fraction: float = 0.001
    max_relative_norm: float = 0.02
    forgetting_threshold: float = 0.0
    improvement_threshold: float = 0.0
    require_attempt: bool = False
    seed: int = 42
    cpu_threads: int = 2
    max_model_parameters: int = 300_000_000
    max_model_bytes: int = 1024**3
    max_matrix_elements: int = 2_000_000

    def __post_init__(self):
        for key, maximum in {
            "max_rounds": 4,
            "max_seconds": 1800,
            "cpu_threads": 8,
            "max_model_parameters": 1_000_000_000,
            "max_model_bytes": 4 * 1024**3,
            "max_matrix_elements": 8_000_000,
        }.items():
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{key} must be an integer in [1, {maximum}]")
        for key in (
            "edited_fraction",
            "max_relative_norm",
            "forgetting_threshold",
            "improvement_threshold",
        ):
            probability(key, getattr(self, key))
        if self.edited_fraction > 0.01 or self.max_relative_norm > 0.1:
            raise ValueError("Self-edit policy caps edited fraction at 1% and relative norm at 10%")
        if (
            type(self.require_attempt) is not bool
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("require_attempt must be boolean and seed an integer in [0, 2**63)")


def scoped_path(root, value):
    """MCP boundary: reject traversal, hidden paths and symlink components; not a hostile-FS sandbox."""
    root = Path(root).resolve()
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError("A bounded path string is required")
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Path escapes the configured workspace")
    relative = candidate.relative_to(root) if candidate.is_relative_to(root) else None
    if relative is None or any(part.startswith(".") for part in relative.parts):
        raise ValueError("Traversal and hidden paths are not allowed")
    check = candidate
    while check != root:
        if check.is_symlink():
            raise ValueError("Symlinked paths are not allowed")
        check = check.parent
    return resolved


def load_self_edit_config(path, *, workspace=None):
    path = scoped_path(workspace, str(path)) if workspace is not None else Path(path).resolve()
    if path.stat().st_size > 64 * 1024:
        raise ValueError("Self-edit config exceeds 64 KiB")
    values = json.loads(path.read_text(encoding="utf-8"))
    required = {"model", "policy", "data", "output_dir", "license", "provenance"}
    if not isinstance(values, dict) or set(values) != required:
        raise ValueError("Self-edit config requires exactly: " + ", ".join(sorted(required)))
    for key in ("license", "provenance"):
        if not isinstance(values[key], str) or not 1 <= len(values[key].strip()) <= 4000:
            raise ValueError(f"Explicit bounded {key} is required")

    def resolve(value):
        if not isinstance(value, str) or not value:
            raise ValueError("Config paths must be nonempty strings")
        candidate = path.parent / Path(value).expanduser()
        return str(
            scoped_path(workspace, str(candidate)) if workspace is not None else candidate.resolve()
        )

    model_values = dict(values["model"])
    model_values["model_name_or_path"] = resolve(model_values.get("model_name_or_path"))
    model = ModelConfig(**model_values)
    if model.use_qlora or model.max_new_tokens > 64:
        raise ValueError("Self-edit supports unquantized models and at most 64 generated tokens")
    data = values["data"]
    if not isinstance(data, dict) or set(data) != {"train", "validation", "test"}:
        raise ValueError(
            "Self-edit requires separate train/practice, validation and final-test files"
        )
    splits = {}
    for key in ("train", "validation", "test"):
        data_path = Path(resolve(data[key]))
        if data_path.stat().st_size > 512 * 1024:
            raise ValueError("Self-edit data file exceeds 512 KiB")
        splits[key] = load_samples(data_path)
        if not 1 <= len(splits[key]) <= 32 or any(
            len(s.instruction) > 2000 or len(s.target) > 1000 for s in splits[key]
        ):
            raise ValueError("Each split needs 1-32 bounded smoke examples")
    dataset = DatasetBundle(splits["train"], splits["validation"], test=splits["test"])
    return {
        "model": model,
        "policy": SelfEditPolicy(**values["policy"]),
        "dataset": dataset,
        "output_dir": resolve(values["output_dir"]),
        "license_name": values["license"],
        "provenance": values["provenance"],
    }


def self_edit_config_summary(config):
    return {
        "valid": True,
        "model_loaded": False,
        "network_accessed": False,
        "model": asdict(config["model"]),
        "policy": asdict(config["policy"]),
        "counts": {
            key: len(getattr(config["dataset"], key)) for key in ("train", "validation", "test")
        },
        "dataset_sha256": config["dataset"].fingerprint(),
        "note": "Schema/split check only: model presence, tensor shapes, license rights and memory are not verified.",
    }


def preflight_model(config):
    """Bound on-disk payload and parameter elements before allocating model weights."""
    path = Path(config["model"].model_name_or_path)
    policy = config["policy"]
    from .model_files import model_file

    if path.is_dir() and any(p.is_symlink() for p in path.iterdir() if model_file(p.name)):
        raise ValueError("Self-edit model files must not be symlinks")
    files = sorted(path.glob("*.safetensors"))
    index = path / "model.safetensors.index.json"
    if index.exists():
        if index.stat().st_size > 1024 * 1024:
            raise ValueError("Model shard index exceeds 1 MiB")
        metadata = json.loads(index.read_text(encoding="utf-8"))
        mapping = metadata.get("weight_map", {}) if isinstance(metadata, dict) else {}
        if (
            not isinstance(mapping, dict)
            or not mapping
            or any(
                not isinstance(name, str) or name not in {file.name for file in files}
                for name in mapping.values()
            )
        ):
            raise ValueError("Shard index must refer only to local inventoried safetensors files")
    if not files or len(files) > 64:
        raise ValueError("A local safetensors model is required; use models fetch explicitly first")
    if sum(file.stat().st_size for file in files) > policy.max_model_bytes:
        raise ValueError("Model exceeds on-disk byte budget")
    elements = 0
    for file in files:
        inspection = inspect_safetensors(file, limit=1000)
        if inspection["truncated"]:
            raise ValueError("Model tensor inventory exceeds inspection budget")
        elements += sum(t["numel"] for t in inspection["tensors"])
    if elements > policy.max_model_parameters:
        raise ValueError("Model exceeds parameter-element budget")
    receipt = verify_model_files(path) if (path / RECEIPT).exists() else None
    return {"stored_tensor_elements": elements, "download_receipt": receipt}


def model_fingerprint(model):
    """Content hash of the complete state dict, one CPU tensor at a time (still dense RAM/I/O)."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode() + b"\0" + _tensor_hash(tensor).encode() + b"\n")
    return digest.hexdigest()


def edit_menu(model, policy):
    """Finite coordinate-ablation menu, never a claim of causal/semantic circuit discovery."""
    import torch

    preferred, fallback = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2 or not 4 <= parameter.numel() <= policy.max_matrix_elements:
            continue
        if parameter.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            continue
        if any(part in name for part in ("mlp", "feed_forward", "fc2")):
            fallback.append((name, parameter))
            if name.endswith(("down_proj.weight", "mlp.c_proj.weight", "fc2.weight")):
                preferred.append((name, parameter))
    selected = (preferred or fallback)[-2:]
    menu = []
    for name, parameter in selected:
        # Direction is a named, inspectable coordinate, not an invented activation diagnosis.
        row = int(parameter.detach().float().square().sum(dim=1).argmax().cpu())
        for strength in (0.1, 0.25):
            menu.append(
                {
                    "parameter_name": name,
                    "shape": list(parameter.shape),
                    "row": row,
                    "strength": strength,
                    "direction": "unit coordinate of largest-norm output row",
                }
            )
    if not menu:
        raise ValueError("No supported small MLP matrix found in this model")
    return menu


def proposal_prompt(practice, predictions, choices, history):
    practice_rows = [
        {"instruction": s.instruction, "expected": s.target, "your_prediction": p}
        for s, p in zip(practice, predictions)
    ]
    return (
        "You are choosing a bounded numeric experiment on YOUR OWN current neural-network weights. "
        "The proposer and target are the same model. Choose one menu label, nothing else. "
        "Coordinate names do not reveal semantics; improvement is uncertain. The trusted host will "
        "measure separate validation data, reject regressions/plateaus and restore rejected edits. "
        "Final-test questions/answers are not available to you.\n"
        f"Practice only: {json.dumps(practice_rows)}\n"
        f"Allowed menu: {json.dumps(choices)}\n"
        f"Previous choices/decisions: {json.dumps(history)}\nChoice:"
    )


def run_self_edit(backend, dataset, policy, output_dir, *, provenance, license_name, source=None):
    """Run a small model-selected experiment; accepted chains are used in-memory then undone.

    This function never saves a full model. Accepted patches are separately replayable in order.
    Provided backend callbacks remain trusted. Hard termination cannot write a restoration report.
    """
    import torch

    if not dataset.test:
        raise ValueError("A separate final-test split is required")
    if not provenance.strip() or not license_name.strip():
        raise ValueError("Model license and provenance are required")
    model = backend.model
    if sum(p.numel() for p in model.parameters()) > policy.max_model_parameters:
        raise ValueError("Loaded model exceeds parameter budget")
    root = Path(output_dir)
    if root.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    directory = root / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    report_path = directory / "self_edit.json"
    started = time.monotonic()
    before = model_fingerprint(model)
    modes = [(module, module.training) for module in model.modules()]
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
    report = {
        "schema_version": 1,
        "kind": "model-selected-self-edit",
        "status": "running",
        "run_id": directory.name,
        "policy": asdict(policy),
        "source": source,
        "model": backend.describe(),
        "provenance": provenance,
        "license": license_name,
        "dataset_sha256": dataset.fingerprint(),
        "selection_method": "constrained-next-token-choice",
        "proposer_is_target_model": True,
        "base_state_sha256": before,
        "rounds": [],
        "accepted_chain": [],
        "base_restored": False,
        "final_test": {"status": "pending", "used_for_promotion": False},
    }
    chain, attempted = [], set()
    atomic_json(report_path, report)
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            report["baseline"] = evaluate_samples(backend.predict, dataset.validation)
            atomic_json(report_path, report)
            menu = edit_menu(model, policy)
            for round_number in range(1, policy.max_rounds + 1):
                if time.monotonic() - started >= policy.max_seconds:
                    report["status"] = "budget_exhausted"
                    break
                available = [item for i, item in enumerate(menu) if i not in attempted]
                if not available:
                    break
                choices = {chr(66 + i): item for i, item in enumerate(available)}
                if not policy.require_attempt:
                    choices["A"] = {"action": "abstain"}
                practice = dataset.train[:2]
                practice_result = evaluate_samples(backend.predict, practice)
                prior = [
                    {
                        "choice": r["proposal"]["choice"],
                        "edit": r["plan"],
                        "status": r["trial"]["status"],
                        "reason": r["trial"].get("reason"),
                    }
                    for r in report["rounds"]
                ]
                prompt = proposal_prompt(practice, practice_result["predictions"], choices, prior)
                proposal = backend.choose(prompt, sorted(choices))
                if proposal.get("choice") not in choices:
                    raise ValueError("Proposer returned a choice outside the allowed menu")
                if proposal["choice"] == "A":
                    report.update(status="abstained", abstention=proposal)
                    break
                item = choices[proposal["choice"]]
                attempted.add(menu.index(item))
                direction = np.zeros(item["shape"][0], dtype=np.float64)
                direction[item["row"]] = 1
                plan = WeightEditPlan(
                    item["parameter_name"],
                    strength=item["strength"],
                    seed=policy.seed,
                    edited_fraction=policy.edited_fraction,
                    max_relative_norm=policy.max_relative_norm,
                    forgetting_threshold=policy.forgetting_threshold,
                    improvement_threshold=policy.improvement_threshold,
                )
                trial = run_weight_edit_trial(
                    model,
                    backend.predict,
                    dataset.validation,
                    direction,
                    plan,
                    directory / "trials",
                    provenance=provenance,
                    license_name=license_name,
                    model_info=backend.describe(),
                    reference_scores=report["baseline"]["scores"],
                )
                record = {
                    "round": round_number,
                    "menu": choices,
                    "proposal": proposal,
                    "practice": practice_result,
                    "plan": item,
                    "trial": trial,
                }
                report["rounds"].append(record)
                if trial["passed"]:
                    apply_weight_patch(model, trial["patch"])
                    chain.append(trial["patch"])
                    report["accepted_chain"] = list(chain)
                atomic_json(report_path, report)
            if report["status"] == "running":
                report["status"] = "completed"
            report["selected_state_sha256"] = model_fingerprint(model)
            report["final_test"]["status"] = "running"
            atomic_json(report_path, report)
            report["final_test"] = {
                "status": "completed",
                "used_for_promotion": False,
                **evaluate_samples(backend.predict, dataset.test),
            }
    except BaseException as exc:
        report.update(
            status="interrupted" if not isinstance(exc, Exception) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        if report["final_test"]["status"] == "pending":
            report["final_test"]["status"] = "skipped_after_failure"
        elif report["final_test"]["status"] == "running":
            report["final_test"]["status"] = (
                "failed" if isinstance(exc, Exception) else "interrupted"
            )
        raise
    finally:
        # Patches are undone in reverse order, including when final evaluation is interrupted.
        # Caller Torch RNG is already restored by fork_rng before this finalization.
        try:
            for patch in reversed(chain):
                apply_weight_patch(model, patch, reverse=True)
            after = model_fingerprint(model)
            report.update(restored_state_sha256=after, base_restored=after == before)
            if after != before:
                report.update(status="failed", error="Full state fingerprint did not restore")
        except BaseException as exc:
            report.update(
                status="failed",
                base_restored=False,
                restoration_error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            for module, training in modes:
                module.training = training
            report["elapsed_seconds"] = time.monotonic() - started
            atomic_json(report_path, report)
    if not report["base_restored"]:
        raise RuntimeError("Model restoration failed; see self-edit report")
    return {**report, "report_path": str(report_path.resolve())}


def run_configured_self_edit(config):
    from .backends.local_model import LocalCausalLM

    source = preflight_model(config)
    import torch

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(config["policy"].cpu_threads)
        backend = LocalCausalLM(config["model"])
        result = run_self_edit(
            backend,
            config["dataset"],
            config["policy"],
            config["output_dir"],
            provenance=config["provenance"],
            license_name=config["license_name"],
            source=source,
        )
        from .core.checkpoints import file_sha256

        model_path = Path(config["model"].model_name_or_path)
        result["source_files_unchanged"] = all(
            file_sha256(model_path / name) == expected
            for name, expected in backend.base_model_files.items()
        )
        if not result["source_files_unchanged"]:
            result["status"] = "failed"
        atomic_json(result["report_path"], result)
        if not result["source_files_unchanged"]:
            raise RuntimeError("Model source files changed during the experiment")
        return result
    finally:
        torch.set_num_threads(previous)
