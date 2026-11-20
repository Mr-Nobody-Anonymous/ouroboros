"""Bounded observe→diagnose→hypothesize→design→train→independent gates→record state machine.

The controller is ordinary trusted Python. Research models only propose data. Baselines are
never overwritten, and a crashed/invalid proposer never acquires host-written fallback actions.
"""

import contextlib
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from ..core.checkpoints import file_sha256, verify_checkpoint
from ..lab.config import strict_json
from ..tensor_inspection import inspect_safetensors
from .contracts import Budget, Proposal, Recipe, from_dict, identity, load_config
from .evaluation import choose_practice, diagnosis, gate, retention_gate
from .memory import TERMINAL, ExperimentMemory
from .provenance import (
    assert_software,
    model_inventory,
    regular,
    software_inventory,
    verify_binding,
    write_once,
)
from .researcher import host_proposal, rank_recipes, reference_recipe


def preflight(config, data, *, check_model=True):
    budget = from_dict(Budget, config["budget"])
    if config["seed"] + budget.repeats >= 2**31:
        raise ValueError("Replication seeds exceed the integer range")
    if config["model"].get("max_new_tokens", 64) > budget.max_new_tokens:
        raise ValueError("Model generation exceeds the fixed token budget")
    if not isinstance(config["data"], dict) or len(data.roles["train"]) > budget.max_train_examples:
        raise ValueError("Training cohort exceeds the fixed budget")
    for rows in data.roles.values():
        if len(rows) > 3000 or any(len(s.instruction) + len(s.target) > 16000 for s in rows):
            raise ValueError("Evaluation/example text exceeds the bounded foundation profile")
    for value in config["data"].values():
        regular(value)
    output, database = Path(config["output_dir"]), Path(config["database"])
    model_paths = [Path(config["model"]["model_name_or_path"])]
    if config.get("teacher"):
        model_paths.append(Path(config["teacher"]["model_name_or_path"]))
    if config["researcher"]["kind"] == "local_model":
        model_paths.append(Path(config["researcher"]["model_path"]))
    for target in (output, database):
        if target.is_symlink() or any(p.is_symlink() for p in target.parents):
            raise ValueError("Research output/database paths must not be symlinks")
        if any(target.resolve().is_relative_to(p.resolve()) for p in model_paths):
            raise ValueError("Outputs/database cannot be placed inside immutable model inputs")
        if any(target.resolve() == Path(p).resolve() for p in config["data"].values()):
            raise ValueError("Output/database collides with an input dataset")
    inventories, counts = {}, {}
    if check_model:
        for directory in model_paths:
            inventory = model_inventory(directory)
            parameters = 0
            for name in inventory:
                if name.endswith(".safetensors"):
                    value = inspect_safetensors(directory / name, limit=1000)
                    if value["truncated"]:
                        raise ValueError(
                            "Model tensor metadata exceeds the bounded inspection profile"
                        )
                    parameters += sum(row["numel"] for row in value["tensors"])
            if parameters > budget.max_parameters:
                raise ValueError(
                    "A target/researcher/teacher model exceeds the parameter budget before loading"
                )
            inventories[str(directory)], counts[str(directory)] = inventory, parameters
    model_path = config["model"]["model_name_or_path"]
    files, parameters = inventories.get(model_path), counts.get(model_path, 0)
    return {
        "schema_version": 1,
        "kind": "autonomous-research-foundation-check",
        "checked_only": True,
        "data_sha256": data.fingerprint(),
        "role_counts": {k: len(v) for k, v in data.roles.items()},
        "question_overlap_screen": data.quality,
        "model_files": files,
        "auxiliary_model_files": inventories,
        "stored_parameters": parameters,
        "budget": asdict(budget),
        "alpha": budget.alpha,
        "gates": "strict per-seed accuracy/significance and zero-domain forgetting; reserved/final once, no rescue",
        "researcher_kind": config["researcher"]["kind"],
        "loads_model": False,
        "is_os_sandbox": False,
        "does_not_establish_capability_gain": True,
    }


@contextlib.contextmanager
def execution_lock(database):
    import fcntl

    path = Path(str(database) + ".execution.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another foundation controller owns this database execution lock"
            ) from exc
        yield
    finally:
        os.close(descriptor)


def _stop(process):
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def _rss(pid):
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        return next(int(line.split()[1]) * 1024 for line in lines if line.startswith("VmRSS:"))
    except (OSError, StopIteration):
        return 0


def _tree_bytes(root):
    total = 0
    for path in root.rglob("*"):
        # Managed checkpoints rename temporary trees atomically while this watchdog runs.
        with contextlib.suppress(FileNotFoundError):
            if path.is_file():
                total += path.stat().st_size
    return total


def bind_exported_checkpoints(registry, trained):
    """Bind every exported checkpoint of one training run before any of them is measured.

    Intermediate selection needs several checkpoints from the same run; each is hashed here, so
    a later evaluator still refuses a swapped or edited candidate.
    """
    root = Path(trained["checkpoint"]).parent
    names = trained.get("checkpoints") or [Path(trained["checkpoint"]).name]
    bound = []
    for name in names:
        path = (root / name).resolve()
        if path.parent != root or path.is_symlink() or not path.is_dir():
            raise ValueError("Exported checkpoint escapes its training run")
        registry[str(path)] = identity(verify_checkpoint(str(path)))
        bound.append(str(path))
    return bound


class WorkerExecutor:
    def __init__(
        self,
        config,
        data_bindings,
        model_files,
        software,
        root,
        run_id,
        deadline,
        *,
        auxiliary_model_files=None,
    ):
        self.config, self.inputs, self.model_files, self.software = (
            config,
            data_bindings,
            model_files,
            software,
        )
        self.root, self.run_id, self.deadline = root, run_id, deadline
        self.auxiliary_model_files = auxiliary_model_files or {
            config["model"]["model_name_or_path"]: model_files
        }
        self.budget, self.count, self.planned_steps = from_dict(Budget, config["budget"]), 0, 0
        self.receipts = []
        self.planned_exports = 0
        self.checkpoints = {}

    def run(
        self,
        operation,
        *,
        inputs=None,
        recipe=None,
        checkpoint=None,
        seed=None,
        extra=None,
        model=None,
    ):
        remaining = math.floor(self.deadline - time.monotonic())
        if remaining < 1:
            raise TimeoutError("Whole research run exhausted its immutable wall budget")
        if operation in ("train", "followup"):
            amount = recipe.steps if operation == "train" else extra["followup_recipe"]["steps"]
            if self.planned_steps + amount > self.budget.max_total_steps:
                raise ValueError("Whole run would exceed its fixed total optimizer-step budget")
            exports = math.ceil(amount / self.budget.checkpoint_every)
            if self.planned_exports + exports > self.budget.max_checkpoint_exports:
                raise ValueError("Run would exceed its immutable checkpoint export budget")
            self.planned_steps += amount
            self.planned_exports += exports
        self.count += 1
        directory = self.root / "workers" / f"{self.count:03d}-{operation}"
        directory.mkdir(parents=True, exist_ok=False)
        output = directory / "result.json"
        model = model or self.config["model"]
        extra = dict(extra or {})
        if checkpoint:
            expected = self.checkpoints.get(str(checkpoint))
            if expected is None or identity(verify_checkpoint(checkpoint)) != expected:
                raise ValueError("Candidate no longer matches its measured checkpoint binding")
            extra["checkpoint_manifest_sha256"] = expected
        job = {
            "schema_version": 1,
            "operation": operation,
            "budget": asdict(replace(self.budget, max_seconds=remaining)),
            "model": model,
            "model_files": self.auxiliary_model_files[model["model_name_or_path"]],
            "software": self.software,
            "seed": self.config["seed"] if seed is None else seed,
            "inputs": inputs or {},
            "output": str(output),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "recipe": asdict(recipe) if recipe else None,
            "root": str(directory / "training"),
            "run_id": self.run_id,
            "extra": extra or {},
        }
        binding = write_once(directory / "job.json", job)
        started, peak = time.monotonic(), 0
        receipt = {
            "operation": operation,
            "job": binding,
            "worker_root": str(directory),
            "status": "started",
            "seed": job["seed"],
        }
        self.receipts.append(receipt)
        process = None
        try:
            with (
                (directory / "stdout.log").open("xb") as stdout,
                (directory / "stderr.log").open("xb") as stderr,
            ):
                process = subprocess.Popen(
                    [sys.executable, "-m", "ouroboros.research.worker", binding["file"]],
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                    env={
                        **os.environ,
                        "HF_HUB_OFFLINE": "1",
                        "HF_DATASETS_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "TOKENIZERS_PARALLELISM": "false",
                        "OMP_NUM_THREADS": str(self.budget.cpu_threads),
                        "MKL_NUM_THREADS": str(self.budget.cpu_threads),
                    },
                )
                last_disk_check = 0.0
                while process.poll() is None:
                    peak = max(peak, _rss(process.pid))
                    if time.monotonic() >= self.deadline:
                        raise TimeoutError("Whole-run wall budget reached; worker stopped")
                    if peak > self.budget.max_rss_bytes:
                        raise MemoryError("Worker exceeded its RSS watchdog budget; worker stopped")
                    if time.monotonic() - last_disk_check > 0.5:
                        if _tree_bytes(self.root) > self.budget.max_output_bytes:
                            raise OSError(
                                "Run exceeded its output-bytes watchdog budget; worker stopped"
                            )
                        last_disk_check = time.monotonic()
                    try:
                        process.wait(timeout=0.05)
                    except subprocess.TimeoutExpired:
                        continue
            if process.returncode:
                detail = (directory / "stderr.log").read_text(errors="replace")[-3000:]
                raise RuntimeError(f"{operation} worker exited {process.returncode}: {detail}")
            if _tree_bytes(self.root) > self.budget.max_output_bytes:
                raise OSError("Run exceeded its output-bytes budget; no promotion")
            result = strict_json(regular(output).read_text())
            if not result.get("worker", {}).get("source_files_unchanged"):
                raise RuntimeError("Worker did not confirm source integrity")
            if operation in ("train", "followup"):
                checkpoint_path = (
                    result["checkpoint"]
                    if operation == "train"
                    else result["training"]["checkpoint"]
                )
                self.checkpoints[checkpoint_path] = identity(verify_checkpoint(checkpoint_path))
            receipt.update(status="completed", result_sha256=None)
        except BaseException as exc:
            receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if process is not None:
                _stop(process)
                receipt["exit_code"] = process.returncode
            receipt.update(seconds=time.monotonic() - started, peak_rss_bytes_observed=peak)
            if output.is_file():
                receipt["result_sha256"] = file_sha256(output)
            write_once(directory / "receipt.json", receipt)
        assert_software(self.software)
        return result


def _candidate_log(report, sealed_index):
    """Every trained candidate, including rejected ones, for offline policy replay."""
    baseline = report.get("baseline_practice", {}).get("macro_accuracy")
    if baseline is None or not report.get("candidates"):
        return []
    rows = []
    for index, candidate in enumerate(report["candidates"]):
        scores = [row["practice"]["macro_accuracy"] for row in candidate["replications"]]
        if not scores:
            continue
        rows.append(
            {
                "family": candidate["recipe"]["family"],
                "recipe_sha256": identity(candidate["recipe"]),
                "practice": min(scores),
                "baseline": baseline,
                "selected": index == sealed_index,
                "gate_passed": bool(index == sealed_index and report["full_gate_passed"]),
            }
        )
    return rows


def run_research(config_path, *, check_only=False):
    config, data = load_config(config_path)
    checked = preflight(config, data)
    if check_only:
        return checked
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "Foundation resource supervision currently requires Linux /proc and process groups"
        )
    with execution_lock(config["database"]):
        return _run(config, data, checked, config_path)


def _run(config, data, checked, config_path):
    budget = from_dict(Budget, config["budget"])
    started = time.monotonic()
    run_id = "research-" + uuid.uuid4().hex
    root = Path(config["output_dir"]) / run_id
    root.mkdir(parents=True, exist_ok=False)
    software = software_inventory()
    bindings = {
        role: write_once(root / "inputs" / f"{role}.json", [s.as_dict() for s in rows])
        for role, rows in data.roles.items()
    }
    files = checked["model_files"]
    original_inputs = {
        str(regular(path)): file_sha256(path) for path in [config_path, *config["data"].values()]
    }
    context_id = identity(
        {
            "base": files,
            "domains": sorted({s.domain for s in data.roles["train"]}),
            "prompt_format": config["model"].get("prompt_format", "plain"),
        }
    )
    plan = {
        "schema_version": 1,
        "kind": "autonomous-research-foundation-plan",
        "run_id": run_id,
        "config": config,
        "config_file_sha256": file_sha256(config_path),
        "original_input_sha256": original_inputs,
        "question_overlap_screen": data.quality,
        "auxiliary_model_files": checked["auxiliary_model_files"],
        "data_sha256": data.fingerprint(),
        "inputs": bindings,
        "model_files": files,
        "software": software,
        "baseline_id": identity({"model_files": files}),
        "seeds": [config["seed"] + i for i in range(budget.repeats)],
        "selection": "One recipe chosen by worst-seed practice accuracy; seal before validation. Never select a lucky seed.",
        "checkpoint_selection": config["checkpoint_selection"],
        "gates": {
            "alpha": budget.alpha,
            "family_size": budget.family_size,
            "strict_accuracy_and_significance_each_seed": True,
            "zero_domain_forgetting": True,
            "reserved_final_retention_once_per_prompt_in_database": True,
        },
        "controls": ["no_edit", "host_reference_sft", "matched_followup_base"],
        "followup_recipe": config["followup_recipe"],
        "deadline_seconds": budget.max_seconds,
        "role_separation_is_not_os_isolation": True,
    }
    plan_binding = write_once(root / "plan.json", plan)
    memory = ExperimentMemory(config["database"])
    executor = WorkerExecutor(
        config,
        bindings,
        files,
        software,
        root,
        run_id,
        started + budget.max_seconds,
        auxiliary_model_files=checked["auxiliary_model_files"],
    )
    report = {
        "schema_version": 1,
        "kind": "autonomous-research-foundation",
        "run_id": run_id,
        "plan": plan_binding,
        "data_sha256": data.fingerprint(),
        "researcher_kind": config["researcher"]["kind"],
        "full_gate_passed": False,
        "promoted_checkpoints": [],
        "gates": {},
        "candidates": [],
        "autonomous_improvement_demonstrated": False,
        "improved_future_selection_demonstrated": False,
        "baseline_files_never_overwritten": True,
        "worker_separation_is_not_os_sandbox": True,
    }
    family, state = "none", "failed"
    sealed_index = None
    registered = False
    try:
        memory.register_baseline({"model_files": files})
        memory.register(run_id, plan)
        registered = True
        memory.register_roles(run_id, {role: data.role_keys(role) for role in data.roles})
        memory.transition(run_id, "observing")
        observed = executor.run("observe", inputs={"practice": bindings["practice"]})
        baseline = observed["evaluation"]
        memory.transition(run_id, "diagnosing", {"observation_sha256": identity(observed)})
        symptoms = diagnosis(baseline)
        report["baseline_practice"] = baseline["summary"]
        context = {
            "diagnosis": symptoms,
            "introspection": observed["introspection"],
            "train_domains": sorted({s.domain for s in data.roles["train"]}),
            "practice_examples": [
                {"instruction": s.instruction[:200], "reference": s.target[:100]}
                for s in data.roles["practice"][:2]
            ],
            "history": [
                {key: row[key] for key in ("context", "family", "useful", "cost")}
                for row in memory.search(limit=4)
            ],
            "baseline_sha256": identity(files),
        }
        write_once(root / "researcher_context.json", context)
        memory.transition(run_id, "researching", {"practice_context_sha256": identity(context)})
        reference = reference_recipe(observed["introspection"], budget)
        if config["researcher"]["kind"] == "host_reference":
            proposal = host_proposal(context, budget)
            report["proposal_origin"] = "host-authored reference; not autonomous research"
        else:
            researcher = config["researcher"]
            model = {**config["model"], "model_name_or_path": researcher["model_path"]}
            generated = executor.run(
                "research",
                model=model,
                extra={"context": context, "max_new_tokens": researcher.get("max_new_tokens", 256)},
            )
            report["proposal_origin"] = (
                "same-target-model"
                if generated["researcher"]["base_model_files"] == files
                else "external-research-model-assisted"
            )
            if not generated["valid"]:
                report["invalid_model_proposal"] = generated["error"]
                proposal = Proposal(
                    1, "abstain", "Invalid model output retained; no host fallback was executed."
                )
            else:
                proposal = Proposal.parse(generated["proposal"])
        proposal.validate_context(
            [e["id"] for e in symptoms["evidence"]], budget, context["train_domains"]
        )
        write_once(root / "proposal.json", asdict(proposal))
        for h in proposal.hypotheses:
            memory.claim(
                run_id,
                h["explanation"],
                "unverified research hypothesis; " + report["proposal_origin"],
            )
        if proposal.action == "abstain":
            state = "abstained"
            report["reason"] = proposal.reason
        else:
            recipes = [from_dict(Recipe, raw) for raw in proposal.recipes]
            ranking = rank_recipes(recipes, memory, context_id)
            selected_recipes = [
                from_dict(Recipe, row["recipe"]) for row in ranking[: budget.max_candidates]
            ]
            family = selected_recipes[0].family
            design = {
                "proposal_sha256": identity(asdict(proposal)),
                "ranking": ranking,
                "recipes": [asdict(r) for r in selected_recipes],
                "host_control_recipe": asdict(reference),
                "seeds": plan["seeds"],
                "gates": plan["gates"],
                "followup_recipe": plan["followup_recipe"],
                "selection_uses_practice_only": True,
                "sealed_before_any_candidate_training": True,
            }
            design_binding = write_once(root / "design.json", design)
            memory.transition(run_id, "designed", {"design": design_binding})
            memory.transition(run_id, "executing")
            # Optional verified distillation sees only training prompts/references; freeze before SFT.
            training = bindings["train"]
            distilled = None
            if any(r.family == "distill_sft" for r in selected_recipes):
                if not config.get("teacher"):
                    raise ValueError(
                        "distill_sft requires an explicitly configured independent local teacher"
                    )
                distillation = executor.run(
                    "distill", inputs={"train": training}, model=config["teacher"]
                )
                report["distillation"] = {
                    "accepted": distillation["accepted"],
                    "attempted_prompts": distillation["attempted_prompts"],
                }
                if not distillation["accepted"]:
                    raise ValueError(
                        "No independently verified teacher completions; refusing unverified distillation"
                    )
                distilled = distillation["corpus"]
            cache = {}

            def train_and_practice(recipe, seed, cohort_binding):
                weighted_count = sum(
                    recipe.domain_weights.get(s.domain, 1) for s in data.roles["train"]
                )
                if weighted_count > budget.max_train_examples:
                    raise ValueError("Weighted training exceeds the predeclared example budget")
                key = identity(
                    {"recipe": asdict(recipe), "seed": seed, "data": cohort_binding["sha256"]}
                )
                if key in cache:
                    return {**cache[key], "reused_exact_host_control": True}
                trained = executor.run(
                    "train", inputs={"train": cohort_binding}, recipe=recipe, seed=seed
                )
                rule = config["checkpoint_selection"]
                registry = getattr(executor, "checkpoints", None)
                exported = (
                    bind_exported_checkpoints(registry, trained)
                    if registry is not None
                    else [trained["checkpoint"]]
                )
                # Practice-only intermediate selection, declared in the config before the run.
                # Ties fall back to the earliest checkpoint: fewer steps and less overfitting,
                # never the luckiest late draw. The measured evaluation object is kept intact so
                # the same admission/validation code applies as for a single final checkpoint.
                if rule == "practice_macro_accuracy" and len(exported) > 1:
                    scored = [
                        {
                            "step_order": order,
                            "checkpoint": path,
                            "evaluation": executor.run(
                                "evaluate",
                                inputs={"evaluation": bindings["practice"]},
                                checkpoint=path,
                            ),
                        }
                        for order, path in enumerate(exported)
                    ]
                    chosen = max(
                        scored,
                        key=lambda row: (
                            row["evaluation"]["summary"]["macro_accuracy"],
                            -row["step_order"],
                        ),
                    )
                    checkpoint = chosen["checkpoint"]
                    evaluation = chosen["evaluation"]
                    selection_record = {
                        "rule": rule,
                        "chosen_step_order": chosen["step_order"],
                        "evaluated": len(scored),
                        "candidates": [
                            {
                                "step_order": row["step_order"],
                                "checkpoint": Path(row["checkpoint"]).name,
                                "macro_accuracy": row["evaluation"]["summary"]["macro_accuracy"],
                            }
                            for row in scored
                        ],
                    }
                else:
                    checkpoint = trained["checkpoint"]
                    evaluation = executor.run(
                        "evaluate",
                        inputs={"evaluation": bindings["practice"]},
                        checkpoint=checkpoint,
                    )
                    selection_record = {
                        "rule": "final_only",
                        "chosen_step_order": len(exported) - 1,
                        "evaluated": 1,
                    }
                marker = verify_checkpoint(checkpoint)
                result = {
                    "training": trained,
                    "evaluation": evaluation,
                    "checkpoint_manifest_sha256": identity(marker),
                    "checkpoint_selection": selection_record,
                    "reused_exact_host_control": False,
                }
                cache[key] = result
                return result

            host_control = [train_and_practice(reference, seed, training) for seed in plan["seeds"]]
            report["host_control_practice"] = [r["evaluation"]["summary"] for r in host_control]
            candidate_rows = []
            for recipe in selected_recipes:
                rows = [
                    train_and_practice(
                        recipe, seed, distilled if recipe.family == "distill_sft" else training
                    )
                    for seed in plan["seeds"]
                ]
                candidate_rows.append(rows)
                predicted = proposal.hypotheses[recipe.hypothesis_index]["prediction"]
                deltas = [
                    r["evaluation"]["summary"]["macro_accuracy"]
                    - baseline["summary"]["macro_accuracy"]
                    for r in rows
                ]
                prediction_audit = {
                    "hypothesis_index": recipe.hypothesis_index,
                    "predicted_practice_accuracy_delta": predicted,
                    "observed_deltas_each_seed": deltas,
                    "mean_prediction_error": sum(deltas) / len(deltas) - predicted,
                    "causal_explanation_established": False,
                }
                report["candidates"].append(
                    {
                        "recipe": asdict(recipe),
                        "prediction_audit": prediction_audit,
                        "replications": [
                            {
                                "checkpoint": row["training"]["checkpoint"],
                                "selected_checkpoint": row["checkpoint_selection"][
                                    "chosen_step_order"
                                ],
                                "checkpoint_selection": row["checkpoint_selection"],
                                "training": row["training"],
                                "practice": row["evaluation"]["summary"],
                                "reused_exact_host_control": row["reused_exact_host_control"],
                            }
                            for row in rows
                        ],
                    }
                )
            index = choose_practice(
                baseline, [[r["evaluation"] for r in rows] for rows in candidate_rows]
            )
            sealed_index = index
            # Seal the exact recipe/all seed checkpoints BEFORE even validation inference.
            frozen = {
                "selected_recipe_index": index,
                "checkpoints": [r["training"]["checkpoint"] for r in candidate_rows[index]]
                if index is not None
                else [],
                "checkpoint_manifest_sha256": [
                    r["checkpoint_manifest_sha256"] for r in candidate_rows[index]
                ]
                if index is not None
                else [],
                "checkpoint_selection_rule": config["checkpoint_selection"],
                "checkpoint_selection": [
                    row["checkpoint_selection"] for row in candidate_rows[index]
                ]
                if index is not None
                else [],
                "candidate_selection_finished": True,
                "future_reselection_allowed": False,
            }
            selection = write_once(root / "sealed_selection.json", frozen)
            memory.transition(run_id, "validating", {"selection": selection})
            state = "rejected"
            report["reason"] = (
                "No candidate strictly improved practice at every fixed seed; hidden gates unopened."
            )
            if index is not None:
                recipe = selected_recipes[index]
                family = recipe.family
                checkpoints = frozen["checkpoints"]
                report["selected_recipe"] = asdict(recipe)
                allowed = True
                for role, next_state in (
                    ("validation", "reserved"),
                    ("reserved", "final"),
                    ("final", "robustness"),
                    ("robustness", "retention"),
                ):
                    if not allowed:
                        break
                    if role != "validation":
                        memory.consume_holdout(run_id, role, data.role_keys(role))
                    before = executor.run("evaluate", inputs={"evaluation": bindings[role]})
                    after = [
                        executor.run(
                            "evaluate", inputs={"evaluation": bindings[role]}, checkpoint=checkpoint
                        )
                        for checkpoint in checkpoints
                    ]
                    measured_gate = gate(
                        before,
                        after,
                        alpha=budget.alpha,
                        seed=config["seed"],
                        require_improvement=role != "robustness",
                    )
                    report["gates"][role] = measured_gate
                    allowed = measured_gate["passed"]
                    if allowed:
                        memory.transition(
                            run_id, next_state, {"gate_sha256": identity(measured_gate)}
                        )
                if allowed:
                    memory.consume_holdout(run_id, "retention", data.role_keys("retention"))
                    before = executor.run("evaluate", inputs={"evaluation": bindings["retention"]})
                    after = [
                        executor.run(
                            "evaluate", inputs={"evaluation": bindings["retention"]}, checkpoint=c
                        )
                        for c in checkpoints
                    ]
                    base_followups, candidate_followups = [], []
                    extra = {
                        "original_recipe": asdict(recipe),
                        "followup_recipe": config["followup_recipe"],
                        "evaluation_seed": config["seed"],
                    }
                    for seed, checkpoint in zip(plan["seeds"], checkpoints):
                        for target, destination in (
                            (None, base_followups),
                            (checkpoint, candidate_followups),
                        ):
                            trained = executor.run(
                                "followup",
                                inputs={"followup": bindings["followup"]},
                                checkpoint=target,
                                seed=seed,
                                extra=extra,
                            )
                            destination.append(
                                executor.run(
                                    "evaluate",
                                    inputs={"evaluation": bindings["retention"]},
                                    checkpoint=trained["training"]["checkpoint"],
                                )
                            )
                    retention = retention_gate(
                        before,
                        after,
                        base_followups,
                        candidate_followups,
                        alpha=budget.alpha,
                        seed=config["seed"],
                    )
                    report["gates"]["retention"] = retention
                    allowed = retention["passed"]
                if allowed:
                    state = "completed"
                    report.update(
                        full_gate_passed=True,
                        promoted_checkpoints=checkpoints,
                        reason="All fixed independent gates and matched follow-up retention passed; original base remains immutable.",
                    )
                    # Real model origin does not alone demonstrate autonomous general research ability.
                    report["target_improvement_scope"] = (
                        "this fixed local task and recipe, under these declared seeds/gates only"
                    )
                else:
                    report["reason"] = (
                        "Independent gate rejected the sealed candidate; no alternative or practice/loss rescue."
                    )
    except BaseException as exc:
        state = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        report.update(
            reason=f"{type(exc).__name__}: {exc}", full_gate_passed=False, promoted_checkpoints=[]
        )
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            report["operator_interruption"] = True
    finally:
        try:
            unchanged = all(
                model_inventory(path) == expected
                for path, expected in checked["auxiliary_model_files"].items()
            )
            assert_software(software)
            unchanged = unchanged and all(
                file_sha256(regular(path)) == expected for path, expected in original_inputs.items()
            )
            unchanged = unchanged and all(
                file_sha256(verify_binding(b)) == b["sha256"] for b in bindings.values()
            )
        except (OSError, ValueError):
            unchanged = False
        report.update(
            state=state,
            source_files_unchanged=unchanged,
            baseline_files_never_overwritten=unchanged,
            workers=executor.receipts,
            planned_optimizer_steps_consumed=executor.planned_steps,
            seconds=time.monotonic() - started,
            rollback="all worker processes ended; candidate checkpoints retained as evidence, never applied to baseline",
        )
        if not unchanged:
            report.update(
                state="failed",
                full_gate_passed=False,
                promoted_checkpoints=[],
                reason="Frozen source/input integrity check failed; no promotion.",
            )
            state = "failed"
        artifact_hashes = {
            str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*")) if p.is_file()
        }
        inventory = write_once(root / "artifacts.json", artifact_hashes)
        report["artifacts"] = inventory
        report_binding = write_once(root / "research.json", report)
        if registered and memory.state(run_id) not in TERMINAL:
            summary = {
                "report": report_binding,
                "state": state,
                "full_gate_passed": report["full_gate_passed"],
                "researcher_kind": config["researcher"]["kind"],
                "reason": report["reason"],
                "capability_claim": "No autonomous general self-improvement or improved future selection established.",
            }
            result = {
                "context": context_id,
                "family": family,
                "useful": report["full_gate_passed"],
                "cost": float(report["seconds"]),
                "summary": summary,
            }
            memory.transition(
                run_id, state, {"result_sha256": identity(result), "report": report_binding}
            )
            # Counterfactual log: needed to ask later whether the ranking policy chose well.
            memory.record_candidates(
                run_id, _candidate_log(report, sealed_index), context=context_id
            )
            memory.record_result(run_id, **result)
        memory.verify()
        memory.close()
    return {**report, "report_file": str(root / "research.json")}


def run_campaign(paths, *, max_cycles):
    if type(max_cycles) is not int or not 1 <= max_cycles <= 8 or not 1 <= len(paths) <= max_cycles:
        raise ValueError(
            "Campaign requires 1–8 explicit configs within max_cycles; no unbounded loop"
        )
    # All plans/data fixed before the first target runs. Fresh hidden prompts and one database.
    loaded = [load_config(p) for p in paths]
    if len({c["database"] for c, _ in loaded}) != 1:
        raise ValueError("Campaign cycles must share the authoritative memory/holdout ledger")
    hidden = set()
    for config, data in loaded:
        preflight(config, data)
        if from_dict(Budget, config["budget"]).family_size < max(
            44, sum(from_dict(Budget, c["budget"]).repeats * 3 for c, _ in loaded)
        ):
            raise ValueError(
                "Family correction does not cover all predeclared campaign seed/gate checks"
            )
        keys = set().union(
            *(
                set(data.role_keys(role))
                for role in ("reserved", "final", "robustness", "retention")
            )
        )
        if keys & hidden:
            raise ValueError("Each campaign cycle needs fresh reserved/final/retention prompts")
        hidden |= keys
    all_roles = {}
    for _, data in loaded:
        for role in data.roles:
            for key in data.role_keys(role):
                if key in all_roles and all_roles[key] != role:
                    raise ValueError("Campaign prompt roles changed between cycles")
                all_roles[key] = role
    configs = [
        identity(
            {
                "config": c,
                "data": d.fingerprint(),
                "models": preflight(c, d)["auxiliary_model_files"],
            }
        )
        for c, d in loaded
    ]
    frozen_software = software_inventory()
    reports = []
    for path, binding in zip(paths, configs):
        config, data = load_config(path)
        assert_software(frozen_software)
        if (
            identity(
                {
                    "config": config,
                    "data": data.fingerprint(),
                    "models": preflight(config, data)["auxiliary_model_files"],
                }
            )
            != binding
        ):
            raise ValueError("Campaign config/data/model changed after its initial validation")
        report = run_research(path)
        reports.append(
            {
                "run_id": report["run_id"],
                "state": report["state"],
                "full_gate_passed": report["full_gate_passed"],
                "report_file": report["report_file"],
            }
        )
        if report["state"] in ("failed", "interrupted"):
            break
    return {
        "kind": "bounded-research-campaign",
        "reports": reports,
        "automatic_baseline_overwrite": False,
        "improved_future_selection_demonstrated": False,
    }
