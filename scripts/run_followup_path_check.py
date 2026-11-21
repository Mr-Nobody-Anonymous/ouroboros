"""Machinery check for the matched follow-up / retention path on the real pretrained model.

NOT capability evidence: it replays the four-cell retention design on an already rejected run
so the downstream code path is exercised end to end. Every number here is labelled a path test.
"""

import json
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path

from ouroboros.config import ModelConfig
from ouroboros.data import Sample
from ouroboros.research.contracts import Budget, Recipe, from_dict, identity
from ouroboros.research.evaluation import retention_gate
from ouroboros.research.provenance import model_inventory
from ouroboros.research.training import load_target, train_audited
from ouroboros.research.worker import measure

ROOT = Path("outputs/g1-owned-01").absolute()
RUN = next((ROOT / "runs").iterdir())
raw = json.loads((ROOT / "foundation.json").read_text())
report = json.loads((RUN / "research.json").read_text())
budget = Budget(**raw["budget"])
model = from_dict(ModelConfig, raw["model"])
original = from_dict(Recipe, report["selected_recipe"])
effective = replace(
    from_dict(Recipe, raw["followup_recipe"]),
    family=original.family,
    rank=original.rank,
    alpha=original.alpha,
    target_modules=original.target_modules,
)
SOURCE = identity(model_inventory(model.model_name_or_path))
SEEDS = (17, 18)
EVAL_SEED = 17  # every retention measurement uses one fixed evaluation seed


def cohort(role):
    rows = json.loads((ROOT / "data" / f"{role}.json").read_text())
    return [Sample.from_dict(row) for row in rows]


retention, followup = cohort("retention"), cohort("followup")
work = ROOT / "plumbing"
work.mkdir(exist_ok=False)
checkpoints = [row["checkpoint"] for row in report["candidates"][0]["replications"]]
assert len(checkpoints) == len(SEEDS)
started = time.time()


def followup_cell(name, checkpoint, seed):
    backend = load_target(
        model, budget, recipe=None if checkpoint else original, seed=seed, checkpoint=checkpoint
    )
    trained = train_audited(
        backend,
        followup,
        effective,
        budget,
        seed=seed,
        root=work / name,
        run_id=f"plumbing-{name}",
        source_binding=SOURCE,
        deadline=time.monotonic() + budget.max_seconds,
    )
    after = measure(
        load_target(model, budget, seed=seed, checkpoint=trained["checkpoint"]),
        retention,
        budget,
        EVAL_SEED,
    )
    print(f"{name}: after {after['summary']['macro_accuracy']:.3f}", flush=True)
    return after


before_base = measure(
    load_target(model, budget, recipe=original, seed=SEEDS[0]), retention, budget, EVAL_SEED
)
before_candidates = [
    measure(load_target(model, budget, seed=seed, checkpoint=path), retention, budget, EVAL_SEED)
    for seed, path in zip(SEEDS, checkpoints)
]
base_after = [followup_cell(f"base-seed{seed}", None, seed) for seed in SEEDS]
candidate_after = [
    followup_cell(f"candidate-seed{seed}", path, seed) for seed, path in zip(SEEDS, checkpoints)
]

verdict = retention_gate(
    before_base, before_candidates, base_after, candidate_after, alpha=budget.alpha, seed=EVAL_SEED
)
result = {
    "kind": "followup-plumbing-check",
    "scientific_evidence": False,
    "non_claims": [
        "replays a rejected candidate; it cannot promote or rescue the G1 result",
        "no new prompts; the retention role is consumed by this replay",
        "exercises the matched follow-up cells and the retention gate only",
    ],
    "effective_followup_recipe": asdict(effective),
    "cells": {
        "before_base": before_base["summary"]["macro_accuracy"],
        "before_candidates": [row["summary"]["macro_accuracy"] for row in before_candidates],
        "after_base": [row["summary"]["macro_accuracy"] for row in base_after],
        "after_candidates": [row["summary"]["macro_accuracy"] for row in candidate_after],
    },
    "retention_gate_passed": verdict["passed"],
    "retention_gate": verdict,
    "seconds": round(time.time() - started, 1),
    "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
}
(work / "followup_report.json").write_text(json.dumps(result, indent=1, sort_keys=True))
print(json.dumps({k: result[k] for k in ("cells", "seconds", "peak_rss_mb")}, indent=1))
print("retention passed:", verdict["passed"])
