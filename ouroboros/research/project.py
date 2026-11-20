"""Offline foundation scaffold and data-only artifact verification; no model loads/downloads."""

from dataclasses import asdict
from pathlib import Path

from ..core.checkpoints import file_sha256
from ..data import Sample
from ..lab.config import strict_json
from .contracts import ROLES, Budget, Recipe
from .provenance import regular, verify_binding, write_once


def init_foundation(directory, model_path):
    root = Path(directory).absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError("Foundation scaffold requires a new directory; no overwrite")
    if any(p.is_symlink() for p in root.parents):
        raise ValueError("Scaffold parent must not be symlinked")
    root.mkdir(parents=True)
    for role in ROLES:
        # Deliberately trivial constant-answer fixture, not a benchmark/generalization claim.
        values = [
            Sample(
                f"Return ready only. {role} record {index}",
                "ready",
                "fixture",
                "ouroboros-constant-answer-integration-fixture-v1",
                True,
            ).as_dict()
            for index in range(12)
        ]
        write_once(root / "data" / f"{role}.json", values)
    config = {
        "schema_version": 1,
        "model": {
            "model_name_or_path": str(Path(model_path).absolute()),
            "device": "cpu",
            "torch_dtype": "float32",
            "max_new_tokens": 4,
        },
        "data": {role: f"data/{role}.json" for role in ROLES},
        "budget": asdict(Budget()),
        "researcher": {"kind": "host_reference"},
        "output_dir": "runs",
        "database": "experiments.sqlite3",
        "seed": 17,
        "followup_recipe": asdict(Recipe(steps=2, learning_rate=0.0001)),
    }
    write_once(root / "foundation.json", config)
    (root / "README.md").write_text(
        "# Foundation integration fixture\n\n"
        "These eight prompt-disjoint constant-answer roles only exercise control flow. "
        "They are NOT an independent capability/generalization benchmark. "
        "Replace them with licensed verified domain data before making scientific claims.\n\n"
        "Run `ouroboros research run --config foundation.json --check-only` first. "
        "No model is downloaded. Real execution needs the optional train extra. "
        "Host-reference proposals are not autonomous research.\n",
        encoding="utf-8",
    )
    return {
        "directory": str(root),
        "config": str(root / "foundation.json"),
        "model_downloaded": False,
        "dataset_is_synthetic_fixture": True,
        "baseline_must_be_local": True,
    }


def verify_run(report_path, *, database=None):
    path = regular(report_path)
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("Foundation report exceeds the reader budget")
    report = strict_json(path.read_text())
    if report.get("kind") != "autonomous-research-foundation" or report.get("schema_version") != 1:
        raise ValueError("Expected a versioned foundation report")
    root = path.parent.resolve()
    inventory_path = verify_binding(report["artifacts"])
    if inventory_path.resolve() != root / "artifacts.json":
        raise ValueError("Artifact inventory escapes its run")
    inventory = strict_json(inventory_path.read_text())
    for name, expected in inventory.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe artifact inventory path")
        item = regular(root / relative)
        if file_sha256(item) != expected:
            raise ValueError(f"Research artifact checksum mismatch: {name}")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(inventory) | {"artifacts.json", path.name}:
        raise ValueError("Research artifact file inventory changed")
    plan_path = verify_binding(report["plan"])
    if plan_path.resolve() != root / "plan.json":
        raise ValueError("Plan binding escapes its run")
    from .memory import ExperimentMemory

    plan = strict_json(plan_path.read_text())
    memory = ExperimentMemory(database or plan["config"]["database"], readonly=True)
    try:
        row = memory.db.execute(
            "SELECT summary FROM results WHERE experiment_id=?", (report["run_id"],)
        ).fetchone()
        if row is None or strict_json(row[0])["report"]["sha256"] != file_sha256(path):
            raise ValueError("Report does not match its authoritative SQLite event-bound result")
    finally:
        memory.close()
    return {
        "status": "verified",
        "files": len(inventory),
        "run_id": report["run_id"],
        "state": report["state"],
        "full_gate_passed": report["full_gate_passed"],
        "report_bound_to_database": True,
        "note": "Integrity only, not identity authentication, inference replay or a reproduced benchmark.",
    }
