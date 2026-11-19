"""Conservative, run-scoped Pareto retention over measured accuracy and actual bytes."""

import math
import shutil
from pathlib import Path

from .checkpoints import NAME, verify_checkpoint


class ParetoCheckpointPruner:
    """Never removes unowned directories. Preview is the default; deletion requires apply=True."""

    def __init__(self, max_checkpoints_to_keep=2):
        if max_checkpoints_to_keep < 1:
            raise ValueError("At least one checkpoint must be retained")
        self.max_checkpoints_to_keep = max_checkpoints_to_keep

    def plan(self, checkpoint_dir, *, run_id, protected=()):
        root = Path(checkpoint_dir)
        if root.is_symlink():
            raise ValueError("Checkpoint root cannot be a symlink")
        if not root.exists():
            return {"keep": [], "delete": [], "reclaim_bytes": 0}
        candidates = []
        for path in sorted(root.iterdir()):
            if not NAME.fullmatch(path.name) or path.is_symlink() or not path.is_dir():
                continue
            try:
                marker = verify_checkpoint(path)
                scores = marker["scores"]
                if (
                    marker["run_id"] != run_id
                    or not scores
                    or any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values())
                ):
                    continue
                size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                candidates.append(
                    {
                        "name": path.name,
                        "bytes": size,
                        "accuracy": sum(scores.values()) / len(scores),
                    }
                )
            except (OSError, ValueError, KeyError, TypeError):
                # Corrupt/unrecognized artifacts are for manual inspection, not automatic deletion.
                continue
        frontier = []
        for point in candidates:
            dominated = any(
                other["accuracy"] >= point["accuracy"]
                and other["bytes"] <= point["bytes"]
                and (other["accuracy"] > point["accuracy"] or other["bytes"] < point["bytes"])
                for other in candidates
            )
            if not dominated:
                frontier.append(point)
        keep = {
            p["name"]
            for p in sorted(frontier, key=lambda p: (-p["accuracy"], p["bytes"], p["name"]))[
                : self.max_checkpoints_to_keep
            ]
        }
        # Safety/rollback points are retained even if that exceeds the nominal cap.
        keep.update(protected)
        delete = [p for p in candidates if p["name"] not in keep]
        return {
            "keep": sorted(p["name"] for p in candidates if p["name"] in keep),
            "delete": [p["name"] for p in delete],
            "reclaim_bytes": sum(p["bytes"] for p in delete),
        }

    def prune_checkpoints(
        self, checkpoint_dir, history=None, *, run_id=None, protected=(), apply=False
    ):
        if run_id is None:
            raise ValueError(
                "An explicit run_id is required; legacy unowned checkpoints are not pruned"
            )
        plan = self.plan(checkpoint_dir, run_id=run_id, protected=protected)
        if not apply:
            return plan["delete"]
        root = Path(checkpoint_dir).resolve()
        deleted = []
        for name in plan["delete"]:
            path = root / name
            # Recheck just before deleting. The run must own its checkpoint root exclusively.
            if path.is_symlink() or path.parent.resolve() != root:
                raise ValueError("Refusing unsafe checkpoint deletion")
            marker = verify_checkpoint(path)
            if marker["run_id"] != run_id:
                raise ValueError("Checkpoint ownership changed")
            shutil.rmtree(path)
            deleted.append(name)
        return deleted
