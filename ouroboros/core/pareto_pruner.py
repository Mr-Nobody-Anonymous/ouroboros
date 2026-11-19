"""
Pareto Checkpoint Pruner: Manages storage overhead by pruning dominated generational checkpoints.
"""
import os
import shutil
from typing import Dict, Any, List, Optional

class ParetoCheckpointPruner:
    """
    Evaluates checkpoints on multiple objectives (e.g. Accuracy vs. Checkpoint Size)
    and removes strictly dominated or stale generational checkpoints to preserve disk space.
    """

    def __init__(self, max_checkpoints_to_keep: int = 2):
        self.max_checkpoints_to_keep = max_checkpoints_to_keep

    def prune_checkpoints(self, checkpoint_dir: str, history: List[Dict[str, Any]]) -> List[str]:
        """
        Keeps the top K non-dominated checkpoints and safely deletes inferior ones.
        """
        if not os.path.exists(checkpoint_dir):
            return []

        # Sort history by overall average score descending
        sorted_history = sorted(
            [h for h in history if h.get("checkpoint")],
            key=lambda x: sum(x["scores"].values()) / len(x["scores"]),
            reverse=True
        )

        keep_names = {h["checkpoint"] for h in sorted_history[:self.max_checkpoints_to_keep]}
        deleted = []

        for entry in os.listdir(checkpoint_dir):
            full_path = os.path.join(checkpoint_dir, entry)
            if os.path.isdir(full_path) and entry not in keep_names:
                try:
                    shutil.rmtree(full_path)
                    deleted.append(entry)
                    print(f"[ParetoPruner] Pruned dominated checkpoint to reclaim space: {entry}")
                except Exception as e:
                    print(f"[ParetoPruner] Could not prune {entry}: {e}")

        return deleted
