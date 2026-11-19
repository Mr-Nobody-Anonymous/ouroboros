"""Non-destructive workspace inventory. Checkpoint pruning lives in core.pareto_pruner."""

import os
from pathlib import Path


def audit_storage(root, large_threshold=50 * 1024 * 1024):
    root = Path(root).resolve()
    if not root.is_dir() or large_threshold < 1:
        raise ValueError("Existing directory and positive threshold required")
    total, count, large, caches = 0, 0, [], []
    cache_names = {"__pycache__", ".pytest_cache", ".ruff_cache", ".cache"}
    for directory, dirs, files in os.walk(root, followlinks=False):
        # Do not scan or delete Git objects, virtualenvs, or upstream checkouts by default.
        dirs[:] = [
            name
            for name in dirs
            if name not in {".git", ".venv", "external", "node_modules"}
            and not (Path(directory) / name).is_symlink()
        ]
        path = Path(directory)
        if path.name in cache_names:
            caches.append(str(path.relative_to(root)))
        for name in files:
            file = path / name
            if file.is_symlink():
                continue
            size = file.stat().st_size
            count += 1
            total += size
            if size >= large_threshold:
                large.append({"path": str(file.relative_to(root)), "bytes": size})
    usage = __import__("shutil").disk_usage(root)
    return {
        "root": str(root),
        "scanned_files": count,
        "scanned_bytes": total,
        "free_disk_bytes": usage.free,
        "large_files": sorted(large, key=lambda f: -f["bytes"]),
        "cache_directories": sorted(caches),
        "deleted": [],
        "excluded": [".git", ".venv", "external", "node_modules"],
        "note": "Inventory only. No system, model, or dataset caches were deleted.",
    }
