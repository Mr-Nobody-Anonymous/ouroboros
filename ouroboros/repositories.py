"""Auditable upstream source inventory. Cloning source never installs or executes it."""

import hashlib
import json
import os
import subprocess
from pathlib import Path


def git(root, *args, **kwargs):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        **kwargs,
    ).stdout.strip()


def snapshot_digest(root, relative_path):
    """Hash tracked vendored files, including names, so missing/edited files are detected."""
    names = git(root, "ls-files", "-z", "--", relative_path).split("\0")
    digest, count = hashlib.sha256(), 0
    for name in sorted(name for name in names if name):
        path = Path(root) / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or unsafe vendored source file: {name}")
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
        count += 1
    if count == 0:
        raise ValueError("No vendored source files were found")
    return digest.hexdigest(), count


def load_sources(root):
    root = Path(root).resolve()
    lock = json.loads((root / "external" / "sources.lock.json").read_text(encoding="utf-8"))
    for source in lock["sources"]:
        path = Path(source["path"])
        if path.is_absolute() or ".." in path.parts or path.parts[0] != "external":
            raise ValueError("Unsafe source path in lock file")
        if not source["url"].startswith("https://github.com/"):
            raise ValueError("Only HTTPS GitHub source URLs are allowed")
    return lock["sources"]


def verify_sources(root):
    root = Path(root).resolve()
    result = []
    for source in load_sources(root):
        row = {"name": source["name"], "path": source["path"], "mode": source["mode"]}
        try:
            path = root / source["path"]
            if path.is_symlink():
                raise ValueError("Source directory cannot be a symlink")
            if source["mode"] == "submodule":
                if Path(git(path, "rev-parse", "--show-toplevel")).resolve() != path.resolve():
                    raise ValueError("Submodule is not initialized")
                actual = git(path, "rev-parse", "HEAD")
                if actual != source["revision"]:
                    raise ValueError(f"Revision mismatch: {actual}")
                if git(path, "status", "--porcelain", "--untracked-files=no"):
                    raise ValueError("Source has local tracked changes")
            else:
                actual, _ = snapshot_digest(root, source["path"])
                if actual != source["snapshot_sha256"]:
                    raise ValueError("Vendored snapshot content mismatch")
            row.update(status="verified", revision=source.get("revision"))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            row.update(status="unavailable_or_modified", error=str(exc))
        result.append(row)
    return result


def sync_sources(root, names=()):
    root = Path(root).resolve()
    sources = load_sources(root)
    known = {s["name"] for s in sources}
    if set(names) - known:
        raise ValueError(f"Unknown sources: {sorted(set(names) - known)}")
    selected = [
        s for s in sources if s["mode"] == "submodule" and (not names or s["name"] in names)
    ]
    for source in selected:
        # Pin to the committed gitlink, never --remote; avoid pulling model-sized LFS objects.
        git(
            root,
            "submodule",
            "update",
            "--init",
            "--depth",
            "1",
            "--",
            source["path"],
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
        )
    return verify_sources(root)
