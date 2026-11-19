"""Atomic managed checkpoints with tensor hashes, lineage, and a run-scoped ownership marker."""

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

MARKER = ".ouroboros-checkpoint.json"
NAME = re.compile(r"generation_[0-9]{4,}\Z")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _safe_checkpoint(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("A real checkpoint directory is required, not a symlink")
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Checkpoint symlinks are not allowed")
    return directory


def verify_checkpoint(directory):
    directory = _safe_checkpoint(directory)
    marker = json.loads((directory / MARKER).read_text(encoding="utf-8"))
    if marker.get("schema_version") != 1 or not marker.get("run_id"):
        raise ValueError("Not an Ouroboros managed checkpoint")
    manifest = marker.get("files", {})
    actual = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p.name != MARKER
    }
    if not manifest or set(manifest) != actual:
        raise ValueError("Checkpoint file inventory does not match its manifest")
    for name, expected in manifest.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or file_sha256(directory / relative) != expected
        ):
            raise ValueError(f"Checkpoint checksum mismatch or unsafe path: {name}")
    return marker


class CheckpointStore:
    def __init__(self, root, run_id, min_free_bytes=0):
        self.root = Path(root)
        if self.root.is_symlink():
            raise ValueError("Checkpoint root cannot be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()
        self.run_id = run_id
        self.min_free_bytes = min_free_bytes

    def save(self, backend, generation, *, scores, parent=None, metadata=None):
        name = f"generation_{generation:04d}"
        if not NAME.fullmatch(name) or generation < 0:
            raise ValueError("Invalid generation number")
        target = self.root / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {target}")
        if shutil.disk_usage(self.root).free < self.min_free_bytes:
            raise OSError("Insufficient free disk space for a checkpoint")
        temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
        try:
            backend.save(temporary)
            _safe_checkpoint(temporary)
            files = {
                str(p.relative_to(temporary)): file_sha256(p)
                for p in sorted(temporary.rglob("*"))
                if p.is_file()
            }
            marker = {
                "schema_version": 1,
                "run_id": self.run_id,
                "generation": generation,
                "scores": scores,
                "parent": parent,
                "backend": backend.describe(),
                "metadata": metadata or {},
                "files": files,
            }
            atomic_json(temporary / MARKER, marker)
            verify_checkpoint(temporary)
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return name
