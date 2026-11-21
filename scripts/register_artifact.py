"""Register a real, reviewed file under artifacts/objects; does not stage, upload, or delete data."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register(source, *, license_name, provenance, root=ROOT):
    source, root = Path(source), Path(root).resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError("Source must be a real existing file, not a symlink")
    if any(not isinstance(value, str) or not value.strip() for value in (license_name, provenance)):
        raise ValueError("Explicit license and provenance are required")
    digest = sha256(source)
    relative = Path("artifacts/objects") / digest / source.name
    destination = root / relative
    manifest_path = root / "artifacts/manifest.json"
    for path in (destination, manifest_path):
        # Check ancestors before creating/copying anything. This is not hostile-writer isolation.
        if any(part.is_symlink() for part in (path, *path.parents) if part.is_relative_to(root)):
            raise ValueError("Artifact paths and their parents must not be symlinks")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("Invalid artifact manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256(destination) != digest:
            raise ValueError("Existing artifact path has different contents")
    elif source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    if sha256(destination) != digest:
        raise ValueError("Source changed during registration; no manifest entry was written")
    entry = {
        "path": relative.as_posix(),
        "sha256": digest,
        "bytes": destination.stat().st_size,
        "license": license_name,
        "provenance": provenance,
    }
    existing = [row for row in manifest["files"] if row["path"] == entry["path"]]
    if existing and existing != [entry]:
        raise ValueError("Existing manifest entry differs; review it explicitly")
    if not existing:
        manifest["files"].append(entry)
        fd, temporary = tempfile.mkstemp(prefix=".pending-manifest-", dir=manifest_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, manifest_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return relative.as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--license", required=True, dest="license_name")
    parser.add_argument("--provenance", required=True)
    args = parser.parse_args()
    path = register(args.path, license_name=args.license_name, provenance=args.provenance)
    print(
        f"Registered {path}; not uploaded. Review licensing/storage, install Git LFS, then stage explicitly."
    )


if __name__ == "__main__":
    main()
