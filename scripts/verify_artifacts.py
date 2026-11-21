"""Check registered artifact bytes/checksums. An empty manifest is not an uploaded dataset."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest_path = root / "artifacts/manifest.json"
    if manifest_path.is_symlink() or manifest_path.parent.is_symlink():
        raise ValueError("Artifact manifest must not be a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("Invalid artifact manifest")
    seen = set()
    for artifact in manifest["files"]:
        relative = Path(artifact["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 4
            or relative.parts[:2] != ("artifacts", "objects")
            or relative.as_posix() in seen
        ):
            raise ValueError("Unsafe or duplicate artifact path")
        seen.add(relative.as_posix())
        path = root / relative
        if (
            any(part.is_symlink() for part in (path, *path.parents) if part.is_relative_to(root))
            or not path.is_file()
            or not path.resolve().is_relative_to(root / "artifacts/objects")
        ):
            raise ValueError(f"Missing or unsafe artifact: {relative}")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
            or relative.parts[2] != artifact["sha256"]
            or type(artifact["bytes"]) is not int
            or artifact["bytes"] < 0
            or any(
                not isinstance(artifact.get(key), str) or not artifact[key].strip()
                for key in ("license", "provenance")
            )
        ):
            raise ValueError("Invalid artifact hash, size or provenance")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != artifact["bytes"] or digest.hexdigest() != artifact["sha256"]:
            raise ValueError(f"Artifact checksum/size mismatch: {relative}")
    return {
        "registered_files": len(manifest["files"]),
        "verified_bytes": sum(row["bytes"] for row in manifest["files"]),
        "note": "Verified local content only, not remote LFS availability",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
