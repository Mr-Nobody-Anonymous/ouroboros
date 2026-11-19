"""Build a deterministic core source bundle + SHA256; never publish or include weights/caches.

Third-party trees are intentionally references, not repackaged under the core license.
A full development checkout is obtained by Git clone + submodule initialization.
"""

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {
    "ouroboros",
    "tests",
    "scripts",
    "papers",
    "docs",
    "analysis",
    "examples",
    "constraints",
    "sandbox",
    ".github",
}
TOP = {
    "README.md",
    "AGENTS.md",
    "MANIFEST.in",
    "LICENSE",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "requirements.txt",
    "pyproject.toml",
    ".gitmodules",
    ".gitignore",
    ".gitattributes",
}
EXTRA = {"external/README.md", "external/sources.lock.json", "artifacts/manifest.json"}


def build(root=ROOT, output_dir=None, *, allow_dirty=False):
    root = Path(root).resolve()
    state = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip()
    if state and not allow_dirty:
        raise ValueError(
            "Working tree is dirty. Commit/review first, or use --allow-dirty for a local preview bundle."
        )
    names = (
        subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
        )
        .decode()
        .split("\0")
    )
    files = {}
    for name in sorted(set(filter(None, names))):
        path = Path(name)
        if not (name in TOP | EXTRA or path.parts[0] in FOLDERS):
            continue
        source = root / path
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Unexpected source entry: {name}")
        if source.stat().st_size > 10 * 1024 * 1024 or source.suffix in {
            ".pt",
            ".pth",
            ".bin",
            ".safetensors",
            ".npz",
            ".pyc",
        }:
            raise ValueError(f"Refusing large/binary artifact in core source release: {name}")
        files[name] = source.read_bytes()
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    version_text = (root / "ouroboros/__init__.py").read_text(encoding="utf-8")
    import re

    version = re.search(r'__version__ = "([^"]+)"', version_text).group(1)
    manifest = {
        "version": version,
        "commit": commit,
        "dirty_preview": bool(state),
        "scope": "Ouroboros core source, docs, tests and source references; no upstream trees, model weights or caches",
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    files["RELEASE_CONTENTS.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    output = Path(output_dir) if output_dir else root / "artifacts/releases"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"ouroboros-{version}-core.tar.gz"
    prefix = f"ouroboros-{version}"
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped,
    ):
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            for name, data in sorted(files.items()):
                info = tarfile.TarInfo(f"{prefix}/{name}")
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tar.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_name(archive.name + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return {
        "archive": str(archive),
        "sha256": digest,
        "checksum_file": str(checksum),
        "bytes": archive.stat().st_size,
        "manifest": manifest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    result = build(output_dir=args.output_dir, allow_dirty=args.allow_dirty)
    print(json.dumps({key: value for key, value in result.items() if key != "manifest"}, indent=2))


if __name__ == "__main__":
    main()
