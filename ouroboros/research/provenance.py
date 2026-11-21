"""Run-owned immutable JSON, complete local source/model inventories and data-only bindings."""

import importlib.metadata
import platform
import sys
from pathlib import Path

from ..core.checkpoints import atomic_json, file_sha256
from .contracts import identity

MODEL_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "vocab.txt",
    "added_tokens.json",
    "chat_template.jinja",
)


def regular(path):
    path = Path(path).absolute()
    if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():
        raise ValueError(f"A regular, non-symlinked input file is required: {path}")
    return path


def model_inventory(directory):
    directory = Path(directory).absolute()
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("An existing regular local model directory is required")
    files = sorted({p for pattern in MODEL_PATTERNS for p in directory.glob(pattern)})
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Local model needs safetensors weights; no downloads or pickle")
    return {str(p.relative_to(directory)): file_sha256(regular(p)) for p in files}


def software_inventory():
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for name in (
        "ouroboros",
        "numpy",
        "torch",
        "transformers",
        "peft",
        "safetensors",
        "tokenizers",
        "accelerate",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "files": {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*.py"))},
    }


def assert_software(expected):
    if software_inventory() != expected:
        raise ValueError(
            "Research implementation or dependency versions changed during a frozen run"
        )


def write_once(path, value):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite frozen research artifact: {path}")
    atomic_json(path, value)
    return {"file": str(path), "sha256": file_sha256(path), "content_sha256": identity(value)}


def verify_binding(binding):
    path = regular(binding["file"])
    if file_sha256(path) != binding["sha256"]:
        raise ValueError("Research artifact hash changed")
    return path
