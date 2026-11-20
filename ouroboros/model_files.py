"""Explicit, size-bounded model provisioning. Nothing here runs during ordinary inference."""

import json
import re
import shutil
import tempfile
from pathlib import Path

from .core.checkpoints import atomic_json, file_sha256

RECEIPT = "ouroboros_model.json"
STANDARD_FILES = {
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
    "model.safetensors.index.json",
    "chat_template.jinja",
}


def model_file(name):
    return (
        isinstance(name, str)
        and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name))
        and (name in STANDARD_FILES or name.endswith(".safetensors"))
    )


def verify_model_files(directory):
    directory = Path(directory)
    receipt_path = directory / RECEIPT
    if receipt_path.is_symlink() or receipt_path.stat().st_size > 1024 * 1024:
        raise ValueError("Invalid model receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != 1
        or not isinstance(receipt.get("files"), dict)
        or not receipt["files"]
        or not re.fullmatch(r"[0-9a-f]{40}", receipt.get("resolved_revision", ""))
    ):
        raise ValueError("Invalid model receipt")
    for name, expected in receipt["files"].items():
        path = directory / name
        if (
            not model_file(name)
            or path.is_symlink()
            or not path.is_file()
            or not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            raise ValueError("Invalid/missing model file")
        if file_sha256(path) != expected:
            raise ValueError(f"Model checksum mismatch: {name}")
    actual = {p.name for p in directory.iterdir() if p.is_file() and model_file(p.name)}
    if actual != set(receipt["files"]):
        raise ValueError("Model file inventory differs from receipt")
    return {**receipt, "verified": True, "directory": str(directory.resolve())}


def fetch_model(repo_id, directory, *, revision="main", max_bytes=1024**3):
    """User-requested Hub access; pin the resolved commit, refuse pickle/Python and oversize files.

    The destination must not exist. Failed downloads only remove this function's own temporary
    directory. License metadata is an upstream assertion, not legal approval to redistribute.
    """
    if (
        not isinstance(repo_id, str)
        or not re.fullmatch(
            r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}", repo_id
        )
        or ".." in repo_id
    ):
        raise ValueError("Expected a Hub repository ID such as HuggingFaceTB/SmolLM2-135M-Instruct")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("A revision is required")
    if type(max_bytes) is not int or not 1 <= max_bytes <= 8 * 1024**3:
        raise ValueError("max_bytes must be an integer in [1, 8 GiB]")
    target = Path(directory).expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(
            "Model destination already exists; verify it or choose a new directory"
        )
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install the llm extra to download models") from exc
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    if not isinstance(info.sha, str) or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise ValueError("Hub did not return a pinned commit")
    files = [s for s in info.siblings if model_file(s.rfilename)]
    if not any(s.rfilename.endswith(".safetensors") for s in files) or not any(
        s.rfilename == "config.json" for s in files
    ):
        raise ValueError("A standard config and safetensors weights are required")
    if len(files) > 128 or any(type(s.size) is not int or s.size < 0 for s in files):
        raise ValueError("Unknown file sizes or too many model files")
    total = sum(s.size for s in files)
    if total > max_bytes:
        raise ValueError(f"Model download exceeds byte budget: {total} > {max_bytes}")
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=".pending-model-", dir=target.parent))
    try:
        snapshot_download(
            repo_id,
            revision=info.sha,
            local_dir=str(pending),
            allow_patterns=[s.rfilename for s in files],
        )
        hashes = {}
        for file in files:
            path = pending / file.rfilename
            if path.is_symlink() or not path.is_file() or path.stat().st_size != file.size:
                raise ValueError(f"Downloaded file size/type mismatch: {file.rfilename}")
            hashes[file.rfilename] = file_sha256(path)
        card = info.card_data.to_dict() if info.card_data is not None else {}
        atomic_json(
            pending / RECEIPT,
            {
                "schema_version": 1,
                "repo_id": repo_id,
                "requested_revision": revision,
                "resolved_revision": info.sha,
                "license_metadata": card.get("license", "unknown"),
                "bytes": total,
                "files": hashes,
            },
        )
        verify_model_files(pending)
        pending.rename(target)
    finally:
        if pending.exists():
            shutil.rmtree(pending)
    return verify_model_files(target)


SMOKE_SOURCE = {
    "repository": "https://github.com/simonw/llm-smollm2.git",
    "revision": "afed0963b3a8bcca17c83f451a161eb2edbc59fe",
    "path": "llm_smollm2/SmolLM2-135M-Instruct.Q4_1.gguf",
    "bytes": 98362432,
    "sha256": "b179c9523d0e6a0f98a330c7562b682750a6f8c8c15e5bc70ea373728110db53",
    "original_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "quantizer": "QuantFactory/SmolLM2-135M-Instruct-GGUF",
    "quantizer_revision": "ab810cf68114990406fdf996510dd3d3c6adbdf5",
    "license": "Apache-2.0",
}


def fetch_smoke_model(directory):
    """Explicit GitHub-only fixture provisioning, using a pinned, hashed pretrained GGUF.

    Reads data only from the clone: NEVER imports/executes its plugin or installs its packages.
    Transformers dequantizes Q4_1 to ordinary float32 safetensors. Lost precision is not recovered.
    This is a derivative smoke model, NOT the original full-precision Hub checkpoint.
    """
    import os
    import subprocess

    target = Path(directory).expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(
            "Model destination already exists; verify it or choose a new directory"
        )
    if not shutil.which("git"):
        raise RuntimeError("Git is required for the GitHub smoke-model source")
    try:
        import gguf  # noqa: F401
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the llm and smoke extras for GGUF conversion") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < 1536 * 1024**2:
        raise ValueError("At least 1.5 GiB free disk is required for smoke-model conversion")
    pending = Path(tempfile.mkdtemp(prefix=".pending-smoke-", dir=target.parent))
    threads = torch.get_num_threads()
    try:
        clone, output = pending / "source", pending / "converted"
        clone.mkdir()
        output.mkdir()
        environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
        commands = [
            ["git", "init", "-q", str(clone)],
            [
                "git",
                "-C",
                str(clone),
                "-c",
                "protocol.file.allow=never",
                "fetch",
                "--depth=1",
                "--no-tags",
                SMOKE_SOURCE["repository"],
                SMOKE_SOURCE["revision"],
            ],
            [
                "git",
                "-C",
                str(clone),
                "-c",
                "core.hooksPath=/dev/null",
                "checkout",
                "--detach",
                "FETCH_HEAD",
            ],
        ]
        for command in commands:
            subprocess.run(
                command, check=True, timeout=180, capture_output=True, text=True, env=environment
            )
        gguf_path = clone / SMOKE_SOURCE["path"]
        if (
            gguf_path.is_symlink()
            or gguf_path.stat().st_size != SMOKE_SOURCE["bytes"]
            or (file_sha256(gguf_path) != SMOKE_SOURCE["sha256"])
        ):
            raise ValueError("Pinned pretrained GGUF size/checksum mismatch")
        torch.set_num_threads(2)
        tokenizer = AutoTokenizer.from_pretrained(
            str(gguf_path.parent),
            gguf_file=gguf_path.name,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(gguf_path.parent),
            gguf_file=gguf_path.name,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
        )
        model.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
        shutil.copyfile(clone / "LICENSE", output / "LICENSE")
        hashes = {
            file.name: file_sha256(file) for file in output.iterdir() if model_file(file.name)
        }
        atomic_json(
            output / RECEIPT,
            {
                "schema_version": 1,
                "repo_id": "github:simonw/llm-smollm2",
                "requested_revision": SMOKE_SOURCE["revision"],
                "resolved_revision": SMOKE_SOURCE["revision"],
                "license_metadata": "apache-2.0",
                "source": SMOKE_SOURCE,
                "conversion": "Q4_1 GGUF -> float32 safetensors with Transformers; NOT recovered original precision",
                "bytes": sum((output / name).stat().st_size for name in hashes),
                "files": hashes,
            },
        )
        verify_model_files(output)
        output.rename(target)
    finally:
        torch.set_num_threads(threads)
        if pending.exists():
            shutil.rmtree(pending)
    return verify_model_files(target)
