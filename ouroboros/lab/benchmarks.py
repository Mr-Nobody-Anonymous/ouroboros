"""Explicit local-data preparation and upstream benchmark execution, never implicit downloads.

The pinned evaluation-harness CLI was inspected, not installed by importing this module.
An external run needs a separately installed trusted lm-eval and pre-provisioned datasets.
"""

import json
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

from ..core.checkpoints import atomic_json, file_sha256
from ..data import Sample, unique_samples
from .config import integer, strict_json
from .metrics import last_boxed

MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
HARNESS_REVISION = "d6de81643928d653435c431bae19945d41d32520"


def import_local_data(input_path, output_dir, *, format, source, license_name, domain="math"):
    """Convert explicit local math/GSM8K/DeepScaleR or Sample records; never create self-labels."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("Dataset destination must be new; nothing overwritten")
    if format not in ("math", "gsm8k", "deepscaler", "sample"):
        raise ValueError("Supported local formats: math, gsm8k, deepscaler, sample")
    if any(
        not isinstance(value, str) or not 1 <= len(value.strip()) <= 2000
        for value in (source, license_name, domain)
    ):
        raise ValueError("Explicit bounded dataset source, license and domain required")
    if (
        not input_path.is_file()
        or input_path.is_symlink()
        or input_path.stat().st_size > 64 * 1024**2
    ):
        raise ValueError("A nonsymlinked local dataset <=64 MiB required")
    text = input_path.read_text(encoding="utf-8")
    rows = (
        [strict_json(line) for line in text.splitlines() if line.strip()]
        if input_path.suffix == ".jsonl"
        else strict_json(text)
    )
    if not isinstance(rows, list) or not 1 <= len(rows) <= 50_000:
        raise ValueError("Expected 1-50000 local dataset rows")
    digest = file_sha256(input_path)
    provenance = f"{source}; asserted license: {license_name}; input_sha256:{digest}"
    samples = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Dataset rows must be objects")
        if format == "sample":
            sample = Sample.from_dict(row)
            sample = Sample(sample.instruction, sample.target, sample.domain, provenance, False)
        else:
            prompt = row.get("question") if format == "gsm8k" else row.get("problem")
            reference = row.get("answer")
            if reference is None and format == "math":
                solution = row.get("solution")
                reference = last_boxed(solution) if isinstance(solution, str) else None
            if (
                not isinstance(prompt, str)
                or not isinstance(reference, (str, int, float))
                or isinstance(reference, bool)
            ):
                raise ValueError(
                    "Expected an explicit problem/question and reference answer (or a boxed MATH solution)"
                )
            target = str(reference).strip()
            if format == "gsm8k":
                target = target.rsplit("####", 1)[-1].strip()
            if not prompt.endswith(MATH_INSTRUCTION):
                prompt += "\n" + MATH_INSTRUCTION
            sample = Sample(prompt, target, domain, provenance, False)
        if len(sample.instruction) > 32000 or len(sample.target) > 4000:
            raise ValueError("Converted sample exceeds the research character budget")
        samples.append(sample)
    samples = unique_samples(samples)
    output_dir.mkdir(parents=True, exist_ok=False)
    data_path = output_dir / "samples.jsonl"
    with data_path.open("x", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample.as_dict(), ensure_ascii=False, allow_nan=False) + "\n")
    receipt = {
        "kind": "ouroboros-local-dataset-import",
        "format": format,
        "source": source,
        "license": license_name,
        "input_sha256": digest,
        "output_sha256": file_sha256(data_path),
        "input_rows": len(rows),
        "output_rows": len(samples),
        "independently_verified_labels": False,
        "network_accessed": False,
        "note": "Caller-supplied reference labels and license assertions, not correctness/legal certification. Keep practice/reserved/final files separate; exact dedup is not semantic decontamination.",
    }
    atomic_json(output_dir / "dataset.json", receipt)
    return {**receipt, "samples": str(data_path)}


def benchmark_plan(model_dir, output_dir, tasks, *, limit=100, device="cpu", max_new_tokens=4096):
    model_dir, output_dir = Path(model_dir).resolve(), Path(output_dir).resolve()
    if not model_dir.is_dir() or not list(model_dir.glob("*.safetensors")):
        raise ValueError("A local safetensors model directory required")
    if any(character in str(model_dir) for character in (",", "=", "\n", "\r", "'", '"')):
        raise ValueError("Model path contains upstream argument-parser delimiters")
    if output_dir.exists():
        raise ValueError("Benchmark output destination must be new")
    if (
        not isinstance(tasks, list)
        or not 1 <= len(tasks) <= 32
        or any(
            not isinstance(task, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,99}", task) is None
            for task in tasks
        )
    ):
        raise ValueError("Choose 1-32 installed standard task identifiers, not files/expressions")
    integer("benchmark limit", limit, 1, 50_000)
    integer("benchmark max_new_tokens", max_new_tokens, 1, 32768)
    if not isinstance(device, str) or re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device) is None:
        raise ValueError("Benchmark device must be cpu or cuda[:index]")
    executable = shutil.which("lm-eval")
    return {
        "kind": "ouroboros-external-benchmark-plan",
        "executed": False,
        "installed_executable": executable is not None,
        "reviewed_source_revision": HARNESS_REVISION,
        "argv": [
            executable or "lm-eval",
            "run",
            "--model",
            "hf",
            "--model_args",
            f"pretrained={model_dir}",
            "trust_remote_code=False",
            "local_files_only=True",
            "--tasks",
            *tasks,
            "--device",
            device,
            "--batch_size",
            "1",
            "--num_fewshot",
            "0",
            "--limit",
            str(limit),
            "--gen_kwargs",
            "do_sample=False",
            f"max_new_tokens={max_new_tokens}",
            "--output_path",
            str(output_dir / "results"),
        ],
        "output_dir": str(output_dir),
        "environment_overrides": {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "limits": "Separately installed upstream executable/datasets required. This is a fixed greedy protocol, not the paper's sampling protocol. Offline flags are not a network firewall; upstream libraries/task code remain trusted. Does not apply bundles or overwrite source weights.",
    }


def execute_benchmark(
    model_dir, output_dir, tasks, *, allow_execution=False, timeout=1800, **kwargs
):
    plan = benchmark_plan(model_dir, output_dir, tasks, **kwargs)
    if not allow_execution:
        return plan
    if not plan["installed_executable"]:
        raise RuntimeError(
            "Install the reviewed lm-evaluation-harness separately; no fallback or download was attempted"
        )
    integer("benchmark timeout", timeout, 1, 86400)
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    result, process = {**plan, "executed": True, "status": "running"}, None
    try:
        with (
            (output_dir / "stdout.log").open("x") as stdout,
            (output_dir / "stderr.log").open("x") as stderr,
        ):
            process = subprocess.Popen(
                plan["argv"],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env={**os.environ, **plan["environment_overrides"]},
                start_new_session=os.name == "posix",
            )
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("External benchmark timed out; process group stopped") from exc
        result.update(
            status="external-tool-completed" if code == 0 else "external-tool-failed",
            returncode=code,
        )
        if code:
            raise RuntimeError(
                f"External benchmark exited {code}; inspect owned stdout/stderr logs"
            )
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if process is not None and process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
        result["validated_improvement"] = False
        atomic_json(output_dir / "invocation.json", result)
    return result
