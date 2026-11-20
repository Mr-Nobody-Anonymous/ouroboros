"""Offline dataset conversion and explicit external-driver boundaries, not benchmark results."""

import json
import os
import sys

import pytest

from ouroboros.data import load_samples
from ouroboros.lab.benchmarks import benchmark_plan, execute_benchmark, import_local_data


def test_explicit_local_math_and_gsm_data_keep_provenance_and_never_self_label(tmp_path):
    math = tmp_path / "math.json"
    math.write_text(
        json.dumps([{"problem": "What is half?", "solution": r"Compute: \boxed{\frac{1}{2}}"}])
    )
    result = import_local_data(
        math,
        tmp_path / "converted",
        format="math",
        source="my licensed local collection",
        license_name="CC-BY-4.0",
    )
    rows = load_samples(result["samples"])
    assert rows[0].target == r"\frac{1}{2}"
    assert "\\boxed{}" in rows[0].instruction and not rows[0].verified
    assert "CC-BY-4.0" in rows[0].source and result["input_sha256"] in rows[0].source
    assert not result["network_accessed"]
    with pytest.raises(ValueError, match="new"):
        import_local_data(
            math, tmp_path / "converted", format="math", source="local", license_name="MIT"
        )
    gsm = tmp_path / "gsm.jsonl"
    gsm.write_text(json.dumps({"question": "Two plus two?", "answer": "Reasoning. #### 4"}) + "\n")
    result = import_local_data(
        gsm, tmp_path / "gsm", format="gsm8k", source="local", license_name="asserted"
    )
    assert load_samples(result["samples"])[0].target == "4"
    broken = tmp_path / "broken.json"
    broken.write_text('[{"problem":"No answer"}]')
    with pytest.raises(ValueError, match="reference"):
        import_local_data(
            broken, tmp_path / "not-written", format="math", source="local", license_name="asserted"
        )
    assert not (tmp_path / "not-written").exists()


def test_external_plan_never_installs_downloads_or_executes_and_blocks_argument_injection(
    tmp_path, monkeypatch
):
    model = tmp_path / "local model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"plan-only fixture; not loaded")
    monkeypatch.setattr("shutil.which", lambda name: None)
    result = execute_benchmark(model, tmp_path / "results", ["gsm8k"])
    assert not result["executed"] and not result["installed_executable"]
    assert result["argv"][1:4] == ["run", "--model", "hf"]
    assert "local_files_only=True" in result["argv"] and "trust_remote_code=False" in result["argv"]
    assert result["environment_overrides"]["HF_DATASETS_OFFLINE"] == "1"
    assert not (tmp_path / "results").exists()
    with pytest.raises(RuntimeError, match="separately"):
        execute_benchmark(model, tmp_path / "results", ["gsm8k"], allow_execution=True)
    with pytest.raises(ValueError, match="identifiers"):
        benchmark_plan(model, tmp_path / "results", ["../untrusted_task.py"])
    with pytest.raises(ValueError, match="device"):
        benchmark_plan(model, tmp_path / "results", ["gsm8k"], device="cpu; echo nope")


@pytest.mark.skipif(os.name != "posix", reason="POSIX executable/process-group fixture")
def test_external_executor_uses_fixed_args_offline_env_and_propagates_failure(
    tmp_path, monkeypatch
):
    model, bin_dir = tmp_path / "model", tmp_path / "bin"
    model.mkdir()
    bin_dir.mkdir()
    (model / "model.safetensors").write_bytes(b"plan fixture")
    executable = bin_dir / "lm-eval"
    executable.write_text(
        f"#!{sys.executable}\nimport os,sys\nassert os.environ['HF_DATASETS_OFFLINE'] == '1'\nassert 'trust_remote_code=False' in sys.argv\nprint('driver fixture, not a benchmark')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    result = execute_benchmark(model, tmp_path / "results", ["gsm8k"], allow_execution=True)
    assert result["executed"] and result["returncode"] == 0 and not result["validated_improvement"]
    assert (tmp_path / "results/invocation.json").is_file()
    executable.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(7)\n")
    with pytest.raises(RuntimeError, match="exited 7"):
        execute_benchmark(model, tmp_path / "failed", ["gsm8k"], allow_execution=True)
    report = json.loads((tmp_path / "failed/invocation.json").read_text())
    assert report["status"] == "failed" and not report["validated_improvement"]


def test_import_and_plan_are_wired_to_lightweight_json_cli(tmp_path, capsys):
    from ouroboros.cli import main

    path = tmp_path / "local.json"
    path.write_text(
        '[{"question":"A local reviewed arithmetic question?","answer":"Reason. #### 7"}]'
    )
    assert (
        main(
            [
                "lab",
                "import-data",
                "--input",
                str(path),
                "--output-dir",
                str(tmp_path / "data"),
                "--format",
                "gsm8k",
                "--source",
                "local reviewed fixture",
                "--license",
                "asserted test license",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert not result["network_accessed"] and load_samples(result["samples"])[0].target == "7"
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"plan fixture, not loaded")
    assert (
        main(
            [
                "lab",
                "benchmark",
                "--model-dir",
                str(model),
                "--output-dir",
                str(tmp_path / "planned"),
                "--task",
                "gsm8k",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert not plan["executed"] and not (tmp_path / "planned").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX executable/process-group fixture")
def test_external_timeout_records_failure_and_stops_child(tmp_path, monkeypatch):
    model, bin_dir = tmp_path / "model", tmp_path / "bin"
    model.mkdir()
    bin_dir.mkdir()
    (model / "model.safetensors").write_bytes(b"plan fixture")
    pid_path = tmp_path / "pid"
    executable = bin_dir / "lm-eval"
    executable.write_text(
        f"#!{sys.executable}\nimport os,time\nfrom pathlib import Path\nPath({str(pid_path)!r}).write_text(str(os.getpid()))\ntime.sleep(60)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    with pytest.raises(RuntimeError, match="timed out"):
        execute_benchmark(model, tmp_path / "timed-out", ["gsm8k"], allow_execution=True, timeout=1)
    report = json.loads((tmp_path / "timed-out/invocation.json").read_text())
    assert report["status"] == "failed" and not report["validated_improvement"]
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), 0)
