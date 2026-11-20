"""Real official SDK/stdin-stdout protocol checks, plus opt-in execution-boundary tests."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from ouroboros.mcp_server import create_server
from ouroboros.project import init_project

pytestmark = pytest.mark.mcp
pytest.importorskip("mcp")


def unpack(result):
    return json.loads(result[0].text)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "lab"
    init_project(root)
    return root


def test_default_tools_are_read_only_and_config_check_never_leaks_held_out_content(project):
    async def check():
        server = create_server(project)
        tools = await server.list_tools()
        assert len(tools) == 6 and all(t.annotations.readOnlyHint for t in tools)
        assert "run_self_edit" not in {t.name for t in tools}
        status = unpack(await server.call_tool("status", {}))
        assert status["permissions"]["read_only"] and not status["permissions"]["experiments"]
        config = unpack(
            await server.call_tool("check_self_edit_config", {"config": "self_edit.json"})
        )
        assert config["counts"] == {"train": 2, "validation": 4, "test": 2}
        assert "Rome" not in json.dumps(config) and "capital of" not in json.dumps(config)
        with pytest.raises(Exception, match="Unknown tool"):
            await server.call_tool("run_self_edit", {"config": "self_edit.json"})

    asyncio.run(check())


def test_mcp_path_limits_symlinks_reports_and_safetensors_prefixes(project, tmp_path):
    save_file({"weights": np.ones((2, 2), dtype=np.float32)}, str(project / "weights.safetensors"))
    outside = tmp_path / "outside.safetensors"
    save_file({"x": np.zeros(1, dtype=np.float32)}, str(outside))
    (project / "linked.safetensors").symlink_to(outside)
    report = project / "outputs/self-edit/id/self_edit.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"kind": "model-selected-self-edit", "status": "completed"}))
    (project / "wrong.json").write_text('{"secret":"not a report"}')

    async def check():
        server = create_server(project)
        result = unpack(
            await server.call_tool("inspect_weights", {"file": "weights.safetensors", "values": 2})
        )
        assert result["tensors"][0]["flat_prefix_values"] == [1.0, 1.0]
        for path in ("../outside.safetensors", ".env", "linked.safetensors"):
            with pytest.raises(Exception):
                await server.call_tool("inspect_weights", {"file": path})
        with pytest.raises(Exception, match="limit"):
            await server.call_tool("inspect_weights", {"file": "weights.safetensors", "limit": 101})
        listing = unpack(await server.call_tool("list_self_edit_reports", {}))
        assert listing["reports"] == ["outputs/self-edit/id/self_edit.json"]
        result = unpack(
            await server.call_tool("read_self_edit_report", {"report": listing["reports"][0]})
        )
        assert result["status"] == "completed"
        with pytest.raises(Exception, match="self_edit.json"):
            await server.call_tool("read_self_edit_report", {"report": "wrong.json"})

    asyncio.run(check())


def test_execution_needs_both_explicit_opt_in_and_allowlist(project):
    with pytest.raises(ValueError, match="BOTH"):
        create_server(project, allow_experiments=True)
    with pytest.raises(ValueError, match="BOTH"):
        create_server(project, approved_configs=["self_edit.json"])
    with pytest.raises(ValueError, match="timeout"):
        create_server(project, timeout_seconds=1801)


def test_experiments_accept_only_unchanged_allowlisted_configs_and_use_offline_child(
    project, monkeypatch
):
    original = asyncio.create_subprocess_exec
    launches = []

    async def fake_child(*args, **kwargs):
        launches.append((args, kwargs))
        # Substitute ONLY the test child. Production always invokes the fixed Ouroboros module.
        return await original(
            sys.executable,
            "-c",
            "import json;print(json.dumps({'kind':'model-selected-self-edit','base_restored':True}))",
            **kwargs,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_child)

    async def check():
        server = create_server(project, allow_experiments=True, approved_configs=["self_edit.json"])
        tools = {t.name: t for t in await server.list_tools()}
        assert not tools["run_self_edit"].annotations.readOnlyHint
        result = unpack(await server.call_tool("run_self_edit", {"config": "self_edit.json"}))
        assert result["base_restored"]
        args, options = launches[0]
        assert args == (
            sys.executable,
            "-m",
            "ouroboros.cli",
            "self-edit",
            "--config",
            str(project / "self_edit.json"),
        )
        assert (
            options["env"]["HF_HUB_OFFLINE"] == "1"
            and options["env"]["TRANSFORMERS_OFFLINE"] == "1"
        )
        assert "shell" not in options
        path = project / "self_edit.json"
        path.write_text(path.read_text() + "\n")
        with pytest.raises(Exception, match="changed"):
            await server.call_tool("run_self_edit", {"config": "self_edit.json"})
        with pytest.raises(Exception, match="approved"):
            await server.call_tool("run_self_edit", {"config": "practice.json"})
        assert len(launches) == 1

    asyncio.run(check())


def test_experiment_timeout_stops_its_child(project, monkeypatch):
    original = asyncio.create_subprocess_exec
    children = []

    async def sleeping_child(*args, **kwargs):
        child = await original(
            sys.executable,
            "-c",
            "import sys,time;sys.stdout.write('x'*262144);sys.stdout.flush();time.sleep(60)",
            **kwargs,
        )
        children.append(child)
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", sleeping_child)

    async def check():
        server = create_server(
            project, allow_experiments=True, approved_configs=["self_edit.json"], timeout_seconds=1
        )
        with pytest.raises(Exception):
            await server.call_tool("run_self_edit", {"config": "self_edit.json"})
        assert children[0].returncode is not None

    asyncio.run(check())


def test_real_stdio_protocol_initialize_list_and_call_read_only_tools(project):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def check():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "ouroboros.mcp_server", "--workspace", str(project)],
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                assert initialized.serverInfo.name == "Ouroboros"
                tools = await session.list_tools()
                assert len(tools.tools) == 6 and all(
                    t.annotations.readOnlyHint for t in tools.tools
                )
                result = await session.call_tool("status", {})
                assert not result.isError and unpack(result.content)["permissions"]["read_only"]
                result = await session.call_tool(
                    "check_self_edit_config", {"config": "self_edit.json"}
                )
                assert not result.isError and unpack(result.content)["valid"]
                bad = await session.call_tool("inspect_weights", {"file": "../escape.safetensors"})
                assert bad.isError

    asyncio.run(check())


@pytest.mark.parametrize("cancel", [False, True])
def test_experiments_serialize_and_cancelled_children_are_stopped(project, monkeypatch, cancel):
    original = asyncio.create_subprocess_exec
    children = []

    async def check():
        launched = asyncio.Event()

        async def controlled_child(*args, **kwargs):
            code = (
                "import time;time.sleep(60)"
                if cancel
                else "import json,time;time.sleep(.2);print(json.dumps({'kind':'model-selected-self-edit','base_restored':True}))"
            )
            child = await original(sys.executable, "-c", code, **kwargs)
            children.append(child)
            launched.set()
            return child

        monkeypatch.setattr(asyncio, "create_subprocess_exec", controlled_child)
        server = create_server(project, allow_experiments=True, approved_configs=["self_edit.json"])
        task = asyncio.create_task(server.call_tool("run_self_edit", {"config": "self_edit.json"}))
        await launched.wait()
        with pytest.raises(Exception, match="already running"):
            await server.call_tool("run_self_edit", {"config": "self_edit.json"})
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert unpack(await task)["base_restored"]
        assert len(children) == 1 and children[0].returncode is not None

    asyncio.run(check())
