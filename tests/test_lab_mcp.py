"""Official SDK research permissions/content-binding tests, no Torch or model download."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from ouroboros.mcp_server import create_server
from ouroboros.project import init_project

pytestmark = pytest.mark.mcp
pytest.importorskip("mcp")


def unpack(result):
    return json.loads(result[0].text)


def test_research_default_tools_are_read_only_and_do_not_expose_labels_or_raw_files(tmp_path):
    root = tmp_path / "lab"
    init_project(root)
    path = root / "outputs/research/id/lab.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"kind": "ouroboros-research-run", "base_restored": True}))
    (path.parent / "operator_damage.json").write_text('{"private":"not a summary"}')

    async def check():
        server = create_server(root)
        tools = {tool.name: tool for tool in await server.list_tools()}
        assert len(tools) == 11 and all(tool.annotations.readOnlyHint for tool in tools.values())
        assert "run_research_lab" not in tools
        catalog = unpack(await server.call_tool("research_tools", {}))
        assert "add" in catalog["weight_operations"] and "contrast" in catalog["actions"]
        config = unpack(await server.call_tool("check_lab_config", {"config": "research_lab.json"}))
        assert config["counts"] == {"train": 2, "validation": 4, "test": 2}
        assert "Rome" not in json.dumps(config) and "capital of" not in json.dumps(config)
        reports = unpack(await server.call_tool("list_lab_reports", {}))
        assert reports["reports"] == ["outputs/research/id/lab.json"]
        assert unpack(await server.call_tool("read_lab_report", {"report": reports["reports"][0]}))[
            "base_restored"
        ]
        for raw in ("outputs/research/id/operator_damage.json", "../outside/lab.json"):
            with pytest.raises(Exception):
                await server.call_tool("read_lab_report", {"report": raw})

    asyncio.run(check())


def test_research_execution_binds_data_not_only_config_and_uses_fixed_offline_command(
    tmp_path, monkeypatch
):
    root = tmp_path / "lab"
    init_project(root)
    original, launches = asyncio.create_subprocess_exec, []

    async def fake_child(*args, **kwargs):
        launches.append((args, kwargs))
        return await original(
            sys.executable,
            "-c",
            "import json;print(json.dumps({'kind':'ouroboros-research-run','base_restored':True,'source_files_unchanged':True}))",
            **kwargs,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_child)

    async def check():
        server = create_server(
            root, allow_experiments=True, approved_lab_configs=["research_lab.json"]
        )
        tools = {tool.name: tool for tool in await server.list_tools()}
        assert "run_research_lab" in tools and "run_self_edit" not in tools
        assert not tools["run_research_lab"].annotations.readOnlyHint
        result = unpack(await server.call_tool("run_research_lab", {"config": "research_lab.json"}))
        assert result["base_restored"]
        args, kwargs = launches[0]
        assert args == (
            sys.executable,
            "-m",
            "ouroboros.cli",
            "lab",
            "run",
            "--config",
            str(root / "research_lab.json"),
        )
        assert kwargs["env"]["HF_HUB_OFFLINE"] == kwargs["env"]["HF_DATASETS_OFFLINE"] == "1"
        assert kwargs["stdin"] == asyncio.subprocess.DEVNULL and "shell" not in kwargs
        practice = root / "practice.json"
        practice.write_text(practice.read_text() + "\n")
        with pytest.raises(Exception, match="changed"):
            await server.call_tool("run_research_lab", {"config": "research_lab.json"})
        assert len(launches) == 1
        with pytest.raises(Exception, match="approved"):
            await server.call_tool("run_research_lab", {"config": "self_edit.json"})

    asyncio.run(check())
    with pytest.raises(ValueError, match="BOTH"):
        create_server(root, approved_lab_configs=["research_lab.json"])


def test_research_and_smoke_share_the_same_execution_lock(tmp_path, monkeypatch):
    root = tmp_path / "lab"
    init_project(root)
    original = asyncio.create_subprocess_exec

    async def check():
        launched = asyncio.Event()
        children = []

        async def fake_child(*args, **kwargs):
            child = await original(sys.executable, "-c", "import time;time.sleep(60)", **kwargs)
            children.append(child)
            launched.set()
            return child

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_child)
        server = create_server(
            root,
            allow_experiments=True,
            approved_configs=["self_edit.json"],
            approved_lab_configs=["research_lab.json"],
        )
        task = asyncio.create_task(
            server.call_tool("run_research_lab", {"config": "research_lab.json"})
        )
        await launched.wait()
        with pytest.raises(Exception, match="already running"):
            await server.call_tool("run_self_edit", {"config": "self_edit.json"})
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(children) == 1 and children[0].returncode is not None

    asyncio.run(check())


def test_research_tools_on_real_stdio_protocol(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    root = tmp_path / "lab"
    init_project(root)

    async def check():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "ouroboros.mcp_server", "--workspace", str(root)],
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("research_tools", {})
                assert not result.isError and "submit" in unpack(result.content)["actions"]
                result = await session.call_tool(
                    "check_lab_config", {"config": "research_lab.json"}
                )
                assert not result.isError and unpack(result.content)["valid"]
                bad = await session.call_tool("run_research_lab", {"config": "research_lab.json"})
                assert bad.isError

    asyncio.run(check())
