"""Installed-package quickstart scaffolding; never overwrite an existing project."""

import json
from importlib.resources import files
from pathlib import Path

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def init_project(directory):
    root = Path(directory).expanduser().resolve()
    if root.exists():
        raise FileExistsError(
            "Project destination exists; choose a new directory (nothing overwritten)"
        )
    resources = files("ouroboros.resources")
    names = ("self_edit.json", "research_lab.json", "practice.json", "validation.json", "test.json")
    contents = {name: resources.joinpath(name).read_text(encoding="utf-8") for name in names}
    for content in contents.values():
        json.loads(content)
    root.mkdir(parents=True, exist_ok=False)
    for name, content in contents.items():
        with (root / name).open("x", encoding="utf-8") as stream:
            stream.write(content)
    (root / ".gitignore").write_text("models/\noutputs/\n", encoding="utf-8")
    return {
        "project_dir": str(root),
        "config": str(root / "self_edit.json"),
        "research_config": str(root / "research_lab.json"),
        "research_commands": [
            ["ouroboros", "lab", "tools"],
            [
                "ouroboros",
                "lab",
                "run",
                "--config",
                str(root / "research_lab.json"),
                "--check-only",
            ],
            ["ouroboros", "lab", "run", "--config", str(root / "research_lab.json")],
        ],
        "next_commands": [
            [
                "ouroboros",
                "models",
                "fetch-smoke",
                "--directory",
                str(root / "models/smollm2"),
            ],
            ["ouroboros", "self-edit", "--config", str(root / "self_edit.json"), "--check-only"],
            ["ouroboros", "self-edit", "--config", str(root / "self_edit.json")],
        ],
        "alternative_model_command": [
            "ouroboros",
            "models",
            "fetch",
            "--repo",
            DEFAULT_MODEL,
            "--directory",
            str(root / "models/smollm2"),
        ],
        "note": "Model download is explicit. fetch-smoke needs llm+smoke extras; it uses a Q4_1-derived model, not original precision.",
    }
