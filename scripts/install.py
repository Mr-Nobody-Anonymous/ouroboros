"""One-command local venv setup; no model download, shell execution or system-package changes."""

import argparse
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["core", "llm", "research", "train"], default="core")
    parser.add_argument(
        "--venv",
        default=str(ROOT / ".venv"),
        help="Dedicated venv path (default: repository .venv)",
    )
    parser.add_argument("--dev", action="store_true", help="Also install tests/lint/build tools")
    parser.add_argument(
        "--unconstrained", action="store_true", help="Skip the tested ML dependency constraints"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the install plan without creating files"
    )
    args = parser.parse_args(argv)
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required")
    target = Path(args.venv).expanduser().absolute()
    if target.is_symlink() or (target.exists() and not (target / "pyvenv.cfg").is_file()):
        parser.error(
            "Venv destination must be new or an existing dedicated venv; nothing was changed"
        )
    executable = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    extras = ["mcp"]
    if args.profile != "core":
        extras += ["llm", "smoke"]
    if args.profile == "research":
        extras += ["math", "structured"]
    if args.profile == "train":
        extras += ["train"]
    if args.dev:
        extras += ["dev"]
    command = [str(executable), "-m", "pip", "install", "-e", f"{ROOT}[{','.join(extras)}]"]
    tested = args.profile != "core" and not args.unconstrained
    if tested:
        if sys.version_info >= (3, 13):
            parser.error(
                "The tested Torch 2.5.1 stack needs Python 3.10-3.12; use one of those or --unconstrained"
            )
        command += ["-c", str(ROOT / "constraints/ml-tested.txt")]
    plan = {
        "venv": str(target),
        "profile": args.profile,
        "install": command,
        "tested_ml_constraints": tested,
        "downloads_model": False,
        "next_commands": [
            [str(executable), "-m", "ouroboros.cli", "doctor"],
            [str(executable), "-m", "ouroboros.cli", "init", "ouroboros-lab"],
        ],
        "note": "ML packages are large. Choose a CPU/CUDA PyTorch build appropriate for your machine; no model is downloaded.",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    try:
        if not target.exists():
            venv.EnvBuilder(with_pip=True).create(target)
        subprocess.run(command, check=True)
        subprocess.run([str(executable), "-m", "pip", "check"], check=True)
        subprocess.run([str(executable), "-m", "ouroboros.cli", "doctor"], check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        print(
            f"Installation failed: {exc}. Your source files and models were not removed.",
            file=sys.stderr,
        )
        return 1
    print(json.dumps({**plan, "installed": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
