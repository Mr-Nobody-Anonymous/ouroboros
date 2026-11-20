#!/usr/bin/env python3
"""Explicitly clone pinned GSM8K data and prepare a new, immutable small evaluation."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from ouroboros.lab.discovery_data import GSM8K_SOURCE, prepare_discovery_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=".cache/upstream/grade-school-math")
    parser.add_argument("--output-dir", default="ouroboros/resources/discovery")
    parser.add_argument(
        "--fetch", action="store_true", help="Explicitly allow GitHub data-only clone if absent"
    )
    args = parser.parse_args(argv)
    clone = Path(args.source_dir).resolve()
    if not clone.exists() and args.fetch:
        clone.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
        subprocess.run(["git", "init", "-q", str(clone)], check=True, timeout=30, env=env)
        subprocess.run(
            [
                "git",
                "-C",
                str(clone),
                "-c",
                "protocol.file.allow=never",
                "fetch",
                "--depth=1",
                "--no-tags",
                GSM8K_SOURCE["repository"],
                GSM8K_SOURCE["revision"],
            ],
            check=True,
            timeout=180,
            env=env,
        )
        subprocess.run(
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
            check=True,
            timeout=30,
            env=env,
        )
    result = prepare_discovery_data(clone, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
