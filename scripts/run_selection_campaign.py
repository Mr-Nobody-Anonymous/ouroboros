"""Run real selection-policy episodes, then replay the frozen comparison.

Each episode is a full foundation run on its own freshly generated `compact` dataset (fresh
hidden prompts, three domains), all sharing one authoritative ledger so the counterfactual
candidate log accumulates in a single selection context. No prompt is reused.

This collects *data*; it does not change the predeclared design in
`ouroboros/research/selection.py`.
"""

import argparse
import json
import sqlite3
from pathlib import Path

from ouroboros.research.controller import run_campaign
from ouroboros.research.dataset import write_dataset
from ouroboros.research.memory import ExperimentMemory
from ouroboros.research.selection import analyse


def prepare(root, *, episodes, first_seed, model_path):
    root.mkdir(parents=True, exist_ok=True)
    database = str(root / "experiments.sqlite3")
    configs = []
    for seed in range(first_seed, first_seed + episodes):
        directory = root / f"episode-{seed}"
        write_dataset(directory, seed=seed, model_path=model_path, profile="compact")
        path = directory / "foundation.json"
        config = json.loads(path.read_text())
        config["database"] = database
        config["output_dir"] = str(root / "runs" / f"episode-{seed}")
        path.write_text(json.dumps(config, indent=2, sort_keys=True))
        configs.append(str(path))
    return database, configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=201)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--batch", type=int, default=7, help="campaign cycles per invocation (max 8)"
    )
    parser.add_argument("--analyse", action="store_true", help="Run the frozen replay at the end")
    arguments = parser.parse_args()
    if not 1 <= arguments.batch <= 8:
        raise SystemExit("A campaign holds at most 8 predeclared cycles")

    root = Path(arguments.workdir).absolute()
    database, configs = prepare(
        root,
        episodes=arguments.episodes,
        first_seed=arguments.first_seed,
        model_path=arguments.model,
    )
    for start in range(0, len(configs), arguments.batch):
        chunk = configs[start : start + arguments.batch]
        print(f"=== campaign {start // arguments.batch + 1}: {len(chunk)} cycles", flush=True)
        print(json.dumps(run_campaign(chunk, max_cycles=len(chunk)))[:400], flush=True)

    ledger = sqlite3.connect(database, timeout=10)
    episodes_logged = ledger.execute(
        "select count(*) from (select distinct experiment_id from candidates)"
    ).fetchone()[0]
    print(f"=== {episodes_logged} episodes with a logged candidate set")
    ledger.close()
    if arguments.analyse:
        memory = ExperimentMemory(database, readonly=True)
        try:
            result = analyse(memory, directory=root / "selection")
        finally:
            memory.close()
        print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
