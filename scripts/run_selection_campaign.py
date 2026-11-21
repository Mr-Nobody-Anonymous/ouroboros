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

from ouroboros.research.contracts import Budget, identity
from ouroboros.research.controller import run_campaign
from ouroboros.research.dataset import write_dataset
from ouroboros.research.memory import ExperimentMemory
from ouroboros.research.provenance import write_once
from ouroboros.research.selection import (
    analyse,
    analyse_magnitude,
    design,
    design_magnitude,
)

CONSUMED_SEED_RANGES = ((201, 220), (601, 620), (701, 720))
DEFAULT_FIRST_SEED = 721
MAGNITUDE_V2_SHA256 = "f166076cfc02432e604bf089056006da3e2589c7425df42fa8f19afb9caa43e5"
COMPACT_CAMPAIGN_BUDGET = Budget(
    max_steps=60,
    max_seconds=3600,
    max_rss_bytes=2400 * 1024**2,
    max_seq_length=192,
    max_new_tokens=12,
    max_train_examples=256,
    max_total_steps=900,
    max_candidates=2,
    repeats=2,
    checkpoint_every=30,
    cpu_threads=2,
)


def validate_fresh_seed_range(first_seed, episodes):
    """Refuse known G6 seed blocks; a new directory alone does not make data fresh."""
    if type(first_seed) is not int or first_seed < 0:
        raise ValueError("first_seed must be a nonnegative integer")
    if type(episodes) is not int or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    last_seed = first_seed + episodes - 1
    if last_seed >= 2**31 - 2:
        raise ValueError("Campaign seeds leave no room for the fixed evaluation-seed budget")
    for consumed_first, consumed_last in CONSUMED_SEED_RANGES:
        if first_seed <= consumed_last and last_seed >= consumed_first:
            raise ValueError(
                f"Seed range {first_seed}–{last_seed} overlaps consumed G6 seeds "
                f"{consumed_first}–{consumed_last}; choose a fresh range"
            )
    return first_seed, last_seed


def prepare(root, *, episodes, first_seed, model_path):
    validate_fresh_seed_range(first_seed, episodes)
    root.mkdir(parents=True, exist_ok=True)
    database = str(root / "experiments.sqlite3")
    configs = []
    for seed in range(first_seed, first_seed + episodes):
        directory = root / f"episode-{seed}"
        write_dataset(
            directory,
            seed=seed,
            model_path=model_path,
            profile="compact",
            budget=COMPACT_CAMPAIGN_BUDGET,
        )
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
    parser.add_argument(
        "--first-seed",
        type=int,
        default=DEFAULT_FIRST_SEED,
        help=(
            "First fresh seed (consumed G6 ranges 201–220, 601–620 and 701–720 are refused)"
        ),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--batch",
        type=int,
        default=7,
        help=(
            "campaign cycles per invocation; the family-correction budget allows 7 "
            "(7 cycles x 2 repeats x 3 gates = 42 <= family_size 44), so 8 is refused"
        ),
    )
    parser.add_argument("--analyse", action="store_true", help="Run the frozen replay at the end")
    parser.add_argument(
        "--design",
        default="binary_v1",
        choices=("binary_v1", "magnitude_v2"),
        help="Which frozen selection design to replay; the design is never changed here",
    )
    arguments = parser.parse_args()
    if not 1 <= arguments.batch <= 7:
        raise SystemExit(
            "A campaign holds at most 7 cycles per invocation: the family-correction budget "
            "covers 7 x 2 repeats x 3 gate checks = 42 of 44"
        )
    try:
        validate_fresh_seed_range(arguments.first_seed, arguments.episodes)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if arguments.analyse and arguments.episodes < 20:
        raise SystemExit("magnitude_v2 confirmation requires at least 20 logged fresh episodes")

    frozen = design_magnitude() if arguments.design == "magnitude_v2" else design()
    if arguments.design == "magnitude_v2" and identity(frozen) != MAGNITUDE_V2_SHA256:
        raise SystemExit("Frozen magnitude_v2 design hash changed; refusing to collect episodes")

    root = Path(arguments.workdir).absolute()
    if root.exists() or root.is_symlink():
        raise SystemExit("Campaign workdir must be new; never reuse consumed episodes or a ledger")
    root.mkdir(parents=True)
    predeclared = write_once(root / "selection_design_predeclared.json", frozen)
    print(
        f"=== selection design frozen before data collection: {predeclared['content_sha256']}",
        flush=True,
    )
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
        if episodes_logged != arguments.episodes:
            raise RuntimeError(
                f"Expected {arguments.episodes} complete candidate episodes, observed "
                f"{episodes_logged}; refusing to report an incomplete replay"
            )
        memory = ExperimentMemory(database, readonly=True)
        try:
            replay = analyse_magnitude if arguments.design == "magnitude_v2" else analyse
            result = replay(memory, directory=root / "selection")
        finally:
            memory.close()
        if result["design_sha256"] != predeclared["content_sha256"]:
            raise RuntimeError("Analysis design differs from the pre-run frozen design")
        write_once(root / "selection_result.json", result)
        print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
