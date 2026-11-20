"""Append-only SQLite episodic/semantic/strategic records. Hash chains detect, not prevent, tampering."""

import json
import math
import os
import sqlite3
import time
from pathlib import Path

from .contracts import canonical, identity, integer, text

TERMINAL = {"completed", "rejected", "abstained", "failed", "interrupted"}
TRANSITIONS = {
    "registered": {"observing", "interrupted", "failed"},
    "observing": {"diagnosing", "failed", "interrupted"},
    "diagnosing": {"researching", "failed", "interrupted"},
    "researching": {"designed", "abstained", "failed", "interrupted"},
    "designed": {"executing", "failed", "interrupted"},
    "executing": {"validating", "failed", "interrupted"},
    "validating": {"reserved", "rejected", "failed", "interrupted"},
    "reserved": {"final", "rejected", "failed", "interrupted"},
    "final": {"robustness", "rejected", "failed", "interrupted"},
    "robustness": {"retention", "rejected", "failed", "interrupted"},
    "retention": {"completed", "rejected", "failed", "interrupted"},
}


def process_token(pid):
    """Linux start time avoids mistaking a reused PID for the original controller."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class ExperimentMemory:
    def __init__(self, path, *, readonly=False):
        path = Path(path)
        if path.is_symlink() or any(p.is_symlink() for p in path.parents):
            raise ValueError("Memory database must not be symlinked")
        if not readonly:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.readonly = path.resolve(), readonly
        self.db = sqlite3.connect(
            self.path.as_uri() + ("?mode=ro" if readonly else "?mode=rwc"),
            uri=True,
            timeout=10,
            isolation_level=None,
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=10000")
        if not readonly:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata(version INTEGER NOT NULL);
                INSERT INTO metadata SELECT 1 WHERE NOT EXISTS(SELECT 1 FROM metadata);
                CREATE TABLE IF NOT EXISTS baselines(
                    id TEXT PRIMARY KEY, metadata TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiments(
                    id TEXT PRIMARY KEY, created REAL NOT NULL, plan TEXT NOT NULL,
                    plan_sha256 TEXT NOT NULL, owner_pid INTEGER NOT NULL, owner_token TEXT
                );
                CREATE TABLE IF NOT EXISTS events(
                    experiment_id TEXT NOT NULL REFERENCES experiments(id), sequence INTEGER NOT NULL,
                    state TEXT NOT NULL, payload TEXT NOT NULL, previous_sha256 TEXT NOT NULL,
                    sha256 TEXT NOT NULL, PRIMARY KEY(experiment_id,sequence)
                );
                CREATE TABLE IF NOT EXISTS results(
                    experiment_id TEXT PRIMARY KEY REFERENCES experiments(id), context TEXT NOT NULL,
                    family TEXT NOT NULL, useful INTEGER NOT NULL CHECK(useful IN(0,1)),
                    cost REAL NOT NULL CHECK(cost>=0), summary TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS claims(
                    experiment_id TEXT NOT NULL REFERENCES experiments(id), claim TEXT NOT NULL,
                    provenance TEXT NOT NULL, verified INTEGER NOT NULL CHECK(verified=0)
                );
                CREATE TABLE IF NOT EXISTS holdouts(
                    prompt_sha256 TEXT NOT NULL, role TEXT NOT NULL, experiment_id TEXT NOT NULL REFERENCES experiments(id),
                    PRIMARY KEY(prompt_sha256)
                );
                CREATE TABLE IF NOT EXISTS prompt_roles(
                    prompt_sha256 TEXT PRIMARY KEY, role TEXT NOT NULL,
                    experiment_id TEXT NOT NULL REFERENCES experiments(id)
                );
            """)
            for table in (
                "baselines",
                "experiments",
                "events",
                "results",
                "claims",
                "holdouts",
                "prompt_roles",
            ):
                for operation in ("UPDATE", "DELETE"):
                    self.db.execute(
                        f"CREATE TRIGGER IF NOT EXISTS no_{table}_{operation} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT,'append-only research memory'); END"
                    )
        if [r[0] for r in self.db.execute("SELECT version FROM metadata")] != [1]:
            raise ValueError("Unsupported experiment memory version")
        self.verify()

    def close(self):
        self.db.close()

    def register_baseline(self, metadata):
        key = identity(metadata)
        self.db.execute("INSERT OR IGNORE INTO baselines VALUES(?,?)", (key, canonical(metadata)))
        return key

    def register(self, experiment_id, plan):
        text(experiment_id, "experiment id", 100)
        encoded = canonical(plan)
        payload, previous = {"plan_sha256": identity(plan)}, "0" * 64
        sha = identity(
            {
                "experiment_id": experiment_id,
                "sequence": 0,
                "state": "registered",
                "payload": payload,
                "previous_sha256": previous,
            }
        )
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "INSERT INTO experiments VALUES(?,?,?,?,?,?)",
                (
                    experiment_id,
                    time.time(),
                    encoded,
                    identity(plan),
                    os.getpid(),
                    process_token(os.getpid()),
                ),
            )
            self.db.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?)",
                (experiment_id, 0, "registered", canonical(payload), previous, sha),
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def state(self, experiment_id):
        row = self.db.execute(
            "SELECT state FROM events WHERE experiment_id=? ORDER BY sequence DESC LIMIT 1",
            (experiment_id,),
        ).fetchone()
        return row[0] if row else None

    def transition(self, experiment_id, state, payload=None):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT sequence,state,sha256 FROM events WHERE experiment_id=? ORDER BY sequence DESC LIMIT 1",
                (experiment_id,),
            ).fetchone()
            if (row is None and state != "registered") or (
                row is not None and state not in TRANSITIONS.get(row["state"], set())
            ):
                raise ValueError(f"Invalid research state transition to {state}")
            sequence = row["sequence"] + 1 if row else 0
            previous = row["sha256"] if row else "0" * 64
            data = payload or {}
            sha = identity(
                {
                    "experiment_id": experiment_id,
                    "sequence": sequence,
                    "state": state,
                    "payload": data,
                    "previous_sha256": previous,
                }
            )
            self.db.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?)",
                (experiment_id, sequence, state, canonical(data), previous, sha),
            )
            self.db.execute("COMMIT")
            return sha
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def register_roles(self, experiment_id, roles):
        """Prevent previously trained/visible items being relabeled as fresh hidden evidence."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            for role, keys in roles.items():
                for key in keys:
                    row = self.db.execute(
                        "SELECT role FROM prompt_roles WHERE prompt_sha256=?", (key,)
                    ).fetchone()
                    if row and row[0] != role:
                        raise ValueError("A previously registered prompt changed dataset role")
                    self.db.execute(
                        "INSERT OR IGNORE INTO prompt_roles VALUES(?,?,?)",
                        (key, role, experiment_id),
                    )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def consume_holdout(self, experiment_id, role, prompt_keys):
        """Consume before inference, including failed/interrupted looks. Never silently re-open."""
        if role not in ("reserved", "final", "robustness", "retention") or not prompt_keys:
            raise ValueError("A fixed nonempty hidden role is required")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.executemany(
                "INSERT INTO holdouts VALUES(?,?,?)",
                [(key, role, experiment_id) for key in prompt_keys],
            )
            self.db.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            self.db.execute("ROLLBACK")
            raise ValueError(
                "Reserved/final/retention prompt was already consumed; use a fresh independent dataset"
            ) from exc
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def record_result(self, experiment_id, *, context, family, useful, cost, summary):
        if (
            self.state(experiment_id) not in TERMINAL
            or type(useful) is not bool
            or not math.isfinite(cost)
            or cost < 0
        ):
            raise ValueError("Only terminal, finite, host-measured results enter strategy memory")
        if useful and (
            self.state(experiment_id) != "completed" or not summary.get("full_gate_passed")
        ):
            raise ValueError("Useful outcome requires the complete independently measured gate")
        cost = float(cost)
        binding = identity(
            {
                "context": context,
                "family": family,
                "useful": useful,
                "cost": cost,
                "summary": summary,
            }
        )
        row = self.db.execute(
            "SELECT payload FROM events WHERE experiment_id=? ORDER BY sequence DESC LIMIT 1",
            (experiment_id,),
        ).fetchone()
        if json.loads(row[0]).get("result_sha256") != binding:
            raise ValueError("Outcome must be bound to the terminal hash-chain event")
        self.db.execute(
            "INSERT INTO results VALUES(?,?,?,?,?,?)",
            (experiment_id, context, family, int(useful), cost, canonical(summary)),
        )

    def claim(self, experiment_id, claim, provenance):
        self.db.execute(
            "INSERT INTO claims VALUES(?,?,?,0)",
            (experiment_id, text(claim, "claim"), text(provenance, "provenance")),
        )

    def search(self, query="", *, limit=20):
        integer(limit, "history limit", 1, 100)
        # Literal substring matching, never model-generated SQL/FTS expressions.
        text_query = str(query)[:200]
        rows = self.db.execute(
            "SELECT e.id,e.plan_sha256,r.context,r.family,r.useful,r.cost,r.summary FROM results r JOIN experiments e ON e.id=r.experiment_id WHERE instr(r.summary,?)>0 ORDER BY e.created DESC LIMIT ?",
            (text_query, limit),
        )
        return [{**dict(row), "summary": json.loads(row["summary"])} for row in rows]

    def posterior(self, context, family):
        row = self.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(useful),0),COALESCE(AVG(cost),0) FROM results WHERE context=? AND family=?",
            (context, family),
        ).fetchone()
        n, successes, cost = row
        a, b = 1 + successes, 1 + n - successes
        mean = a / (a + b)
        variance = a * b / ((a + b) ** 2 * (a + b + 1))
        return {
            "alpha": a,
            "beta": b,
            "observations": n,
            "mean_verified_success": mean,
            "standard_deviation": math.sqrt(variance),
            "mean_cost_seconds": cost,
            "interpretation": "Beta-Bernoulli empirical strategy posterior, not a trained researcher or proof of improved future selection.",
        }

    def recover_interrupted(self, experiment_id):
        row = self.db.execute(
            "SELECT owner_pid,owner_token FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
        if not row or self.state(experiment_id) in TERMINAL:
            raise ValueError("Expected an unfinished registered experiment")
        token = process_token(row["owner_pid"])
        if token is not None and token == row["owner_token"]:
            raise ValueError("Original controller is still alive; recovery refused")
        self.transition(
            experiment_id,
            "interrupted",
            {
                "restoration_verified": False,
                "automatically_resumed": False,
                "reason": "Explicit operator recovery of an unfinished dead-controller record; held-out consumption preserved.",
            },
        )

    def verify(self):
        for row in self.db.execute("SELECT * FROM baselines"):
            if identity(json.loads(row["metadata"])) != row["id"]:
                raise ValueError("Baseline inventory hash changed")
        for row in self.db.execute("SELECT * FROM experiments"):
            plan = json.loads(row["plan"])
            if identity(plan) != row["plan_sha256"]:
                raise ValueError("Experiment plan hash changed")
            if (
                plan.get("baseline_id")
                and not self.db.execute(
                    "SELECT id FROM baselines WHERE id=?", (plan["baseline_id"],)
                ).fetchone()
            ):
                raise ValueError("Baseline registry record is missing")
            previous, state = "0" * 64, None
            for index, event in enumerate(
                self.db.execute(
                    "SELECT * FROM events WHERE experiment_id=? ORDER BY sequence", (row["id"],)
                )
            ):
                expected = identity(
                    {
                        "experiment_id": row["id"],
                        "sequence": index,
                        "state": event["state"],
                        "payload": json.loads(event["payload"]),
                        "previous_sha256": previous,
                    }
                )
                if (
                    event["sequence"] != index
                    or event["previous_sha256"] != previous
                    or event["sha256"] != expected
                ):
                    raise ValueError("Research event hash chain changed")
                if (state is None and event["state"] != "registered") or (
                    state is not None and event["state"] not in TRANSITIONS.get(state, set())
                ):
                    raise ValueError("Research event history has an invalid transition")
                previous, state = expected, event["state"]
            if state is None:
                raise ValueError("Registered experiment has no initial event")
            result = self.db.execute(
                "SELECT * FROM results WHERE experiment_id=?", (row["id"],)
            ).fetchone()
            if result:
                payload = json.loads(event["payload"])
                binding = identity(
                    {
                        "context": result["context"],
                        "family": result["family"],
                        "useful": bool(result["useful"]),
                        "cost": result["cost"],
                        "summary": json.loads(result["summary"]),
                    }
                )
                if state not in TERMINAL or payload.get("result_sha256") != binding:
                    raise ValueError("Recorded strategy result disagrees with its event binding")
        return True
