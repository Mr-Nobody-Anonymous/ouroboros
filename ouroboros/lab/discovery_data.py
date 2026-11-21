"""Reproducible, response-independent selection of a broader small evaluation.

GSM8K is adapted from the explicitly cloned MIT source. We never execute its scripts, use
rationales as editor observations, or claim pretrained decontamination. Synthetic tasks are
owned diagnostics, not independent benchmark evidence. Final roles include shifted templates.
"""

import hashlib
import json
import re
from pathlib import Path

from ..core.checkpoints import atomic_json, file_sha256
from ..data import Sample, prompt_key

GSM8K_SOURCE = {
    "repository": "https://github.com/openai/grade-school-math.git",
    "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
    "license": "MIT, Copyright (c) 2021 OpenAI",
    "files": {
        "grade_school_math/data/train.jsonl": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
        "grade_school_math/data/test.jsonl": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        "LICENSE": "86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b",
    },
}
COUNTS = {"practice": 8, "development": 8, "reserved": 12, "final": 16, "retention": 16}
DOMAINS = ("arithmetic", "copying", "compositional_reasoning", "gsm8k")
SALT = "ouroboros-discovery-v1-2026-11-19"


def rank(value):
    return hashlib.sha256(f"{SALT}:{value}".encode()).hexdigest()


def _arithmetic(far):
    rows = []
    for a in range(30, 91) if far else range(1, 26):
        for b in range(1, 26):
            if not far and a > b:
                continue
            for op in ("+", "*"):
                key = f"arithmetic:{a}:{op}:{b}"
                question = f"Compute {a} {op} {b}. Reply with only the integer."
                if far:
                    question = (
                        f"Return only the integer result, without explanation: ({a} {op} {b}) = ?"
                    )
                rows.append((rank(key), question, str(a + b if op == "+" else a * b)))
    return sorted(rows)


def _copying(far):
    rows = []
    for i in range(500):
        digest = rank(f"copy:{int(far)}:{i}")
        value = "".join(chr(97 + int(c, 16)) for c in digest[:4])
        if far:
            value = f"{value.upper()}{i:03d}"
        question = f"Copy this text exactly, with no extra words: {value}"
        if far:
            question = f"Return exactly the text inside the tags, no tags or explanation. <item>{value}</item>"
        rows.append((digest, question, value))
    # Collision rejection is independent of model responses.
    unique = {row[1]: row for row in sorted(rows, reverse=True)}
    return sorted(unique.values())


def _reasoning(far):
    rows = []
    for a in range(2, 16):
        for b in range(2, 10):
            for c in range(1, 4):
                key = f"reasoning:{int(far)}:{a}:{b}:{c}"
                question = f"There are {a} bags with {b} marbles each. Then {c} marbles are removed. How many marbles remain? Reply only with the integer."
                answer = a * b - c
                if far:
                    question = f"A shop has {a} boxes, each holding {b} pens. It sells {c} pens and receives 7 more pens. How many pens does it have now? Output only the integer."
                    answer += 7
                rows.append((rank(key), question, str(answer)))
    return sorted(rows)


def prepare_discovery_data(clone, directory):
    clone, directory = Path(clone), Path(directory)
    for name, sha in GSM8K_SOURCE["files"].items():
        path = clone / name
        if path.is_symlink() or not path.is_file() or file_sha256(path) != sha:
            raise ValueError(f"Pinned GSM8K data/license hash mismatch: {name}")
    if directory.exists():
        raise FileExistsError("Refuse to overwrite a prepared evaluation; choose a new directory")
    directory.mkdir(parents=True)
    splits = {key: {domain: [] for domain in DOMAINS} for key in COUNTS}
    source_rows = {key: [] for key in COUNTS}
    for domain, factory in (
        ("arithmetic", _arithmetic),
        ("copying", _copying),
        ("compositional_reasoning", _reasoning),
    ):
        for far, roles in (
            (False, ("practice", "development", "reserved")),
            (True, ("final", "retention")),
        ):
            pool = factory(far)
            offset = 0
            for role in roles:
                for _, question, target in pool[offset : offset + COUNTS[role]]:
                    splits[role][domain].append(
                        Sample(
                            question,
                            target,
                            domain,
                            f"owned-discovery-v1:{'shifted' if far else 'in-distribution'}",
                            True,
                        )
                    )
                offset += COUNTS[role]
    for upstream, roles in (
        ("train", ("practice", "development", "reserved")),
        ("test", ("final", "retention")),
    ):
        path = clone / f"grade_school_math/data/{upstream}.jsonl"
        pool = []
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            row = json.loads(line)
            final = row["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
            if (
                "####" not in row["answer"]
                or len(row["question"]) > 800
                or not re.fullmatch(r"-?\d+(?:\.\d+)?", final)
            ):
                continue
            pool.append(
                (
                    rank(f"gsm8k:{upstream}:{row['question']}"),
                    index,
                    row["question"],
                    final,
                    hashlib.sha256(line.encode()).hexdigest(),
                )
            )
        pool.sort()
        offset = 0
        for role in roles:
            for _, index, question, target, sha in pool[offset : offset + COUNTS[role]]:
                prompt = f"Solve this problem. Give only the final number, without explanation or commas.\n{question}"
                splits[role]["gsm8k"].append(
                    Sample(
                        prompt,
                        target,
                        "gsm8k",
                        f"GSM8K@{GSM8K_SOURCE['revision']}/{upstream}:{index}",
                        True,
                    )
                )
                source_rows[role].append(
                    {"upstream_split": upstream, "zero_based_line": index, "raw_row_sha256": sha}
                )
            offset += COUNTS[role]
    all_keys, files = set(), {}
    for role, categories in splits.items():
        # Interleave categories so a small diagnosis prefix is not an arithmetic-only sample.
        rows = [categories[domain][i] for i in range(COUNTS[role]) for domain in DOMAINS]
        keys = {prompt_key(row.instruction) for row in rows}
        if len(keys) != len(rows) or keys & all_keys:
            raise ValueError("Prepared split contamination")
        all_keys |= keys
        atomic_json(directory / f"{role}.json", [row.as_dict() for row in rows])
        files[f"{role}.json"] = file_sha256(directory / f"{role}.json")
    (directory / "GSM8K_LICENSE.txt").write_text((clone / "LICENSE").read_text(), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "kind": "ouroboros-discovery-data",
        "source": GSM8K_SOURCE,
        "files": files,
        "counts": {key: value * len(DOMAINS) for key, value in COUNTS.items()},
        "domain_counts_per_role": COUNTS,
        "domains": list(DOMAINS),
        "gsm8k_selected_rows": source_rows,
        "selection": "Fixed SHA256 ordering with salt "
        + SALT
        + "; GSM8K questions <=800 characters, numeric #### targets (commas removed); no model response, length in model tokens or performance-based filtering. Practice/development/reserved use disjoint upstream train rows; final/retention disjoint upstream test rows. Synthetic problems ranked independently; final/retention use shifted templates/content/ranges.",
        "scoring": "Strict NFKC/case/whitespace-normalized exact match over entire completion; length cuts or unclosed thinking fail. NLL separately includes reference+EOS and never substitutes for accuracy.",
        "limitations": "240 selected diagnostics, of which 60 are independently authored GSM8K problems; synthetic tasks are not an external benchmark. Prompt-disjoint is not semantic or pretrained decontamination. Final/retention distribution shifts are deliberate, not evidence of generalization until measured.",
    }
    atomic_json(directory / "manifest.json", manifest)
    return manifest
