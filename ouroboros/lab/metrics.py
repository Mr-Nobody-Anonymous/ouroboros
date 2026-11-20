"""Paired, problem-clustered measurements. These tests do not manufacture a capability gain."""

import hashlib
import math
import zlib
from collections import defaultdict

import numpy as np

from ..data import normalize, prompt_key
from .config import integer, strict_json


def problem_id(sample):
    return prompt_key(sample.instruction)


def sample_seed(identifier, sample_index, seed_base):
    payload = f"{seed_base}:{identifier}:{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2**31


def last_boxed(text):
    """Read balanced braces, not a greedy regex or a Python/math expression evaluator."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    begin, depth = start + len("\\boxed{"), 1
    for index in range(begin, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:index].strip()
    return None


def require_grader(grader):
    if grader == "boxed_math":
        try:
            import math_verify  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("boxed_math needs: python -m pip install 'ouroboros[math]'") from exc
    elif grader != "exact":
        raise ValueError("Unknown grader")


def grade(record, target, grader, sampling):
    """Cut/unclosed-thinking answers fail closed. Never silently fall back to a lenient score."""
    if record["cut"] or not record["thinking_closed"]:
        return False
    answer = record["text"]
    if sampling.thinking:
        answer = answer.rsplit(sampling.thinking_end, 1)[-1]
    if grader == "exact":
        return normalize(answer) == normalize(target)
    require_grader(grader)
    from math_verify import LatexExtractionConfig, parse, verify

    boxed = last_boxed(answer)
    if boxed is None:
        return False
    # Use only the final box, and an explicitly selected library grader, not eval/sympify here.
    prediction = parse("$" + boxed + "$", extraction_config=[LatexExtractionConfig()])
    reference = parse(
        "$" + (last_boxed(target) or target) + "$", extraction_config=[LatexExtractionConfig()]
    )
    return bool(prediction and reference and verify(reference, prediction))


def repetition(text, cut):
    """Paper's zlib tail heuristic; short completions should not be called diagnosed loops."""
    tail = text[-6000:].encode("utf-8")
    ratio = len(zlib.compress(tail)) / len(tail) if tail else 1.0
    return bool(cut and ratio < 0.15), ratio


def evaluate(backend, samples, *, sampling, grader, samples_per_problem, seed, max_new_tokens=None):
    if not samples:
        raise ValueError("Nonempty evaluation data required")
    requests = [
        {"prompt": sample.instruction, "seed": sample_seed(problem_id(sample), j, seed)}
        for sample in samples
        for j in range(samples_per_problem)
    ]
    generated = backend.generate_records(requests, sampling=sampling, max_new_tokens=max_new_tokens)
    if not isinstance(generated, list) or len(generated) != len(requests):
        raise ValueError("Backend returned the wrong number of generation records")
    rows = []
    for index, record in enumerate(generated):
        sample, sample_index = samples[index // samples_per_problem], index % samples_per_problem
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("text"), str)
            or type(record.get("new_tokens")) is not int
            or record["new_tokens"] < 0
            or type(record.get("cut")) is not bool
            or type(record.get("thinking_closed")) is not bool
        ):
            raise ValueError("Invalid generation record")
        looping, ratio = repetition(record["text"], record["cut"])
        row = {
            "id": problem_id(sample),
            "sample": sample_index,
            "seed": requests[index]["seed"],
            "domain": sample.domain,
            "text": record["text"],
            "new_tokens": record["new_tokens"],
            "cut": record["cut"],
            "thinking_closed": record["thinking_closed"],
            "repetitive": looping,
            "tail_compression_ratio": ratio,
        }
        row["correct"] = grade(row, sample.target, grader, sampling)
        rows.append(row)
    return {"summary": summarize(rows), "records": rows}


def summarize(rows):
    if not rows:
        raise ValueError("Nonempty measured records required")
    domains = defaultdict(list)
    for row in rows:
        domains[row["domain"]].append(int(row["correct"]))
    scores = {key: sum(values) / len(values) for key, values in sorted(domains.items())}
    cut = [row for row in rows if row["cut"]]
    finished = [row for row in rows if not row["cut"] and row["thinking_closed"]]
    return {
        "problems": len({row["id"] for row in rows}),
        "answers": len(rows),
        "accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
        "scores": scores,
        "macro_accuracy": sum(scores.values()) / len(scores),
        "cut_share": len(cut) / len(rows),
        "repetitive_share_of_cuts": sum(int(row["repetitive"]) for row in cut) / len(cut)
        if cut
        else 0.0,
        "finished_accuracy": sum(int(row["correct"]) for row in finished) / len(finished)
        if finished
        else None,
        "mean_new_tokens": sum(row["new_tokens"] for row in rows) / len(rows),
    }


def _difference(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    se = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else None
    return {
        "mean": mean,
        "standard_error": se,
        "normal_95_interval": [max(-1.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)]
        if se is not None
        else None,
    }


def sign_flip_test(differences, *, seed=42, draws=10_000):
    """One-sided sign flips over PROBLEMS, never treating repeated answers as independent."""
    integer("permutation draws", draws, 100, 100_000)
    data = np.asarray(differences, dtype=np.float64)
    if data.ndim != 1 or not data.size or not np.isfinite(data).all() or np.any(np.abs(data) > 1):
        raise ValueError("Invalid per-problem differences")
    nonzero = data[data != 0]
    if not nonzero.size:
        return {"p_value": 1.0, "method": "exact-problem-sign-flips", "draws": 1}
    observed, total, extreme = float(nonzero.sum()), 0, 0
    exact = nonzero.size <= 16
    rng = np.random.default_rng(seed)
    count = 2**nonzero.size if exact else draws
    for start in range(0, count, 256):
        size = min(256, count - start)
        if exact:
            bits = (
                np.arange(start, start + size, dtype=np.uint64)[:, None]
                >> np.arange(nonzero.size, dtype=np.uint64)
            ) & 1
            signs = bits.astype(np.int8) * 2 - 1
        else:
            signs = rng.integers(0, 2, size=(size, nonzero.size), dtype=np.int8) * 2 - 1
        extreme += int(np.count_nonzero(signs @ nonzero >= observed - 1e-12))
        total += size
    return {
        "p_value": extreme / total if exact else (extreme + 1) / (total + 1),
        "method": "exact-problem-sign-flips"
        if exact
        else "monte-carlo-problem-sign-flips-plus-one",
        "draws": total,
    }


def paired_summary(baseline, candidate, *, seed=42, draws=10_000):
    before, after = baseline["records"], candidate["records"]
    if not before or len(before) != len(after):
        raise ValueError("Nonempty equal-length paired evaluations required")
    groups = defaultdict(lambda: {"correct": [], "cut": []})
    rescued = regressed = 0
    cohorts = {"originally_cut": [0, 0, 0], "originally_finished": [0, 0, 0]}
    for left, right in zip(before, after):
        if any(left[key] != right[key] for key in ("id", "sample", "seed", "domain")):
            raise ValueError("Problem/sample/seed/domain mismatch in paired evaluation")
        groups[left["id"]]["correct"].append(int(right["correct"]) - int(left["correct"]))
        groups[left["id"]]["cut"].append(int(right["cut"]) - int(left["cut"]))
        rescued += int(not left["correct"] and right["correct"])
        regressed += int(left["correct"] and not right["correct"])
        cohort = cohorts["originally_cut" if left["cut"] else "originally_finished"]
        cohort[0] += 1
        cohort[1] += int(left["correct"])
        cohort[2] += int(right["correct"])
    correct = [sum(group["correct"]) / len(group["correct"]) for group in groups.values()]
    cuts = [sum(group["cut"]) / len(group["cut"]) for group in groups.values()]
    result = {
        "baseline": baseline["summary"],
        "candidate": candidate["summary"],
        "accuracy_delta": _difference(correct),
        "cut_share_delta": _difference(cuts),
        "wrong_to_right": rescued,
        "right_to_wrong": regressed,
        "problem_sign_flip": sign_flip_test(correct, seed=seed, draws=draws),
        "unit_of_analysis": "problem (samples averaged within each problem)",
        "interval_note": "Normal approximation over problems, not a simultaneous or sequential-testing guarantee.",
        "cohorts": {
            key: {"answers": n, "baseline_correct": a, "candidate_correct": b}
            for key, (n, a, b) in cohorts.items()
        },
    }
    if all(len(group["correct"]) == 1 for group in groups.values()):
        discordant = rescued + regressed
        # lgamma avoids constructing giant integers for large external evaluations.
        p = (
            sum(
                math.exp(
                    math.lgamma(discordant + 1)
                    - math.lgamma(k + 1)
                    - math.lgamma(discordant - k + 1)
                    - discordant * math.log(2)
                )
                for k in range(rescued, discordant + 1)
            )
            if discordant
            else 1.0
        )
        result["paired_sign_test_one_sided"] = min(1.0, p)
    return result


def parse_action(text):
    if not isinstance(text, str) or len(text) > 64 * 1024:
        raise ValueError("Expected an action JSON object of at most 64 KiB")
    value = text.strip()
    if value.startswith("```json\n") and value.endswith("\n```"):
        value = value[8:-4]
    result = strict_json(value)
    if not isinstance(result, dict) or not isinstance(result.get("action"), str):
        raise ValueError("Reply with exactly one JSON object containing action")
    return result
