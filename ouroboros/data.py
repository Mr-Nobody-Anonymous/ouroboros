"""Local training data, provenance, exact deduplication, and split-contamination checks."""

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prompt_key(instruction: str) -> str:
    return hashlib.sha256(normalize(instruction).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Sample:
    instruction: str
    target: str
    domain: str = "general"
    source: str = "local"
    verified: bool = False
    is_anchor: bool = False

    def __post_init__(self):
        for name in ("instruction", "target", "domain", "source"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"Sample.{name} must be a nonempty string")
        if not isinstance(self.verified, bool) or not isinstance(self.is_anchor, bool):
            raise ValueError("verified and is_anchor must be booleans")
        if self.is_anchor and not self.verified:
            raise ValueError("Anchor examples must be explicitly verified")

    @classmethod
    def from_dict(cls, row, *, source="local", anchor=False):
        if not isinstance(row, dict):
            raise ValueError("Each sample must be an object")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(f"Unknown sample fields: {sorted(unknown)}")
        values = {"source": source, **row}
        if anchor:
            values["is_anchor"] = True
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"Invalid sample: {exc}") from exc

    def as_dict(self):
        return asdict(self)


def load_samples(path, *, anchors=False) -> list[Sample]:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in stream if line.strip()]
        else:
            rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON array or JSONL records")
    return [Sample.from_dict(row, source=str(path), anchor=anchors) for row in rows]


def unique_samples(samples: Iterable[Sample]) -> list[Sample]:
    result, seen = [], {}
    for sample in samples:
        key = prompt_key(sample.instruction)
        if key in seen:
            if normalize(seen[key].target) != normalize(sample.target):
                raise ValueError(f"Conflicting targets for prompt: {sample.instruction!r}")
            continue
        seen[key] = sample
        result.append(sample)
    return result


@dataclass
class DatasetBundle:
    train: list[Sample]
    validation: list[Sample]
    anchors: list[Sample] = field(default_factory=list)

    def __post_init__(self):
        self.train = unique_samples(self.train)
        self.validation = unique_samples(self.validation)
        self.anchors = unique_samples(self.anchors)
        if not self.train or not self.validation:
            raise ValueError("Nonempty train and validation splits are required")
        unique_samples([*self.train, *self.anchors])  # detect conflicting replay labels
        training_keys = {prompt_key(s.instruction) for s in [*self.train, *self.anchors]}
        for sample in self.validation:
            if prompt_key(sample.instruction) in training_keys:
                raise ValueError(f"Train/validation contamination: {sample.instruction!r}")
        for anchor in self.anchors:
            if not anchor.verified:
                raise ValueError("Replay anchors must be explicitly verified")
        train_domains = {s.domain for s in [*self.train, *self.anchors]}
        if not {s.domain for s in self.validation} <= train_domains:
            raise ValueError("Every validation domain must have training or anchor examples")

    def fingerprint(self):
        payload = {
            name: [s.as_dict() for s in getattr(self, name)]
            for name in ("train", "validation", "anchors")
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def curate_proposals(
    proposals: Iterable[Sample],
    held_out: Iterable[Sample],
    verifier: Callable[[Sample], bool],
    max_chars=10000,
) -> list[Sample]:
    """Accept externally generated data only with an independent verifier; never self-label by style."""
    blocked = {prompt_key(s.instruction) for s in held_out}
    accepted = []
    for sample in unique_samples(proposals):
        if prompt_key(sample.instruction) in blocked:
            continue
        if len(sample.instruction) + len(sample.target) > max_chars:
            continue
        if verifier(sample):
            accepted.append(Sample(**{**sample.as_dict(), "verified": True}))
    return accepted


def demo_dataset() -> DatasetBundle:
    """Tiny color-to-label classification fixture, NOT an LLM or a math benchmark."""
    colors = {"red": "warm", "orange": "warm", "blue": "cool", "green": "cool"}
    train, validation = [], []
    for color, target in colors.items():
        domain = "warm_colors" if target == "warm" else "cool_colors"
        for template in ("classify {color}", "label {color}", "category {color}"):
            train.append(Sample(template.format(color=color), target, domain, "demo-fixture", True))
        validation.append(Sample(f"please classify {color}", target, domain, "demo-fixture", True))
    anchors = [
        Sample("red category please", "warm", "warm_colors", "demo-fixture", True, True),
        Sample("blue category please", "cool", "cool_colors", "demo-fixture", True, True),
    ]
    return DatasetBundle(train, validation, anchors)


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", normalize(text))
