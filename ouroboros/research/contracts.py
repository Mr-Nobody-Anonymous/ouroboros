"""Versioned research contracts; all model output is untrusted data, never executable code."""

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..data import load_samples, prompt_key
from ..lab.config import strict_json

VERSION = 1
ROLES = (
    "train",
    "practice",
    "validation",
    "reserved",
    "final",
    "robustness",
    "retention",
    "followup",
)
FAMILIES = ("lora_sft", "full_sft", "data_reweight", "distill_sft")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def text(value, name, limit=4000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be nonempty text of at most {limit} characters")
    return value


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return value


def from_dict(cls, value):
    if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
        raise ValueError(f"Unknown/non-object {cls.__name__} fields")
    try:
        return cls(**value)
    except TypeError as exc:
        raise ValueError(f"Invalid {cls.__name__}: {exc}") from exc


@dataclass(frozen=True)
class Budget:
    max_steps: int = 100
    max_seconds: int = 1800
    max_rss_bytes: int = 1800 * 1024**2
    max_seq_length: int = 128
    max_new_tokens: int = 32
    max_parameters: int = 160_000_000
    max_full_parameters: int = 1_000_000
    max_train_examples: int = 512
    max_total_steps: int = 1200
    max_candidates: int = 2
    repeats: int = 2
    checkpoint_every: int = 25
    max_checkpoint_exports: int = 64
    max_output_bytes: int = 512 * 1024**2
    cpu_threads: int = 2
    family_size: int = 44

    def __post_init__(self):
        for name, low, high in (
            ("max_steps", 1, 10000),
            ("max_seconds", 1, 86400),
            ("max_rss_bytes", 64 * 1024**2, 64 * 1024**3),
            ("max_seq_length", 8, 2048),
            ("max_new_tokens", 1, 512),
            ("max_parameters", 1, 2_000_000_000),
            ("max_full_parameters", 0, 100_000_000),
            ("max_train_examples", 1, 100000),
            ("max_total_steps", 1, 100000),
            ("max_candidates", 1, 4),
            ("repeats", 2, 4),
            ("checkpoint_every", 1, 10000),
            ("max_checkpoint_exports", 1, 256),
            ("max_output_bytes", 32 * 1024**2, 64 * 1024**3),
            ("cpu_threads", 1, 32),
            ("family_size", 44, 100000),
        ):
            integer(getattr(self, name), name, low, high)

    @property
    def alpha(self):
        return 0.05 / self.family_size


@dataclass(frozen=True)
class Recipe:
    family: str = "lora_sft"
    hypothesis_index: int = 0
    learning_rate: float = 0.001
    steps: int = 50
    rank: int = 4
    alpha: int = 8
    optimizer: str = "adamw"
    schedule: str = "constant"
    weight_decay: float = 0.0
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    domain_weights: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError("Unsupported intervention family; no silent substitute")
        integer(self.hypothesis_index, "hypothesis_index", 0, 3)
        number(self.learning_rate, "learning_rate", 1e-7, 0.05)
        number(self.weight_decay, "weight_decay", 0, 1)
        integer(self.steps, "steps", 1, 10000)
        integer(self.rank, "rank", 1, 64)
        integer(self.alpha, "alpha", 1, 128)
        if self.optimizer not in ("adamw", "sgd") or self.schedule not in ("constant", "cosine"):
            raise ValueError("Unsupported optimizer/schedule")
        if not isinstance(self.target_modules, list) or not 1 <= len(self.target_modules) <= 8:
            raise ValueError("Expected 1–8 target module names")
        if any(
            not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_]+", v)
            for v in self.target_modules
        ):
            raise ValueError("Target modules must be simple module names, not expressions")
        if not isinstance(self.domain_weights, dict) or len(self.domain_weights) > 32:
            raise ValueError("Invalid training-domain weights")
        if self.family == "data_reweight" and not self.domain_weights:
            raise ValueError("A data_reweight intervention needs explicit domain weights")
        for key, value in self.domain_weights.items():
            text(key, "domain", 100)
            integer(value, "domain repetition weight", 1, 4)

    def validate_budget(self, budget):
        if self.steps > budget.max_steps:
            raise ValueError("Proposed training exceeds the immutable step budget")


@dataclass(frozen=True)
class Hypothesis:
    problem: str
    explanation: str
    confidence: float
    supporting_evidence: list
    contradicting_evidence: list
    alternatives: list
    missing_information: list
    prediction: float
    falsification: str

    def __post_init__(self):
        for key in ("problem", "explanation", "falsification"):
            text(getattr(self, key), key)
        number(self.confidence, "confidence", 0, 1)
        number(self.prediction, "predicted practice accuracy delta", 1e-9, 1)
        for key in (
            "supporting_evidence",
            "contradicting_evidence",
            "alternatives",
            "missing_information",
        ):
            values = getattr(self, key)
            if not isinstance(values, list) or len(values) > 32:
                raise ValueError(f"{key} must be a bounded list")
            for value in values:
                text(value, key, 1000)
        if not self.supporting_evidence or not self.alternatives:
            raise ValueError("Evidence and competing explanations are required")


@dataclass(frozen=True)
class Proposal:
    schema_version: int
    action: str
    reason: str
    hypotheses: list = field(default_factory=list)
    recipes: list = field(default_factory=list)

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != VERSION:
            raise ValueError("Unsupported research schema version")
        if self.action not in ("experiment", "abstain"):
            raise ValueError("Research action must be experiment or abstain")
        text(self.reason, "reason")
        if not isinstance(self.hypotheses, list) or not isinstance(self.recipes, list):
            raise ValueError("Hypotheses and recipes must be lists")
        if self.action == "abstain":
            if self.hypotheses or self.recipes:
                raise ValueError("Abstention cannot smuggle an intervention")
        elif not 1 <= len(self.hypotheses) <= 4 or not 1 <= len(self.recipes) <= 4:
            raise ValueError("An experiment needs 1–4 competing hypotheses and recipes")
        for value in self.hypotheses:
            from_dict(Hypothesis, value)
        for value in self.recipes:
            if from_dict(Recipe, value).hypothesis_index >= len(self.hypotheses):
                raise ValueError("Recipe must bind an existing prospective hypothesis")

    @classmethod
    def parse(cls, value):
        if isinstance(value, str) and len(value) > 64 * 1024:
            raise ValueError("Research proposal exceeds 64 KiB")
        return from_dict(cls, strict_json(value) if isinstance(value, str) else value)

    def validate_context(self, evidence_ids, budget, domains):
        for raw in self.hypotheses:
            hypothesis = from_dict(Hypothesis, raw)
            if not set(hypothesis.supporting_evidence + hypothesis.contradicting_evidence) <= set(
                evidence_ids
            ):
                raise ValueError(
                    "Hypothesis cites evidence absent from the measured practice context"
                )
        for raw in self.recipes:
            recipe = from_dict(Recipe, raw)
            recipe.validate_budget(budget)
            if not set(recipe.domain_weights) <= set(domains):
                raise ValueError("Curriculum references an unknown training domain")


class ResearchData:
    """Eight fixed roles; no normalization-based claim of semantic decontamination."""

    def __init__(self, roles):
        if set(roles) != set(ROLES):
            raise ValueError(
                "Expected train/practice/validation/reserved/final/robustness/retention/followup"
            )
        self.roles, seen = {}, {}
        for name in ROLES:
            samples = tuple(roles[name])
            if not samples or len(samples) > 3000:
                raise ValueError(f"Nonempty {name} role with at most 3000 examples required")
            for sample in samples:
                if len(sample.instruction) + len(sample.target) > 16000:
                    raise ValueError("Research example exceeds 16000 characters")
                key = prompt_key(sample.instruction)
                if key in seen:
                    raise ValueError(f"Duplicate/contaminated prompt across {seen[key]} and {name}")
                seen[key] = name
                if name in ("train", "followup") and not sample.verified:
                    raise ValueError(
                        "Training/follow-up labels need explicit provenance verification"
                    )
            self.roles[name] = samples
        domains = {s.domain for s in self.roles["train"]}
        if len(domains) > 32:
            raise ValueError("At most 32 domains are supported in the bounded foundation")
        if any({s.domain for s in rows} != domains for rows in self.roles.values()):
            raise ValueError("All roles must cover the same nonempty domains")
        if sum(len(s.instruction.split()) for rows in self.roles.values() for s in rows) > 250000:
            raise ValueError("Question-overlap screening exceeds its 250000-word allocation budget")
        from .data_quality import question_overlap

        self.quality = question_overlap(self.roles)
        if self.quality["collisions"]:
            raise ValueError(
                "Visible train/practice/followup questions overlap hidden question word spans; no hidden tuning allowed"
            )

    def fingerprint(self):
        return identity({name: [s.as_dict() for s in rows] for name, rows in self.roles.items()})

    def role_keys(self, role):
        return [prompt_key(s.instruction) for s in self.roles[role]]

    @classmethod
    def load(cls, paths):
        from .provenance import regular

        for path in paths.values():
            if regular(path).stat().st_size > 16 * 1024**2:
                raise ValueError("Research data file exceeds 16 MiB")
        return cls({key: load_samples(path) for key, path in paths.items()})


def load_config(path):
    path = Path(path).resolve()
    if path.stat().st_size > 1024**2:
        raise ValueError("Research config exceeds 1 MiB")
    value = strict_json(path.read_text())
    required = {
        "schema_version",
        "model",
        "data",
        "budget",
        "researcher",
        "output_dir",
        "database",
        "seed",
        "followup_recipe",
    }
    if (
        not isinstance(value, dict)
        or set(value) - required - {"teacher", "checkpoint_selection"}
        or required - set(value)
        or type(value["schema_version"]) is not int
        or value["schema_version"] != VERSION
    ):
        raise ValueError("Expected the versioned research-foundation configuration")
    from ..config import ModelConfig

    model = from_dict(ModelConfig, value["model"])
    if model.use_qlora or model.device != "cpu" or not model.model_name_or_path:
        raise ValueError(
            "Foundation workers currently require an explicit CPU, nonquantized local model"
        )
    value["model"]["model_name_or_path"] = str((path.parent / model.model_name_or_path).absolute())
    if value.get("teacher") is not None:
        teacher = from_dict(ModelConfig, value["teacher"])
        if teacher.use_qlora or teacher.device != "cpu" or not teacher.model_name_or_path:
            raise ValueError("Teacher must be an explicit nonquantized CPU local model")
        value["teacher"]["model_name_or_path"] = str(
            (path.parent / teacher.model_name_or_path).absolute()
        )
    selection = value.get("checkpoint_selection", "final_only")
    if selection not in ("final_only", "practice_macro_accuracy"):
        raise ValueError("Unknown checkpoint-selection rule; declare it before the run")
    value["checkpoint_selection"] = selection
    if not isinstance(value["data"], dict) or set(value["data"]) != set(ROLES):
        raise ValueError("Expected all eight explicit research data paths")
    value["data"] = {
        key: str((path.parent / text(item, "data path", 4096)).absolute())
        for key, item in value["data"].items()
    }
    for key in ("output_dir", "database"):
        value[key] = str((path.parent / text(value[key], key, 4096)).absolute())
    integer(value["seed"], "seed", 0, 2**31 - 1)
    from_dict(Budget, value["budget"])
    from_dict(Recipe, value["followup_recipe"]).validate_budget(from_dict(Budget, value["budget"]))
    researcher = value["researcher"]
    if not isinstance(researcher, dict) or researcher.get("kind") not in (
        "host_reference",
        "local_model",
    ):
        raise ValueError("Explicit host_reference or local_model researcher required; no fallback")
    if set(researcher) - {"kind", "model_path", "max_new_tokens"}:
        raise ValueError("Unknown researcher fields")
    if researcher["kind"] == "local_model":
        researcher["model_path"] = str(
            (
                path.parent / researcher.get("model_path", value["model"]["model_name_or_path"])
            ).absolute()
        )
        integer(researcher.get("max_new_tokens", 256), "researcher token budget", 32, 512)
    return value, ResearchData.load(value["data"])


def commands():
    return {
        "schema_version": VERSION,
        "commands": {
            name: {"authority": authority, "description": description}
            for name, authority, description in (
                (
                    "inspect_model",
                    "host_read_only",
                    "Architecture, source hashes and parameter metadata",
                ),
                (
                    "evaluate_model",
                    "host",
                    "Real practice accuracy/NLL, never self-declared scores",
                ),
                (
                    "analyze_failures",
                    "research",
                    "Evidence-based error categories, not causal diagnoses",
                ),
                (
                    "generate_hypotheses",
                    "research",
                    "Competing uncertain explanations or abstention",
                ),
                (
                    "search_experiments",
                    "read_only_memory",
                    "Bounded verified history, no hidden examples",
                ),
                (
                    "design_experiment",
                    "research",
                    "Versioned proposal; fixed gates remain host-owned",
                ),
                (
                    "propose_intervention",
                    "research",
                    "Numeric recipes constrained by the immutable budget",
                ),
                (
                    "validate_intervention",
                    "host",
                    "Types, provenance, source/data/compute constraints",
                ),
                (
                    "run_experiment",
                    "approved_host",
                    "Fixed local workers; no generated code execution",
                ),
                ("compare_candidates", "host", "Same data and grading, actual paired outcomes"),
                (
                    "evaluate_generalization",
                    "sealed_host",
                    "Once-only reserved/final looks, no reselection",
                ),
                (
                    "record_result",
                    "host_only",
                    "Authoritative execution record, not model narrative",
                ),
                (
                    "update_strategy",
                    "host_only",
                    "Posterior from verified outcomes; not an RL-trained researcher",
                ),
                (
                    "rollback_candidate",
                    "host_only",
                    "Discard candidate worker; never overwrite baseline",
                ),
            )
        },
        "executable_families": list(FAMILIES),
        "arbitrary_python": False,
    }
