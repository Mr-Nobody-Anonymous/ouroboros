"""Strict, offline research configuration, separate from the deliberately tiny smoke profile."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import ModelConfig, probability
from ..core.checkpoints import file_sha256
from ..data import DatasetBundle, load_samples
from ..self_edit import scoped_path


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    # parse_constant does not catch overflowing finite-looking literals such as 1e999.
    json.dumps(result, allow_nan=False)
    return result


def integer(name, value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def number(name, value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    thinking: bool = False
    thinking_end: str = "</think>"

    def __post_init__(self):
        number("temperature", self.temperature, 0, 4)
        number("top_p", self.top_p, 0.001, 1)
        integer("top_k", self.top_k, 0, 1000)
        number("repetition_penalty", self.repetition_penalty, 0.1, 4)
        number("presence_penalty", self.presence_penalty, -2, 2)
        if type(self.thinking) is not bool:
            raise ValueError("thinking must be boolean")
        if not isinstance(self.thinking_end, str) or not 1 <= len(self.thinking_end) <= 64:
            raise ValueError("thinking_end must be a bounded nonempty marker")


def normalize_editor(editor=None):
    """Keep editor decoding/sampling separate from answer generation, for API and CLI alike."""
    editor = (
        {"mode": "model", **({} if editor is None else editor)}
        if isinstance(editor, (dict, type(None)))
        else editor
    )
    allowed = {"mode", "max_new_tokens", "actions", "decoding", "sampling"}
    if not isinstance(editor, dict) or set(editor) - allowed:
        raise ValueError("Invalid editor fields")
    if editor.get("mode") not in ("model", "scripted"):
        raise ValueError(
            "editor.mode must be model or scripted; scripted is NOT autonomous self-editing"
        )
    integer("editor.max_new_tokens", editor.get("max_new_tokens", 512), 16, 12000)
    decoding = editor.get("decoding", "unconstrained")
    if decoding not in ("unconstrained", "json_schema"):
        raise ValueError("editor.decoding must be unconstrained or json_schema")
    values = editor.get("sampling", {})
    if not isinstance(values, dict):
        raise ValueError("editor.sampling must be a Sampling object encoded as JSON")
    try:
        sampling = Sampling(**values)
    except TypeError as exc:
        raise ValueError(f"Invalid editor.sampling: {exc}") from exc
    if decoding == "json_schema" and sampling.thinking:
        raise ValueError(
            "JSON-schema commands require editor.sampling.thinking=false; answer thinking is separate"
        )
    if editor["mode"] == "scripted" and decoding != "unconstrained":
        raise ValueError("Scripted actions are not schema-decoded model replies")
    if editor["mode"] == "model" and "actions" in editor:
        raise ValueError("Model editor cannot be given scripted actions")
    return {"max_new_tokens": 512, **editor, "decoding": decoding, "sampling": asdict(sampling)}


@dataclass(frozen=True)
class LabPolicy:
    max_sessions: int = 1
    max_turns: int = 24
    max_forward_calls: int = 20
    max_edit_actions: int = 40
    max_seconds: int = 300
    max_invalid_actions: int = 3
    max_repeated_actions: int = 3
    max_model_parameters: int = 300_000_000
    max_model_bytes: int = 1024**3
    max_matrix_elements: int = 8_000_000
    max_edit_tensors: int = 8
    max_changed_entries: int = 250_000
    max_edited_fraction: float = 0.001
    max_relative_norm: float = 0.1
    max_patch_bytes: int = 16 * 1024**2
    max_examples_per_split: int = 3000
    max_problems_per_call: int = 16
    samples_per_problem: int = 1
    test_samples_per_problem: int = 1
    diagnosis_problems: int = 2
    max_read_tokens: int = 256
    max_read_texts: int = 8
    max_read_elements: int = 1_000_000
    max_logit_elements: int = 16_000_000
    max_history_chars: int = 16_000
    forgetting_threshold: float = 0.0
    improvement_threshold: float = 0.0
    require_significance: bool = False
    alpha: float = 0.05
    permutation_draws: int = 10_000
    attention_maps: bool = False
    seed: int = 42
    test_seed: int = 20261119
    cpu_threads: int = 2

    def __post_init__(self):
        bounds = {
            "max_sessions": (1, 8),
            "max_turns": (1, 160),
            "max_forward_calls": (1, 100),
            "max_edit_actions": (1, 200),
            "max_seconds": (1, 86_400),
            "max_invalid_actions": (1, 20),
            "max_repeated_actions": (2, 20),
            "max_model_parameters": (1, 100_000_000_000),
            "max_model_bytes": (1, 256 * 1024**3),
            "max_matrix_elements": (1, 512_000_000),
            "max_edit_tensors": (1, 128),
            "max_changed_entries": (1, 4_000_000),
            "max_patch_bytes": (1024, 64 * 1024**2),
            "max_examples_per_split": (1, 50_000),
            "max_problems_per_call": (1, 512),
            "samples_per_problem": (1, 16),
            "test_samples_per_problem": (1, 16),
            "diagnosis_problems": (1, 256),
            "max_read_tokens": (2, 8192),
            "max_read_texts": (1, 256),
            "max_read_elements": (1, 16_000_000),
            "max_logit_elements": (1, 64_000_000),
            "max_history_chars": (2000, 360_000),
            "permutation_draws": (100, 100_000),
            "cpu_threads": (1, 32),
            "seed": (0, 2**63 - 1),
            "test_seed": (0, 2**63 - 1),
        }
        for key, (low, high) in bounds.items():
            integer(key, getattr(self, key), low, high)
        for key in ("max_edited_fraction", "forgetting_threshold", "improvement_threshold"):
            probability(key, getattr(self, key))
        number("max_relative_norm", self.max_relative_norm, 0, 2)
        number("alpha", self.alpha, 0.00001, 0.5)
        if self.max_edited_fraction == 0:
            raise ValueError("max_edited_fraction must be positive")
        if self.seed == self.test_seed:
            raise ValueError("Final-test seed must differ from the session/validation seed")
        for key in ("attention_maps", "require_significance"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if self.samples_per_problem * self.max_problems_per_call > 512:
            raise ValueError("A practice call may request at most 512 answers per weight set")


def load_lab_config(path, *, workspace=None):
    path = scoped_path(workspace, str(path)) if workspace is not None else Path(path).resolve()
    if path.stat().st_size > 64 * 1024:
        raise ValueError("Lab config exceeds 64 KiB")
    values = strict_json(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "model",
        "policy",
        "sampling",
        "editor",
        "grader",
        "data",
        "output_dir",
        "license",
        "provenance",
    }
    if (
        not isinstance(values, dict)
        or not required <= set(values)
        or set(values) - required - {"damage"}
    ):
        raise ValueError("Lab config requires: " + ", ".join(sorted(required)))
    if type(values["schema_version"]) is not int or values["schema_version"] != 1:
        raise ValueError("Unsupported lab schema_version")
    for key in ("license", "provenance"):
        if not isinstance(values[key], str) or not 1 <= len(values[key].strip()) <= 4000:
            raise ValueError(f"Explicit bounded {key} required")

    def resolve(value):
        if not isinstance(value, str) or not value:
            raise ValueError("Nonempty config path required")
        candidate = path.parent / Path(value).expanduser()
        return str(
            scoped_path(workspace, str(candidate)) if workspace is not None else candidate.resolve()
        )

    try:
        model_values = dict(values["model"])
        model_values["model_name_or_path"] = resolve(model_values.get("model_name_or_path"))
        model = ModelConfig(**model_values)
        policy, sampling = LabPolicy(**values["policy"]), Sampling(**values["sampling"])
    except TypeError as exc:
        raise ValueError(f"Invalid lab model/policy/sampling: {exc}") from exc
    if model.use_qlora or model.max_new_tokens > 32768:
        raise ValueError("Lab requires ordinary floating weights and at most 32768 answer tokens")
    if values["grader"] not in ("exact", "boxed_math"):
        raise ValueError("grader must be exact or boxed_math (optional math-verify dependency)")
    editor = normalize_editor(values["editor"])
    input_hashes = {str(path): file_sha256(path)}
    actions = None
    if editor["mode"] == "scripted":
        action_path = Path(resolve(editor.get("actions")))
        if action_path.stat().st_size > 1024**2:
            raise ValueError("Scripted actions exceed 1 MiB")
        actions = strict_json(action_path.read_text(encoding="utf-8"))
        if not isinstance(actions, list) or not 1 <= len(actions) <= policy.max_turns:
            raise ValueError("Scripted actions must be a nonempty list within the turn budget")
        input_hashes[str(action_path)] = file_sha256(action_path)
    elif "actions" in editor:
        raise ValueError("Model editor cannot be given scripted actions")
    data = values["data"]
    if not isinstance(data, dict) or set(data) != {"train", "validation", "test"}:
        raise ValueError("Separate train/practice, reserved validation and final test required")
    splits = {}
    for name in ("train", "validation", "test"):
        data_path = Path(resolve(data[name]))
        if data_path.stat().st_size > 64 * 1024**2:
            raise ValueError("Lab data file exceeds 64 MiB")
        rows = load_samples(data_path)
        if not 1 <= len(rows) <= policy.max_examples_per_split or any(
            len(s.instruction) > 32000 or len(s.target) > 4000 for s in rows
        ):
            raise ValueError("Lab split exceeds its configured example/character budget")
        splits[name] = rows
        input_hashes[str(data_path)] = file_sha256(data_path)
    dataset = DatasetBundle(splits["train"], splits["validation"], test=splits["test"])
    damage = values.get("damage")
    if damage is not None:
        keys = {"parameter_name", "first_row", "rows", "std_multiplier", "seed"}
        if not isinstance(damage, dict) or set(damage) != keys:
            raise ValueError(
                "Damage requires parameter_name, first_row, rows, std_multiplier and seed"
            )
        if not isinstance(damage["parameter_name"], str) or not damage["parameter_name"]:
            raise ValueError("Damage parameter_name is required")
        integer("damage.first_row", damage["first_row"], 0, 1_000_000)
        integer("damage.rows", damage["rows"], 1, 128)
        integer("damage.seed", damage["seed"], 0, 2**63 - 1)
        number("damage.std_multiplier", damage["std_multiplier"], 1, 1000)
    return {
        "model": model,
        "policy": policy,
        "sampling": sampling,
        "editor": editor,
        "actions": actions,
        "grader": values["grader"],
        "dataset": dataset,
        "output_dir": resolve(values["output_dir"]),
        "license_name": values["license"],
        "provenance": values["provenance"],
        "damage": damage,
        "input_hashes": input_hashes,
    }


def lab_config_summary(config):
    return {
        "valid": True,
        "model_loaded": False,
        "network_accessed": False,
        "model": asdict(config["model"]),
        "policy": asdict(config["policy"]),
        "sampling": asdict(config["sampling"]),
        "editor_mode": config["editor"]["mode"],
        "editor_decoding": config["editor"]["decoding"],
        "editor_sampling": config["editor"]["sampling"],
        "grader": config["grader"],
        "damaged_copy_experiment": config["damage"] is not None,
        "counts": {
            key: len(getattr(config["dataset"], key)) for key in ("train", "validation", "test")
        },
        "dataset_sha256": config["dataset"].fingerprint(),
        "note": "Schema/split check only; not a model/memory/license/sandbox or benchmark verification.",
    }


def input_fingerprint(config):
    """Bind config, split and scripted-action bytes for MCP approval; NOT a hostile-FS guarantee."""
    actual = {name: file_sha256(name) for name in config["input_hashes"]}
    if actual != config["input_hashes"]:
        raise ValueError("Lab input changed during validation")
    return hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
