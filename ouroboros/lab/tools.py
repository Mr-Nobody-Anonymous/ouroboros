"""The closed research action language shared by the model, Python API, CLI and MCP catalogue."""

TOOLS = {
    "inventory": {
        "optional": ["offset", "limit"],
        "purpose": "Page canonical parameter names/shapes/protections.",
    },
    "modules": {
        "optional": ["prefix", "offset", "limit"],
        "purpose": "Page named modules for hidden/MLP/attention hooks.",
    },
    "practice": {
        "optional": ["offset", "limit"],
        "purpose": "Read practice prompts and references only; never reserved or test data.",
    },
    "weights": {
        "optional": ["names", "limit", "values", "weights"],
        "purpose": "Target-copy numeric statistics and bounded prefixes; sort by outlier ratio.",
    },
    "read": {
        "required": ["kind", "texts"],
        "optional": ["modules", "weights", "pooling", "last", "stride"],
        "purpose": "Read kind hidden, mlp_input, attention or logprobs on model-chosen raw texts.",
    },
    "contrast": {
        "required": ["module", "failing", "successful"],
        "optional": ["weights", "last"],
        "purpose": "Store a unit failing-minus-successful hidden-state mean direction, referenced as d0, d1, ...",
    },
    "contrast_practice": {
        "required": ["module", "ids"],
        "optional": ["weights", "last"],
        "purpose": "Generate and grade practice answers, then measure actual failing-minus-correct completions. No invented positive/negative examples; unavailable when either cohort is empty.",
    },
    "writers": {
        "required": ["name", "direction"],
        "optional": ["limit"],
        "purpose": "Rank output columns by absolute projection on a measured direction id.",
    },
    "edit": {
        "required": ["operations"],
        "optional": ["replace"],
        "purpose": "Stage a validated multi-tensor data-only program, atomically. Default updates current edit; replace=true starts from original target.",
    },
    "drop": {
        "optional": ["names"],
        "purpose": "Drop some/all edited tensors without changing the resident editor.",
    },
    "save": {
        "required": ["name"],
        "purpose": "Save one of at most four in-memory candidate variants.",
    },
    "restore": {
        "required": ["name"],
        "purpose": "Restore a saved candidate, not a new session base.",
    },
    "test": {
        "required": ["ids"],
        "optional": ["weights", "saved"],
        "purpose": "Generate practice ids (zero-based indices), original/edited/both; optionally up to four saved edits, paired on identical per-problem seeds.",
    },
    "control": {
        "required": ["ids", "kind"],
        "optional": ["seed", "names"],
        "purpose": "Paired no_op, random_direction, random_delta or layer_isolation control; leaves candidate unchanged.",
    },
    "sweep": {
        "required": ["ids", "variants"],
        "purpose": "Up to four independent lists of operations, each replacing the current edit for a paired dose/layer test. Candidate unchanged.",
    },
    "notes": {
        "required": ["text"],
        "purpose": "Keep bounded notes across context compaction; data, not executable instructions.",
    },
    "submit": {
        "purpose": "Seal edit and run one reserved validation check. Afterwards only confirm/withdraw is permitted."
    },
    "confirm": {
        "purpose": "Submit sealed edit; host still rejects plateaus/regressions/failing statistical gates."
    },
    "withdraw": {
        "purpose": "Abstain/drop edit and end session, including after the reserved check."
    },
}

OPERATIONS = {
    "project": {
        "required": ["name", "direction", "strength"],
        "optional": ["columns", "top_columns", "max_entries"],
        "formula": "W[:,J] -= strength * u[:,None] * (u @ W[:,J]); u normalized. Signed strength in [-8,8]. Direction = measured id or finite numeric vector.",
    },
    "clip_outliers": {
        "required": ["name", "factor"],
        "formula": "clip W to +/- factor * median(abs(W)); no clean reference used.",
    },
    "set_entries": {
        "required": ["name", "indices", "values"],
        "formula": "Replace explicit flat indices with finite values; dense tensors of rank 1-32. Matrix-specific operators still require rank 2.",
    },
    "add": {
        "required": ["name", "value"],
        "optional": ["indices"],
        "formula": "Add a finite scalar to selected flat indices, or all entries (e.g. time-step biases).",
    },
    "scale_rows": {
        "required": ["name", "indices", "factor"],
        "formula": "Scale selected matrix output rows; whole-tensor scaling is rejected.",
    },
    "scale_columns": {
        "required": ["name", "indices", "factor"],
        "formula": "Scale selected matrix input columns; whole-tensor scaling is rejected.",
    },
    "low_rank_add": {
        "required": ["name", "left", "right"],
        "formula": "W += left @ right; rank <=16, explicit finite factor arrays.",
    },
}


def capabilities(*, discovery=False):
    from .action_schema import action_schema

    if discovery:
        from .discovery_session import DiscoverySession
        from .journal import SUCCESS_LEVELS

        return {
            "kind": "ouroboros-discovery-tools",
            "schema_version": 1,
            "actions": DiscoverySession.tools,
            "weight_operations": OPERATIONS,
            "editor_action_schema": action_schema(tools=DiscoverySession.tools),
            "sealed_action_schema": action_schema(sealed=True, tools=DiscoverySession.tools),
            "evidence_levels": SUCCESS_LEVELS,
            "opt_in": "Add a discovery section with separately approved development/retention paths and a fixed followup edit to a lab config. Requires strict significance and zero domain forgetting. Gradients must be explicitly enabled and are not paper-parity/no-gradient runs.",
        }
    return {
        "kind": "ouroboros-research-tools",
        "schema_version": 2,
        "actions": TOOLS,
        "editor_action_schema": action_schema(),
        "sealed_action_schema": action_schema(sealed=True),
        "decoding_note": "Optional json_schema filters model tokens, not edit choices or promotion. Structural grammar is a subset; all host shape/protection/numeric/budget checks still run.",
        "weight_operations": OPERATIONS,
        "action_format": "One JSON object: action plus required/optional fields; optional rationale string <=2000 chars. No Python or file paths.",
        "weights": "original = starting target (damaged in repair), edited = current candidate; neither mutates editor reasoning weights.",
        "pooling": "mean or positions; last=0 means all token positions, stride>=1. Reads are bounded and explicitly report token truncation.",
        "example": {
            "action": "edit",
            "operations": [
                {
                    "op": "project",
                    "name": "model.layers.0.mlp.down_proj.weight",
                    "direction": "d0",
                    "strength": 0.5,
                    "top_columns": 4,
                }
            ],
        },
        "not_implemented": [
            "arbitrary persistent Python execution or an OS sandbox",
            "vLLM/tensor-parallel serving, quantized-weight mutation or vision model loading",
            "autonomous mastery of every WeightBench/blog task or demonstrated paper benchmark gains",
        ],
    }


def compact_guide():
    """Signatures, not the complete schema/catalogue, belong in the model's limited context."""
    return {
        "actions": {
            name: ",".join(spec.get("required", []))
            + (" [" + ",".join(spec["optional"]) + "]" if spec.get("optional") else "")
            for name, spec in TOOLS.items()
        },
        "operations": {
            name: ",".join(spec["required"])
            + (" [" + ",".join(spec["optional"]) + "]" if spec.get("optional") else "")
            for name, spec in OPERATIONS.items()
        },
        "semantics": "[] denotes optional fields, not literal JSON. ids are zero-based practice indices. weights=original/edited/both (both only for test). read.kind=hidden/mlp_input/attention/logprobs. project subtracts strength times the measured direction's column projection; use top_columns or columns. All edits remain temporary and must pass strict reserved improvement/retention.",
        "examples": [
            {"action": "inventory", "limit": 4},
            {"action": "test", "ids": [0, 1], "weights": "both"},
            {"action": "withdraw"},
        ],
    }
