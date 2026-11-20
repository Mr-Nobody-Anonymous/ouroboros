"""Structural JSON grammar for editor replies; host policy remains the authority.

This is not a menu of pre-written edits: the model still generates the action, tensor names,
indices, numerical values and experimental choices. Shapes, protections, finite rounded budgets
and the reserved gate are checked independently by LabSession/WorkingCopy after decoding.
"""

from .tools import OPERATIONS, TOOLS

ACTION_SCHEMA_VERSION = 2


def _object(properties, required):
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _array(items, low=1, high=128):
    return {"type": "array", "items": items, "minItems": low, "maxItems": high}


def _string(high=512):
    return {"type": "string", "minLength": 1, "maxLength": high}


def _integer(low=0, high=1_000_000):
    return {"type": "integer", "minimum": low, "maximum": high}


def _enum(*values):
    return {"type": "string", "enum": list(values)}


def action_schema(*, sealed=False, context=None):
    """A bounded structural subset of the public language. Never bypass host validation.

    Array limits keep generated commands practical; the Python/scripted API can supply larger
    arrays within its existing limits. Only confirm/withdraw are grammatical after sealing.
    Numeric ranges are annotations too: not every decoder enforces JSON Schema bounds.
    """
    context = {} if context is None else context

    def choices(values, fallback):
        # Avoid building enormous enum tries for user-raised 100B/50k-example profiles.
        # Large domains still receive independent host checks, not a bogus format guarantee.
        return {**fallback, "enum": list(values)} if 0 < len(values) <= 4096 else fallback

    def bounded_integer(low, high):
        return choices(range(low, high + 1), _integer(low, high))

    name = choices(context.get("parameters", []), _string())
    module_name = choices(context.get("modules", []), _string())
    text = _string(4000)
    number = {"type": "number"}
    indices = _array(_integer(), high=256)
    vector = _array(number, high=4096)
    matrix = _array(_array(number, high=4096), high=4096)
    known_directions = context.get("directions")
    direction_id = choices(known_directions or [], _string(32))
    direction = vector if known_directions == [] else {"anyOf": [direction_id, vector]}
    operation_fields = {
        "name": name,
        "direction": direction,
        "strength": {"type": "number", "minimum": -8, "maximum": 8},
        "columns": indices,
        "top_columns": _integer(1),
        "max_entries": _integer(1, 4_000_000),
        "factor": number,
        "indices": indices,
        "values": vector,
        "value": number,
        "left": matrix,
        "right": matrix,
    }
    operations = []
    for kind, spec in OPERATIONS.items():
        fields = spec["required"] + spec.get("optional", [])
        operations.append(
            _object(
                {"op": _enum(kind), **{key: operation_fields[key] for key in fields}},
                ["op", *spec["required"]],
            )
        )
    program = _array({"anyOf": operations}, high=16)
    fields = {
        "kind": _string(32),
        "offset": _integer(0, 100_000),
        "limit": bounded_integer(1, 100),
        "prefix": {"type": "string", "maxLength": 200},
        "names": _array(name, low=0, high=128),
        "values": bounded_integer(0, 256),
        "weights": _enum("original", "edited"),
        "texts": _array(_string(32000), high=256),
        "modules": _array(module_name, high=16),
        "pooling": _enum("mean", "positions"),
        "last": bounded_integer(0, context.get("max_read_tokens", 8192)),
        "stride": bounded_integer(1, context.get("max_read_tokens", 8192)),
        "module": module_name,
        "failing": _array(_string(32000), high=128),
        "successful": _array(_string(32000), high=128),
        "name": name,
        "direction": direction_id,
        "operations": program,
        "replace": {"type": "boolean"},
        "ids": _array(
            choices(range(context["practice_count"]), _integer(0, 49_999))
            if "practice_count" in context
            else _integer(0, 49_999),
            high=512,
        ),
        "saved": _array(_string(32), low=0, high=4),
        "seed": _integer(0, 2**63 - 1),
        "variants": _array(
            _object({"label": _string(32), "operations": program}, ["label", "operations"]),
            high=4,
        ),
        "text": text,
    }
    actions = []
    for kind, spec in TOOLS.items():
        if sealed and kind not in ("confirm", "withdraw"):
            continue
        if not sealed and kind == "confirm":
            continue
        if kind == "writers" and known_directions == []:
            continue
        if kind == "restore" and context.get("saved") == []:
            continue
        keys = spec.get("required", []) + spec.get("optional", [])
        properties = {"action": _enum(kind), **{key: fields[key] for key in keys}}
        if kind in ("inventory", "modules", "practice"):
            count_key = {
                "inventory": "tensor_count",
                "modules": "module_count",
                "practice": "practice_count",
            }[kind]
            if count_key in context:
                properties["offset"] = bounded_integer(0, max(0, context[count_key] - 1))
            if kind == "practice":
                properties["limit"] = bounded_integer(1, context.get("max_problems_per_call", 100))
        if kind == "writers":
            properties["limit"] = bounded_integer(1, 256)
        if kind == "read":
            properties["kind"] = _enum("hidden", "mlp_input", "attention", "logprobs")
        if kind == "control":
            properties["kind"] = _enum(
                "no_op", "random_direction", "random_delta", "layer_isolation"
            )
        if kind == "test":
            properties["weights"] = _enum("original", "edited", "both")
            if "saved" in context:
                properties["saved"] = _array(
                    choices(context["saved"], _string(32)),
                    low=0,
                    high=min(4, len(context["saved"])),
                )
        if kind == "drop" and "edited" in context:
            properties["names"] = _array(
                choices(context["edited"], _string()),
                low=0,
                high=min(128, len(context["edited"])),
            )
        if kind in ("save", "restore"):
            properties["name"] = _string(32)
        if kind == "restore" and "saved" in context:
            properties["name"] = choices(context["saved"], _string(32))
        properties["rationale"] = {"type": "string", "maxLength": 240}
        actions.append(_object(properties, ["action", *spec.get("required", [])]))
    return {"anyOf": actions}


def require_structured_decoder():
    """Explicit opt-in dependency; never silently fall back to unconstrained generation."""
    try:
        from lmformatenforcer import JsonSchemaParser  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "JSON-schema editor decoding needs: python -m pip install 'ouroboros[structured]' "
            "(or use scripts/install.py --profile research). No fallback editor ran."
        ) from exc
