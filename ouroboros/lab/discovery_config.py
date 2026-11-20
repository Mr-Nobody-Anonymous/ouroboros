"""Optional stricter discovery protocol: five disjoint roles, no held-out editor access."""

import hashlib
import json

from ..data import prompt_key, unique_samples

DISCOVERY_SCHEMA_VERSION = 1


def validate_discovery(spec, dataset, policy):
    required = {"development", "retention", "followup_operations"}
    optional = {
        "gradient_access",
        "measure_loss",
        "feedback",
        "prompt_style",
        "operation_scope",
        "workflow",
    }
    if not isinstance(spec, dict) or not required <= set(spec) or set(spec) - required - optional:
        raise ValueError("Discovery requires development, retention and fixed followup_operations")
    if not policy.require_significance:
        raise ValueError(
            "Discovery requires significance gates; loss-only improvements cannot be promoted"
        )
    if policy.max_sessions != 1:
        raise ValueError(
            "Discovery uses one independent session per arm; preregister separate seeds/arms"
        )
    if policy.forgetting_threshold != 0:
        raise ValueError("Discovery requires zero per-domain forgetting tolerance")
    result = {
        "gradient_access": False,
        "measure_loss": True,
        "feedback": True,
        "prompt_style": "compact",
        "operation_scope": "all",
        "workflow": "free",
        **spec,
    }
    for key in ("gradient_access", "measure_loss", "feedback"):
        if type(result[key]) is not bool:
            raise ValueError(f"discovery.{key} must be boolean")
    if not result["measure_loss"]:
        raise ValueError("Discovery baseline and candidate completion NLL measurement is required")
    if result["prompt_style"] not in ("compact", "full") or result["operation_scope"] not in (
        "all",
        "sparse",
    ):
        raise ValueError("Choose compact/full prompt style and all/sparse operation scope")
    if result["workflow"] not in ("free", "staged"):
        raise ValueError("Discovery workflow is free or explicitly host-staged")
    domains = {s.domain for s in dataset.train}
    seen = set()
    for name, rows in (
        ("practice", dataset.train),
        ("development", result["development"]),
        ("reserved", dataset.validation),
        ("final", dataset.test),
        ("retention", result["retention"]),
    ):
        if not isinstance(rows, list) or not 1 <= len(rows) <= policy.max_examples_per_split:
            raise ValueError(
                f"Discovery {name} split must be nonempty and within its example budget"
            )
        if len(unique_samples(rows)) != len(rows):
            raise ValueError(f"Discovery {name} contains duplicate prompts")
        if {s.domain for s in rows} != domains:
            raise ValueError(
                "Each discovery role must contain the same categories for retention checks"
            )
        keys = {prompt_key(s.instruction) for s in rows}
        if seen & keys:
            raise ValueError("Discovery splits must be mutually prompt-disjoint")
        seen |= keys
    operations = result["followup_operations"]
    if (
        not isinstance(operations, list)
        or not 1 <= len(operations) <= 16
        or any(not isinstance(op, dict) for op in operations)
    ):
        raise ValueError("Predeclare 1-16 data-only followup_operations, not a callback")
    return result


def discovery_summary(spec):
    if spec is None:
        return None
    rows = {key: [s.as_dict() for s in spec[key]] for key in ("development", "retention")}
    return {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        **{key: value for key, value in spec.items() if key not in rows},
        "counts": {key: len(value) for key, value in rows.items()},
        "hidden_roles_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest(),
        "boundary": "Practice alone drives search. Submit first seals development then reserved checks. Final and retention never select or feed back to the editor. Fixed subsequent edits are host interventions, not autonomous meta-improvement.",
    }
