"""Command grammar/config/diagnostic checks without loading or downloading a model."""

import json
import operator
import re

import pytest

from ouroboros.lab.action_schema import action_schema, require_structured_decoder
from ouroboros.lab.config import normalize_editor
from ouroboros.lab.diagnosis import diagnosis_preview, practice_diagnosis, text_preview
from ouroboros.lab.metrics import problem_id, summarize
from ouroboros.lab.tools import OPERATIONS, TOOLS, capabilities, compact_guide


def test_schema_and_compact_guide_cover_the_closed_language_without_importing_decoder():
    schema = action_schema()
    actions = {item["properties"]["action"]["enum"][0]: item for item in schema["anyOf"]}
    assert set(actions) == set(TOOLS) - {"confirm"}
    for name, spec in actions.items():
        assert not spec["additionalProperties"]
        assert spec["required"] == ["action", *TOOLS[name].get("required", [])]
        assert set(spec["properties"]) == {
            "action",
            "rationale",
            *TOOLS[name].get("required", []),
            *TOOLS[name].get("optional", []),
        }
    operations = actions["edit"]["properties"]["operations"]["items"]["anyOf"]
    assert {item["properties"]["op"]["enum"][0] for item in operations} == set(OPERATIONS)
    assert all(
        not item["additionalProperties"] and item["required"][0] == "op" for item in operations
    )
    sealed = action_schema(sealed=True)
    assert {item["properties"]["action"]["enum"][0] for item in sealed["anyOf"]} == {
        "confirm",
        "withdraw",
    }
    assert len(json.dumps(compact_guide())) < len(json.dumps(capabilities())) / 3
    assert set(compact_guide()["actions"]) == set(TOOLS)


@pytest.mark.parametrize(
    "action",
    [
        {"action": "withdraw"},
        {"action": "test", "ids": [0, 1], "weights": "both"},
        {"action": "contrast_practice", "module": "model.layers.0", "ids": [0, 1]},
        {
            "action": "edit",
            "operations": [{"op": "add", "name": "a.weight", "value": 0.01, "indices": [0]}],
        },
    ],
)
def test_real_optional_character_parser_accepts_generated_command_shapes(action):
    enforcer = pytest.importorskip("lmformatenforcer")
    parser = enforcer.JsonSchemaParser(
        action_schema(), config=enforcer.CharacterLevelParserConfig(force_json_field_order=True)
    )
    for char in json.dumps(action, separators=(",", ":")):
        assert char in parser.get_allowed_characters()
        parser = parser.add_character(char)
    assert parser.can_end()


@pytest.mark.parametrize(
    "text",
    [
        '{"action":"unknown"}',
        '{"action":"withdraw","path":"/tmp/file"}',
        '{"action":"confirm"}',
        '{"action":"test"}',
        '{"action":"test","ids":[true]}',
        '{"action":"edit","operations":[{"op":"python","name":"a"}]}',
    ],
)
def test_real_optional_character_parser_does_not_admit_unknown_actions_fields_or_types(text):
    enforcer = pytest.importorskip("lmformatenforcer")
    parser = enforcer.JsonSchemaParser(
        action_schema(), config=enforcer.CharacterLevelParserConfig(force_json_field_order=True)
    )
    accepted = True
    for char in text:
        if char not in parser.get_allowed_characters():
            accepted = False
            break
        parser = parser.add_character(char)
    assert not accepted or not parser.can_end()


@pytest.mark.parametrize(
    "editor",
    [
        {"mode": "other"},
        {"max_new_tokens": True},
        {"decoding": "auto"},
        {"mode": "model", "actions": "actions.json"},
        {"decoding": "json_schema", "sampling": {"thinking": True}},
        {"sampling": {"temperature": True}},
        {"sampling": {"unknown": 2}},
        {"sampling": []},
        {"mode": "scripted", "decoding": "json_schema"},
        {"python": "print(1)"},
        [],
    ],
)
def test_editor_config_rejects_ambiguous_or_unsafe_fields(editor):
    with pytest.raises(ValueError):
        normalize_editor(editor)


def test_editor_defaults_are_independent_and_optional_dependency_has_no_silent_fallback(
    monkeypatch,
):
    import sys

    editor = normalize_editor({"decoding": "json_schema"})
    assert editor["sampling"]["temperature"] == 0 and not editor["sampling"]["thinking"]
    assert editor["decoding"] == "json_schema" and editor["max_new_tokens"] == 512
    monkeypatch.setitem(sys.modules, "lmformatenforcer", None)
    with pytest.raises(RuntimeError, match="No fallback editor ran"):
        require_structured_decoder()


def _measurement(samples, outcomes):
    rows = [
        {
            "id": problem_id(sample),
            "sample": 0,
            "seed": i,
            "domain": sample.domain,
            "text": "actual completion " + str(i),
            "correct": correct,
            "cut": cut,
            "new_tokens": 128 if cut else 3,
            "repetitive": False,
            "thinking_closed": True,
        }
        for i, (sample, (correct, cut)) in enumerate(zip(samples, outcomes))
    ]
    return {"records": rows, "summary": summarize(rows)}


def test_readiness_preserves_strict_scores_and_both_measured_outcome_types():
    from ouroboros.data import Sample

    samples = [Sample("practice " + str(i), str(i)) for i in range(8)]
    measured = _measurement(samples, [(False, True)] * 6 + [(True, False)] * 2)
    diagnosis = practice_diagnosis(samples, measured, answer_tokens=128)
    readiness = diagnosis["readiness"]
    assert readiness["signal"] == "mixed_correct_and_incorrect"
    assert readiness["correct_answers"] == 2 and readiness["cut_answers"] == 6
    assert readiness["measured_contrast_available"]
    assert diagnosis["summary"]["accuracy"] == 0.25  # no lenient replacement
    preview = diagnosis_preview(diagnosis)
    assert any(row["correct"] for p in preview["practice"] for row in p["answers"])
    assert any(not row["correct"] for p in preview["practice"] for row in p["answers"])
    assert len(preview["practice"]) == 6 and preview["omitted_problems"] == 2
    assert any("completion limit" in warning for warning in readiness["warnings"])
    for outcomes, signal in (
        ([(False, False)] * 8, "all_incorrect"),
        ([(True, False)] * 8, "all_correct"),
    ):
        poor = practice_diagnosis(samples, _measurement(samples, outcomes), answer_tokens=128)
        assert poor["readiness"]["signal"] == signal
        assert not poor["readiness"]["measured_contrast_available"]
        assert poor["readiness"]["warnings"]
    assert text_preview("HEAD" + "x" * 2000 + "FINAL ANSWER").endswith("FINAL ANSWER")


def test_research_resources_are_fixed_disjoint_arithmetic_and_legacy_data_are_unchanged(tmp_path):
    from ouroboros.lab.config import load_lab_config
    from ouroboros.project import init_project

    root = tmp_path / "lab"
    init_project(root)
    config = load_lab_config(root / "research_lab.json")
    assert config["model"].max_new_tokens == 128
    assert config["policy"].max_turns >= 16 and config["policy"].max_invalid_actions == 4
    assert config["editor"]["decoding"] == "json_schema"
    assert config["policy"].diagnosis_problems == len(config["dataset"].train) == 18
    functions = {"+": operator.add, "-": operator.sub, "*": operator.mul}
    for split in ("train", "validation", "test"):
        samples = getattr(config["dataset"], split)
        assert len(samples) == 18
        for sample in samples:
            if sample.domain == "copying":
                assert (
                    sample.instruction
                    == "Repeat the following word exactly and output nothing else: " + sample.target
                )
                assert sample.verified
                continue
            match = re.fullmatch(
                r"What is (\d+) ([+*-]) (\d+)\? Reply with only the integer\.", sample.instruction
            )
            assert sample.target == str(functions[match[2]](int(match[1]), int(match[3])))
            assert sample.verified
    assert len(json.loads((root / "practice.json").read_text())) == 2  # legacy self-edit fixture


@pytest.mark.parametrize(
    "text,valid",
    [
        ('{"action":"inventory","offset":-1}', False),
        ('{"action":"inventory","offset":8}', False),
        ('{"action":"inventory","offset":0}', True),
        ('{"action":"inventory","limit":0}', False),
        ('{"action":"test","ids":[-1]}', False),
        ('{"action":"test","ids":[3]}', False),
        ('{"action":"test","ids":[0,2]}', True),
        ('{"action":"weights","names":["unknown"]}', False),
        ('{"action":"weights","names":["a.weight"]}', True),
        ('{"action":"contrast_practice","module":"unknown","ids":[0]}', False),
        ('{"action":"contrast_practice","module":"a","ids":[0]}', True),
        ('{"action":"writers","name":"a.weight","direction":"d0"}', False),
        ('{"action":"restore","name":"missing"}', False),
    ],
)
def test_context_bound_grammar_rejects_real_negative_offset_bug_and_unknown_public_ids(text, valid):
    enforcer = pytest.importorskip("lmformatenforcer")
    context = {
        "tensor_count": 8,
        "module_count": 2,
        "practice_count": 3,
        "parameters": ["a.weight"],
        "modules": ["a"],
        "directions": [],
        "saved": [],
        "edited": [],
        "max_problems_per_call": 3,
        "max_read_tokens": 32,
    }
    parser = enforcer.JsonSchemaParser(
        action_schema(context=context),
        config=enforcer.CharacterLevelParserConfig(force_json_field_order=True),
    )
    accepted = True
    for char in text:
        if char not in parser.get_allowed_characters():
            accepted = False
            break
        parser = parser.add_character(char)
    assert (accepted and parser.can_end()) == valid


def test_context_schema_uses_no_large_unbounded_enum_allocation():
    schema = action_schema(context={"practice_count": 50_000, "tensor_count": 100_000})
    by_action = {
        item["properties"]["action"]["enum"][0]: item["properties"] for item in schema["anyOf"]
    }
    assert "enum" not in by_action["test"]["ids"]["items"]
    assert "enum" not in by_action["inventory"]["offset"]  # host still checks large domains
    small = action_schema(context={"practice_count": 3, "directions": ["d0"], "saved": ["best"]})
    by_action = {
        item["properties"]["action"]["enum"][0]: item["properties"] for item in small["anyOf"]
    }
    assert by_action["writers"]["direction"]["enum"] == ["d0"]
    assert by_action["restore"]["name"]["enum"] == ["best"]
