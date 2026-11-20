"""Question-span guards are limited leakage screens, not semantic/pretraining decontamination."""

import pytest

from ouroboros.data import Sample
from ouroboros.research.contracts import ROLES, ResearchData
from ouroboros.research.data_quality import question_overlap


def roles():
    return {role: [Sample(role + " prompt", "shared class label", verified=True)] for role in ROLES}


def test_long_question_ngram_contamination_is_blocked_before_any_model_load():
    values = roles()
    common = "one two three four five six seven eight nine ten"
    values["train"] = [Sample("Training prefix " + common + " suffix", "yes", verified=True)]
    values["reserved"] = [Sample(common + " hidden suffix", "yes", verified=True)]
    audit = question_overlap(values)
    assert audit["collisions"][0]["matched_words"] == 10
    assert not audit["semantic_decontamination"] and not audit["answer_text_screened"]
    with pytest.raises(ValueError, match="word spans"):
        ResearchData(values)


def test_short_complete_question_substrings_and_role_boundaries():
    values = roles()
    values["practice"] = [Sample("Before this hidden short question after", "yes", verified=True)]
    values["final"] = [Sample("this hidden short question", "yes", verified=True)]
    assert question_overlap(values)["collisions"][0]["matched_words"] == 4
    with pytest.raises(ValueError, match="word spans"):
        ResearchData(values)


def test_shared_answer_labels_and_short_template_fragments_are_not_removed():
    values = roles()
    assert not question_overlap(values)["collisions"]
    assert not ResearchData(values).quality["collisions"]


def test_overlap_screen_stops_at_first_witness_not_quadratic_pairs():
    common = "one two three four five six seven eight nine ten"
    values = roles()
    values["train"] = [Sample(common + str(i), "yes", verified=True) for i in range(50)]
    values["reserved"] = [
        Sample(common + " hidden " + str(i), "yes", verified=True) for i in range(50)
    ]
    # The numeric suffix on word ten changes the token, so use a separate word boundary.
    values["train"] = [Sample(common + " train " + str(i), "yes", verified=True) for i in range(50)]
    result = question_overlap(values)
    assert len(result["collisions"]) == 1 and result["checks"] == 1
    assert result["stops_at_first_collision"]
