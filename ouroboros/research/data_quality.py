"""Bounded question-overlap screening inspired by DeepSeekMath §2.1.

This is not semantic/pretraining decontamination or the paper's entire corpus pipeline. It
screens normalized whitespace-word question spans only; shared short class labels are legal.
"""

from ..data import normalize
from .contracts import identity


def question_overlap(roles, *, n=10):
    visible = ("train", "practice", "followup")
    hidden = tuple(role for role in roles if role not in visible)
    spans = {}
    # At most one witness per span and one collision, not a quadratic pair inventory.
    for role in hidden:
        for sample in roles[role]:
            words = normalize(sample.instruction).split()
            width = min(n, len(words))
            if width < 3:
                continue
            identifier = identity(sample.instruction)
            for offset in range(len(words) - width + 1):
                key = tuple(words[offset : offset + width])
                spans.setdefault((width, key), (role, identifier))
    result = {
        "method": "normalized-question-only-word-ngrams-v1",
        "n": n,
        "short_question_minimum_words": 3,
        "checks": 0,
        "collisions": [],
        "semantic_decontamination": False,
        "pretraining_contamination_known": False,
        "answer_text_screened": False,
        "stops_at_first_collision": True,
    }
    widths = sorted({width for width, _ in spans})
    for role in visible:
        for sample in roles[role]:
            words = normalize(sample.instruction).split()
            for width in widths:
                for offset in range(len(words) - width + 1):
                    result["checks"] += 1
                    match = spans.get((width, tuple(words[offset : offset + width])))
                    if match:
                        result["collisions"] = [
                            {
                                "visible_role": role,
                                "visible_id": identity(sample.instruction),
                                "hidden_role": match[0],
                                "hidden_id": match[1],
                                "matched_words": width,
                            }
                        ]
                        return result
    return result
