from __future__ import annotations

import pytest

from app.controllers.task_classifier import classify
from app.schemas.common import ImageRole as R
from app.schemas.common import TaskType as T

ONE = [R.OPTICAL]
PAIR = [R.OPTICAL, R.OPTICAL]
OPT_SAR = [R.OPTICAL, R.SAR]


@pytest.mark.parametrize(
    "query, roles, expected",
    [
        ("What is visible in this image?", ONE, T.VQA),
        ("How many buildings are there?", ONE, T.VQA),
        ("Is there any water in this image?", ONE, T.VQA),
        ("Describe this scene.", ONE, T.CAPTIONING),
        ("Give me a caption for this image", ONE, T.CAPTIONING),
        ("Locate buildings.", ONE, T.GROUNDING),
        ("Locate all buildings.", ONE, T.GROUNDING),
        ("Where are the water bodies?", ONE, T.GROUNDING),
        ("Show areas of dense vegetation.", ONE, T.GROUNDING),
        ("Find urban areas.", ONE, T.GROUNDING),
        ("What changed between these images?", PAIR, T.CHANGE_ANALYSIS),
        ("Show me what changed", ONE, T.CHANGE_ANALYSIS),  # validation will ask for a 2nd image
        ("Describe the differences between the two images", PAIR, T.CHANGE_ANALYSIS),
        ("What can be identified by combining optical and SAR?", OPT_SAR, T.OPTICAL_SAR_FUSION),
        ("What can be identified by combining these images?", OPT_SAR, T.OPTICAL_SAR_FUSION),
    ],
)
def test_routing(query, roles, expected):
    c = classify(query, roles)
    assert c.task == expected, (c.scores, c.rationale)
    assert c.rationale


def test_unintelligible_query_requests_clarification():
    c = classify("hello", ONE)
    assert c.task is None
    assert c.clarification
    assert c.decided_by == "clarification"


def test_two_images_without_keywords_uses_inputs():
    c = classify("Analyze these.", PAIR)
    assert c.task == T.CHANGE_ANALYSIS
    assert c.decided_by == "query+inputs"


def test_scores_are_normalised():
    c = classify("Locate the buildings", ONE)
    assert abs(sum(c.scores.values()) - 1.0) < 0.01
