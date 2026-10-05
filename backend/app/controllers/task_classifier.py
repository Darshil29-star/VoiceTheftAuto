"""Task classifier: weighted query patterns + input awareness (image count, modality roles).

Deterministic and explainable: every score contribution is recorded as a rationale string.
Replaceable via the registry (tasks.task_classification.primary)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.common import ImageRole, TaskType

T = TaskType

# (pattern, weight). Patterns are matched on the lower-cased query with word boundaries.
PATTERNS: dict[TaskType, list[tuple[str, float]]] = {
    T.CHANGE_ANALYSIS: [
        (r"\bchang(e|ed|es|ing)\b", 3.0), (r"\bdifferen(ce|ces|t)\b", 2.0), (r"\bbefore\b", 1.5), (r"\bafter\b", 1.5),
        (r"\bcompar(e|ed|ison|ing)\b", 2.0), (r"\bover time\b", 2.5), (r"\b(new|recent) (construction|buildings?)\b", 2.0),
        (r"\b(appeared|disappeared|demolish\w*|construct\w*|expan\w*|deforest\w*|damage\w*)\b", 1.5),
        (r"\bbetween (these|the|two) (images|scenes|dates|years)\b", 2.0), (r"\bbi-?temporal\b", 3.0),
    ],
    T.OPTICAL_SAR_FUSION: [
        (r"\bsar\b", 3.0), (r"\bradar\b", 3.0), (r"\bfus(e|ed|ion|ing)\b", 3.0), (r"\bcombin(e|ed|ing)\b", 2.0),
        (r"\bbackscatter\b", 3.0), (r"\bsentinel-?1\b", 3.0), (r"\boptical\b", 1.5), (r"\bboth (images|modalities|sensors)\b", 1.5),
        (r"\bmulti-?(modal|sensor)\b", 2.5), (r"\b(vv|vh)\b", 2.0),
    ],
    T.GROUNDING: [
        (r"\blocat(e|ing|ion of)\b", 3.0), (r"\bfind\b", 2.5), (r"\bwhere\b", 2.5), (r"\bshow( me)?\b", 2.0),
        (r"\bhighlight\b", 3.0), (r"\bmark\b", 2.0), (r"\bdetect\b", 2.5), (r"\bsegment\b", 3.0), (r"\boutline\b", 2.5),
        (r"\bpoint out\b", 2.5), (r"\bbounding box(es)?\b", 3.0), (r"\bmap (the|all)\b", 2.0), (r"\bidentify (all|the) \w+", 1.5),
    ],
    T.CAPTIONING: [
        (r"\bdescri(be|ption)\b", 3.0), (r"\bcaption\b", 3.0), (r"\bsummar(y|ize|ise)\b", 2.5), (r"\boverview\b", 2.0),
        (r"\bwhat does (this|the) (image|scene) (show|contain|depict)\b", 2.5), (r"\btell me about\b", 2.0),
        (r"\b(this|the) scene\b", 1.0),
    ],
    T.VQA: [
        (r"\bhow many\b", 2.5), (r"\bis there\b", 2.5), (r"\bare there\b", 2.5), (r"\bwhat is\b", 1.5), (r"\bwhich\b", 1.5),
        (r"\bwhat (colou?r|type|kind)\b", 2.0), (r"\bcount\b", 2.0), (r"^(does|do|is|are|can|was|were)\b", 2.0),
        (r"\?\s*$", 1.0), (r"\bwhat (is|can be) (visible|seen)\b", 2.0), (r"\bhow (much|large|big)\b", 2.0),
    ],
}


@dataclass
class Classification:
    task: TaskType | None
    scores: dict[str, float]
    rationale: list[str] = field(default_factory=list)
    ambiguous: bool = False
    decided_by: str = "query"
    clarification: str | None = None
    options: list[TaskType] = field(default_factory=list)


def classify(query: str, roles: list[ImageRole], clarify_margin: float = 0.15) -> Classification:
    q = " ".join((query or "").lower().split())
    raw: dict[TaskType, float] = {t: 0.0 for t in TaskType}
    rationale: list[str] = []

    for task, pats in PATTERNS.items():
        for pat, w in pats:
            if re.search(pat, q):
                raw[task] += w
                rationale.append(f"'{re.search(pat, q).group(0).strip()}' → {task.value} (+{w})")

    query_only = dict(raw)
    n = len(roles)
    # ---- input awareness
    if n >= 2:
        has_sar, has_opt = ImageRole.SAR in roles, any(r in (ImageRole.OPTICAL, ImageRole.UNKNOWN) for r in roles)
        if has_sar and has_opt:
            raw[T.OPTICAL_SAR_FUSION] += 2.0
            rationale.append("inputs: optical + SAR pair → optical_sar_fusion (+2.0)")
        elif ImageRole.SAR not in roles or roles.count(ImageRole.SAR) == 2:
            raw[T.CHANGE_ANALYSIS] += 2.0
            rationale.append("inputs: two same-modality images → change_analysis (+2.0)")
    # With one image, two-image tasks are NOT penalised: if the user asks "what changed?" the right
    # response is to route to change analysis and let validation explain that a second image is needed.

    total = sum(raw.values())
    scores = {t.value: round(v / total, 3) if total else 0.0 for t, v in raw.items()}
    ranked = sorted(raw.items(), key=lambda kv: kv[1], reverse=True)
    (top_task, top), (second_task, second) = ranked[0], ranked[1]
    decided_by = "query+inputs" if any(query_only[t] != raw[t] for t in TaskType) else "query"

    if top == 0:
        if q.endswith("?"):
            rationale.append("no task keywords; question form → vqa (default)")
            return Classification(T.VQA, scores, rationale, ambiguous=True, decided_by="query")
        return Classification(
            None, scores, rationale + ["no task keywords matched"], ambiguous=True, decided_by="clarification",
            clarification="I couldn't tell what analysis you want. Do you want a description, object locations, "
                          "an answer to a question, a change comparison, or optical–SAR fusion?",
            options=list(TaskType),
        )

    margin = (top - second) / total if total else 0.0
    if second > 0 and margin < clarify_margin:
        two_image = (T.CHANGE_ANALYSIS, T.OPTICAL_SAR_FUSION)
        if n >= 2 and top_task in two_image and second_task not in two_image:
            rationale.append(
                f"ambiguous between {top_task.value}/{second_task.value}; chose {top_task.value} "
                "because two images were provided and a single-image task would ignore one"
            )
            return Classification(top_task, scores, rationale, ambiguous=True, decided_by="query+inputs")
        # prefer tasks that the inputs can actually support before asking
        feasible = [t for t, v in ranked if v > 0 and _feasible(t, n)]
        if len(feasible) == 1 or (feasible and feasible[0] != top_task):
            chosen = feasible[0]
            rationale.append(f"ambiguous between {top_task.value}/{second_task.value}; chose {chosen.value} (only feasible with {n} image(s))")
            return Classification(chosen, scores, rationale, ambiguous=True, decided_by="query+inputs")
        return Classification(
            None, scores, rationale + [f"ambiguous: {top_task.value} vs {second_task.value} (margin {margin:.2f})"],
            ambiguous=True, decided_by="clarification",
            clarification=f"Your request could mean {_label(top_task)} or {_label(second_task)}. Which one?",
            options=[top_task, second_task],
        )

    if not _feasible(top_task, n):
        rationale.append(f"{top_task.value} needs 2 images but {n} provided — validation will explain what is missing")
    return Classification(top_task, scores, rationale, ambiguous=False, decided_by=decided_by)


def _feasible(task: TaskType, n_images: int) -> bool:
    return n_images >= 2 if task in (T.CHANGE_ANALYSIS, T.OPTICAL_SAR_FUSION) else n_images >= 1


def _label(task: TaskType) -> str:
    return {
        T.VQA: "answering a question about the image",
        T.CAPTIONING: "describing the scene",
        T.GROUNDING: "locating regions/objects",
        T.CHANGE_ANALYSIS: "comparing before/after images",
        T.OPTICAL_SAR_FUSION: "optical–SAR fusion",
    }[task]
