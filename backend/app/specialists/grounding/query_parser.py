"""Grounding query understanding: natural language → target concepts + modifiers.

"Locate all the buildings and roads"   → [building, road]
"Show areas of dense vegetation"      → [forest]  (dense vegetation ≡ forest)
"Find the red cars near the river"    → [vehicle, water]  + attribute 'red' noted (not enforced)
"Find solar panels"                   → [solar panel]
"Locate the helipads"                 → open-vocabulary object 'helipads'
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.specialists.grounding.vocabulary import Concept, synonym_index

_FILLER = re.compile(
    r"\b(please|can you|could you|would you|locate|find|show( me)?|where (are|is)|highlight|mark|detect|segment|outline|"
    r"point out|identify|map|all|the|any|every|of|in|on|this|that|these|image|scene|picture|areas?|regions?|"
    r"parts?|visible|present|there|a|an|and|or|with|me|please|bounding boxes?|for|to|them|it|at)\b",
    re.I,
)
_SIZE = re.compile(r"\b(large|big|small|tiny|major|minor)\b", re.I)
_COLOR = re.compile(r"\b(red|blue|white|black|green|grey|gray|brown|yellow|orange)\b", re.I)


@dataclass
class GroundingTarget:
    concept: Concept | None       # None = open-vocabulary phrase
    phrase: str                   # text as found in the query
    label: str                    # label used in evidence


@dataclass
class ParsedGroundingQuery:
    targets: list[GroundingTarget]
    size_filter: str | None = None
    attributes: list[str] = field(default_factory=list)
    unparsed: str = ""


def parse(query: str) -> ParsedGroundingQuery:
    q = " " + " ".join(query.lower().replace("?", " ").replace(".", " ").replace(",", " , ").split()) + " "
    found: list[tuple[int, GroundingTarget]] = []
    taken = [False] * len(q)
    for syn, concept in synonym_index():
        for m in re.finditer(rf"(?<![a-z]){re.escape(syn)}(?![a-z])", q):
            if any(taken[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            found.append((m.start(), GroundingTarget(concept, syn, concept.name)))

    remainder = "".join(" " if taken[i] else ch for i, ch in enumerate(q))
    size = _SIZE.search(query)
    attrs = [m.group(0).lower() for m in _COLOR.finditer(query)]

    targets: list[GroundingTarget] = []
    seen: set[str] = set()
    for _, t in sorted(found, key=lambda x: x[0]):
        if t.label not in seen:
            seen.add(t.label)
            targets.append(t)

    leftover = _FILLER.sub(" ", _SIZE.sub(" ", _COLOR.sub(" ", remainder)))
    leftover = " ".join(w for w in re.split(r"[\s,]+", leftover) if len(w) > 2 and w.isalpha())
    if not targets and leftover:
        # unknown noun phrase → open-vocabulary detection with the user's own words
        targets.append(GroundingTarget(None, leftover, leftover))

    return ParsedGroundingQuery(targets=targets, size_filter=size.group(0).lower() if size else None,
                                attributes=attrs, unparsed=leftover)
