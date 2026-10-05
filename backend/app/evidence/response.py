"""Response generation. Deterministic and evidence-bound: the final answer is the specialist's
template-generated statement (built only from its outputs) plus labelled qualifiers.
No language model rewrites or embellishes the answer."""

from __future__ import annotations

from app.evidence.confidence import banner
from app.schemas.common import Confidence, ConfidenceLevel

def compose_answer(answer: str, confidence: Confidence | None, fallback_used: bool) -> str:
    text = answer.strip()
    if confidence is not None and confidence.level == ConfidenceLevel.LOW:
        text = f"LOW CONFIDENCE: {text}"
    return text


def confidence_note(confidence: Confidence | None) -> str | None:
    return banner(confidence)


def plural(n: int, word: str, plural_word: str | None = None) -> str:
    return f"{n} {word if n == 1 else (plural_word or word + 's')}"


def fmt_area(px: float | None, m2: float | None) -> str:
    if m2 is not None:
        return f"{m2 / 10_000:.2f} ha" if m2 >= 10_000 else f"{m2:,.0f} m²"
    return f"{px:,.0f} px" if px is not None else "unknown area"


def fmt_pct(fraction: float) -> str:
    return f"{fraction * 100:.1f}%"
