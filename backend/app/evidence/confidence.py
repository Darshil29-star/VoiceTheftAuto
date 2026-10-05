"""Transparent confidence handling.

- model:     a score emitted by the model (reported as-is; uncalibrated unless stated)
- estimated: a documented heuristic over outputs / statistics
- system:    model or estimated confidence, downgraded by input-quality factors
- unavailable: the model gives no usable score — we say so instead of inventing one
"""

from __future__ import annotations

from collections.abc import Sequence

from app.schemas.common import Confidence, ConfidenceKind, ConfidenceLevel

HIGH, MEDIUM = 0.70, 0.40
_ORDER = [ConfidenceLevel.UNAVAILABLE, ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]


def level_for(value: float | None) -> ConfidenceLevel:
    if value is None:
        return ConfidenceLevel.UNAVAILABLE
    if value >= HIGH:
        return ConfidenceLevel.HIGH
    if value >= MEDIUM:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def unavailable(source: str, reason: str = "Confidence unavailable from model") -> Confidence:
    return Confidence(value=None, kind=ConfidenceKind.UNAVAILABLE, level=ConfidenceLevel.UNAVAILABLE,
                      source=source, explanation=reason)


def from_model_score(value: float, source: str, explanation: str, calibrated: bool = False) -> Confidence:
    v = float(min(max(value, 0.0), 1.0))
    return Confidence(value=round(v, 4), kind=ConfidenceKind.MODEL, level=level_for(v), calibrated=calibrated,
                      source=source, explanation=explanation)


def aggregate_model_scores(scores: Sequence[float], source: str, what: str) -> Confidence:
    """Mean of per-item model scores (e.g. detection scores)."""
    if not scores:
        return unavailable(source, f"No {what} were produced, so there is no model score to report.")
    mean = sum(scores) / len(scores)
    return from_model_score(
        mean, source,
        f"Mean model score over {len(scores)} {what} (uncalibrated; not a probability of correctness).",
    )


def estimated(value: float, source: str, explanation: str) -> Confidence:
    v = float(min(max(value, 0.0), 1.0))
    return Confidence(value=round(v, 4), kind=ConfidenceKind.ESTIMATED, level=level_for(v),
                      source=source, explanation=explanation)


def with_input_factors(conf: Confidence, factors: list[tuple[str, ConfidenceLevel]]) -> Confidence:
    """Apply input-quality caps. Factors can only LOWER the level; the numeric value is kept as reported
    so users can still see the original score. Result kind becomes 'system' when a cap applied."""
    if not factors:
        return conf
    capped = conf.level
    reasons = []
    for reason, cap in factors:
        reasons.append(reason)
        if conf.level != ConfidenceLevel.UNAVAILABLE and _ORDER.index(cap) < _ORDER.index(capped):
            capped = cap
    out = conf.model_copy(deep=True)
    out.factors = list(conf.factors) + reasons
    if capped != conf.level:
        out.level = capped
        out.kind = ConfidenceKind.SYSTEM
        out.explanation = f"{conf.explanation} Level lowered to {capped.value.upper()} because: " + "; ".join(reasons) + "."
    return out


def banner(conf: Confidence | None) -> str | None:
    if conf is None or conf.level == ConfidenceLevel.UNAVAILABLE:
        return "Confidence unavailable from model"
    if conf.level == ConfidenceLevel.LOW:
        return "LOW CONFIDENCE"
    return None
