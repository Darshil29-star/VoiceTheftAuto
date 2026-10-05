"""Execution trace builder: timed, structured steps for every request."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from app.schemas.common import TraceStatus, TraceStep


class Trace:
    def __init__(self) -> None:
        self.steps: list[TraceStep] = []

    def add(self, stage: str, message: str, status: TraceStatus = TraceStatus.OK, duration_ms: float | None = None,
            **details: Any) -> TraceStep:
        step = TraceStep(stage=stage, status=status, message=message, duration_ms=duration_ms, details=details)
        self.steps.append(step)
        return step

    @contextmanager
    def step(self, stage: str, message: str, **details: Any) -> Iterator[TraceStep]:
        """Times a block. On exception the step is marked ERROR and the exception re-raised.
        The block may mutate the yielded step (message/status/details)."""
        step = TraceStep(stage=stage, message=message, details=dict(details))
        self.steps.append(step)
        t0 = time.perf_counter()
        try:
            yield step
        except Exception as exc:
            step.status = TraceStatus.ERROR
            step.details["error"] = getattr(exc, "message", None) or type(exc).__name__
            raise
        finally:
            step.duration_ms = round((time.perf_counter() - t0) * 1000, 1)
