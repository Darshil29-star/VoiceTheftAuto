"""Agent Controller / Model Router.

INPUT → VALIDATION → CLASSIFICATION → ROUTING → INFERENCE → EVIDENCE → RESPONSE
Every stage writes to the execution trace; failures return a structured error with the trace so far."""

from __future__ import annotations

import asyncio
import gc
import logging
import threading
import time
import uuid
from collections.abc import Callable

from app.controllers.task_classifier import classify
from app.core.config import Settings, get_settings
from app.core.device import resolve_device
from app.core.errors import (
    GeoSenseError,
    InferenceFailed,
    InferenceTimeout,
    MissingImage,
    ModelUnavailable,
    OutOfMemory,
    UnsupportedQuery,
)
from app.core.storage import StoredFile
from app.core.trace import Trace
from app.evidence.aggregator import aggregate, summarize
from app.evidence.confidence import banner
from app.evidence.response import compose_answer
from app.models.manager import ModelManager, get_model_manager
from app.models.registry import ModelRegistry, get_registry
from app.schemas.api import AnalyzeResponse, ClassificationInfo, Clarification, ErrorBody
from app.schemas.common import ImageRole, TaskType, TraceStatus
from app.specialists import get_specialist
from app.specialists.base import AnalysisContext, Specialist, SpecialistOutput
from app.validators.image_validator import ValidatedImage, inspect_file, validate_for_task

log = logging.getLogger("geosense.controller")

# CPU-first: heavy inference runs one request at a time to keep RAM and latency predictable.
_INFERENCE_LOCK = threading.Lock()


class AgentController:
    def __init__(self, settings: Settings | None = None, registry: ModelRegistry | None = None,
                 manager: ModelManager | None = None,
                 specialist_lookup: Callable[[TaskType], Specialist | None] = get_specialist):
        self.settings = settings or get_settings()
        self.registry = registry or get_registry()
        self.manager = manager or get_model_manager()
        self.lookup = specialist_lookup

    async def analyze(self, query: str, stored: list[StoredFile], task_override: TaskType | None = None,
                      request_id: str | None = None, input_error: GeoSenseError | None = None) -> tuple[int, AnalyzeResponse]:
        request_id = request_id or uuid.uuid4().hex[:12]
        t_start = time.perf_counter()
        trace = Trace()
        device = resolve_device(self.settings.device, self.settings.torch_threads)
        resp = AnalyzeResponse(request_id=request_id, status="ok", query=query, device=device.resolved)
        http_status = 200

        try:
            # ---------------------------------------------------------------- INPUT
            query = (query or "").strip()
            if input_error is not None:
                raise input_error
            if not stored:
                raise MissingImage("Upload at least one image to analyse.")
            if not query:
                raise UnsupportedQuery("Please enter a question or instruction.")
            if len(query) > 1000:
                raise UnsupportedQuery("Query is too long (max 1000 characters).")
            trace.add("INPUT", f"Query received; {len(stored)} image(s) received", request_id=request_id,
                      device=device.resolved, device_reason=device.reason)

            # ---------------------------------------------------------------- VALIDATION
            images: list[ValidatedImage] = []
            with trace.step("VALIDATION", "Inspecting images") as st:
                for i, s in enumerate(stored):
                    role = ImageRole(s.role) if s.role else None
                    img = inspect_file(s.path, i, display_name=s.original_name, user_role=role, settings=self.settings)
                    images.append(img)
                    info = img.info
                    geo = f", {info.crs}, GSD {info.gsd_m:g} m" if info.is_georeferenced else ", not georeferenced"
                    trace.add("VALIDATION", f"{info.filename}: {info.format} {info.width}x{info.height}, {info.bands} band(s), "
                              f"{info.dtype}{geo}; role={info.role.value} ({info.role_source})")
                resp.images = [i.info for i in images]
                st.message = f"{len(images)} image(s) valid"

            # ---------------------------------------------------------------- CLASSIFICATION
            with trace.step("CLASSIFICATION", "Classifying task") as st:
                if task_override:
                    task = task_override
                    resp.classification = ClassificationInfo(task=task, scores={task.value: 1.0},
                                                             rationale=["task selected explicitly by the client"],
                                                             decided_by="override")
                else:
                    c = classify(query, [i.info.role for i in images], self.settings.classifier_clarify_margin)
                    resp.classification = ClassificationInfo(task=c.task, scores=c.scores, rationale=c.rationale,
                                                             decided_by=c.decided_by, ambiguous=c.ambiguous)
                    if c.task is None:
                        st.status = TraceStatus.WARNING
                        st.message = "Query is ambiguous — clarification requested"
                        resp.status = "clarification_needed"
                        resp.clarification = Clarification(question=c.clarification or "", options=c.options)
                        resp.answer = c.clarification
                        return self._finish(resp, trace, t_start, http_status)
                    task = c.task
                resp.task = task
                st.message = f"Task: {task.value.upper()}" + (" (ambiguous; decision recorded)" if resp.classification.ambiguous else "")
                st.details["scores"] = resp.classification.scores

            with trace.step("VALIDATION", f"Checking {task.value} requirements") as st:
                ordered, compat, task_warnings = validate_for_task(task, images)
                resp.images = [i.info for i in images]
                resp.warnings += [f"{i.info.filename}: {w}" for i in ordered for w in i.info.warnings] + task_warnings
                if compat:
                    st.details["compatibility"] = compat
                st.message = f"Inputs satisfy {task.value} requirements" + (f" ({len(task_warnings)} warning(s))" if task_warnings else "")
                if task_warnings:
                    st.status = TraceStatus.WARNING

            # ---------------------------------------------------------------- ROUTING
            with trace.step("ROUTING", "Selecting specialist") as st:
                specialist = self.lookup(task)
                if specialist is None:
                    raise ModelUnavailable(f"The {task.value} specialist is not available in this build.")
                route = self.registry.route(task.value)
                st.message = f"{type(specialist).__name__} selected"
                st.details["registry_route"] = route

            ctx = AnalysisContext(
                request_id=request_id, query=query, task=task, images=ordered, settings=self.settings,
                registry=self.registry, manager=self.manager, trace=trace,
                output_dir=self.settings.processed_dir / request_id, url_prefix=f"/files/{request_id}",
                compatibility=compat, warnings=list(resp.warnings),
            )

            # ---------------------------------------------------------------- INFERENCE
            output = await self._run_specialist(specialist, ctx, trace)

            # ---------------------------------------------------------------- EVIDENCE
            with trace.step("EVIDENCE", "Aggregating evidence") as st:
                evidence, notes = aggregate(output.evidence, ordered)
                counts = summarize(evidence)
                st.message = (", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items()) or "no evidence items") + " collected"
                st.details["counts"] = counts
                if notes:
                    st.status = TraceStatus.WARNING
                    resp.warnings += notes

            # ---------------------------------------------------------------- RESPONSE
            with trace.step("RESPONSE", "Generating evidence-grounded response") as st:
                resp.answer = compose_answer(output.answer, output.confidence, output.fallback_used)
                resp.confidence = output.confidence
                resp.confidence_note = banner(output.confidence)
                resp.evidence = evidence
                resp.visualizations = output.visualizations
                resp.model = output.model
                resp.models_used = output.models_used or [output.model]
                resp.result = output.result
                resp.warnings += [c for c in output.caveats if c not in resp.warnings]
                resp.demo_fallback = output.fallback_used
                if output.fallback_used:
                    st.status = TraceStatus.FALLBACK
                    st.message = "Demo fallback response generated (not model inference)"
                else:
                    st.message = "Evidence-grounded response generated"
                st.details["confidence"] = output.confidence.model_dump(include={"value", "kind", "level"})

        except GeoSenseError as exc:
            http_status = exc.status_code if exc.status_code >= 400 else 400
            resp.status = "error"
            resp.error = ErrorBody(code=exc.code, message=exc.message, details=_safe_details(exc.details))
            resp.answer = exc.message
            if not trace.steps or trace.steps[-1].status != TraceStatus.ERROR:
                trace.add("ERROR", exc.message, TraceStatus.ERROR, code=exc.code)
        except Exception:  # noqa: BLE001 — never leak stack traces to clients
            log.exception("analyze_unhandled", extra={"request_id": request_id})
            http_status = 500
            resp.status = "error"
            resp.error = ErrorBody(code="internal_error", message="An internal error occurred during analysis.")
            resp.answer = resp.error.message
            trace.add("ERROR", resp.error.message, TraceStatus.ERROR)

        return self._finish(resp, trace, t_start, http_status)

    # ------------------------------------------------------------------ helpers
    async def _run_specialist(self, specialist: Specialist, ctx: AnalysisContext, trace: Trace) -> SpecialistOutput:
        def _call(fn):
            with _INFERENCE_LOCK:
                try:
                    return fn(ctx)
                except MemoryError as exc:
                    gc.collect()
                    raise OutOfMemory() from exc
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() or "not enough memory" in str(exc).lower():
                        gc.collect()
                        raise OutOfMemory() from exc
                    raise

        step_index = len(trace.steps)
        trace.add("INFERENCE", f"Running {type(specialist).__name__}")
        t0 = time.perf_counter()
        try:
            output = await asyncio.wait_for(asyncio.to_thread(_call, specialist.run), self.settings.inference_timeout_s)
            trace.steps[step_index].message = f"{type(specialist).__name__} completed"
        except asyncio.TimeoutError as exc:
            trace.steps[step_index].status = TraceStatus.ERROR
            raise InferenceTimeout(
                f"Analysis exceeded {self.settings.inference_timeout_s:.0f} s. Try a smaller image."
            ) from exc
        except ModelUnavailable as exc:
            trace.steps[step_index].status = TraceStatus.ERROR
            trace.steps[step_index].details["error"] = exc.message
            if not self.settings.demo_mode:
                raise
            fb = await asyncio.to_thread(_call, specialist.fallback)
            if fb is None:
                raise
            fb.fallback_used = True
            trace.add("INFERENCE", f"Demo fallback: {exc.message}", TraceStatus.FALLBACK)
            output = fb
        except GeoSenseError:
            trace.steps[step_index].status = TraceStatus.ERROR
            raise
        except Exception as exc:  # noqa: BLE001
            trace.steps[step_index].status = TraceStatus.ERROR
            log.exception("inference_failed", extra={"request_id": ctx.request_id, "task": ctx.task.value})
            raise InferenceFailed(f"The {ctx.task.value} specialist failed ({type(exc).__name__}).") from exc
        finally:
            trace.steps[step_index].duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        return output

    def _finish(self, resp: AnalyzeResponse, trace: Trace, t_start: float, http_status: int) -> tuple[int, AnalyzeResponse]:
        resp.processing_time = round(time.perf_counter() - t_start, 3)
        resp.execution_trace = trace.steps
        log.info("analyze_done", extra={
            "request_id": resp.request_id, "status": resp.status, "task": resp.task.value if resp.task else None,
            "model": resp.model.id if resp.model else None, "device": resp.device,
            "images": [f"{i.width}x{i.height}" for i in resp.images],
            "processing_s": resp.processing_time,
            "confidence": resp.confidence.value if resp.confidence else None,
            "confidence_level": resp.confidence.level.value if resp.confidence else None,
            "error": resp.error.code if resp.error else None,
            "demo_fallback": resp.demo_fallback,
        })
        return http_status, resp


def _safe_details(details: dict) -> dict:
    """Only JSON-simple values reach the client."""
    import json

    try:
        json.dumps(details)
        return details
    except TypeError:
        return {k: str(v) for k, v in details.items()}
