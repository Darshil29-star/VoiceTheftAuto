"""Controller pipeline tests with stub specialists (no model weights needed)."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.controllers.agent_controller import AgentController
from app.core.config import get_settings
from app.core.errors import ModelUnavailable
from app.core.storage import StoredFile
from app.evidence import confidence as conf
from app.schemas.common import Evidence, EvidenceType, ModelRef, TaskType
from app.specialists.base import Specialist, SpecialistOutput

REF = ModelRef(id="stub", name="Stub", version="0", backend="test")


class StubGrounding(Specialist):
    task = TaskType.GROUNDING

    def run(self, ctx):
        return SpecialistOutput(
            answer="Found 2 regions.",
            confidence=conf.aggregate_model_scores([0.9, 0.2], "stub", "detections"),
            model=REF,
            evidence=[
                Evidence(id="", type=EvidenceType.SPATIAL_REGION, label="building", source="stub", image_index=0,
                         bbox=[-10, 5, 50, 60], confidence=conf.from_model_score(0.9, "stub", "score")),
                Evidence(id="", type=EvidenceType.SPATIAL_REGION, label="building", source="stub", image_index=0,
                         bbox=[300, 300, 400, 400]),  # fully outside a 128px image -> dropped
            ],
        )


class Unavailable(Specialist):
    task = TaskType.CAPTIONING

    def run(self, ctx):
        raise ModelUnavailable("weights missing")

    def fallback(self, ctx):
        return SpecialistOutput(answer="Image statistics only.", confidence=conf.unavailable("stats"), model=REF)


class Slow(Specialist):
    task = TaskType.CAPTIONING

    def run(self, ctx):
        time.sleep(2)
        raise AssertionError("should have timed out")


class Crashes(Specialist):
    task = TaskType.CAPTIONING

    def run(self, ctx):
        raise ZeroDivisionError("secret internal detail")


def _controller(mapping, **overrides):
    settings = get_settings().model_copy(update=overrides)
    return AgentController(settings=settings, specialist_lookup=lambda t: mapping.get(t))


def _run(controller, query, files, **kw):
    return asyncio.run(controller.analyze(query, files, **kw))


def test_grounding_pipeline_with_georeferenced_evidence(geotiff_rgb):
    status, r = _run(_controller({TaskType.GROUNDING: StubGrounding()}), "Locate the buildings", [StoredFile(geotiff_rgb, "optical.tif", None)])
    assert status == 200 and r.status == "ok"
    assert r.task == TaskType.GROUNDING
    assert len(r.evidence) == 1, "out-of-bounds box must be discarded"
    ev = r.evidence[0]
    assert ev.bbox == [0, 5, 50, 60]  # clamped
    assert ev.geo is not None and ev.geo.crs == "EPSG:32646" and ev.geo.area_m2 == pytest.approx(50 * 55 * 100)
    stages = [s.stage for s in r.execution_trace]
    for stage in ("INPUT", "VALIDATION", "CLASSIFICATION", "ROUTING", "INFERENCE", "EVIDENCE", "RESPONSE"):
        assert stage in stages
    assert r.processing_time > 0 and r.device == "cpu"
    assert r.confidence.kind.value == "model"


def test_missing_image_returns_structured_error():
    status, r = _run(_controller({}), "Describe this scene", [])
    assert status == 422 and r.status == "error" and r.error.code == "missing_image"
    assert r.execution_trace


def test_clarification_for_ambiguous_query(rgb_png):
    status, r = _run(_controller({}), "hello there", [StoredFile(rgb_png, "scene.png", None)])
    assert status == 200 and r.status == "clarification_needed"
    assert r.clarification and r.clarification.options


def test_change_with_one_image_explains_requirement(rgb_png):
    status, r = _run(_controller({}), "What changed between these images?", [StoredFile(rgb_png, "a.png", None)])
    assert r.status == "error" and r.error.code == "insufficient_images"
    assert r.task == TaskType.CHANGE_ANALYSIS


def test_model_unavailable_without_demo_mode_is_error(rgb_png):
    status, r = _run(_controller({TaskType.CAPTIONING: Unavailable()}, demo_mode=False), "Describe this scene", [StoredFile(rgb_png, "a.png", None)])
    assert status == 503 and r.error.code == "model_unavailable"
    assert not r.demo_fallback


def test_demo_fallback_is_clearly_labelled(rgb_png):
    status, r = _run(_controller({TaskType.CAPTIONING: Unavailable()}, demo_mode=True), "Describe this scene", [StoredFile(rgb_png, "a.png", None)])
    assert status == 200 and r.demo_fallback
    assert any(s.status.value == "fallback" and "Demo fallback" in s.message for s in r.execution_trace)
    assert r.confidence_note == "Confidence unavailable from model"


def test_timeout(rgb_png):
    status, r = _run(_controller({TaskType.CAPTIONING: Slow()}, inference_timeout_s=0.3), "Describe this scene", [StoredFile(rgb_png, "a.png", None)])
    assert status == 504 and r.error.code == "timeout"


def test_internal_errors_do_not_leak(rgb_png):
    status, r = _run(_controller({TaskType.CAPTIONING: Crashes()}), "Describe this scene", [StoredFile(rgb_png, "a.png", None)])
    assert status == 500 and r.error.code == "inference_failed"
    assert "secret internal detail" not in r.model_dump_json()


def test_low_confidence_is_flagged():
    c = conf.from_model_score(0.2, "m", "score")
    assert conf.banner(c) == "LOW CONFIDENCE"
    capped = conf.with_input_factors(conf.from_model_score(0.95, "m", "score"), [("SAR input to optical model", conf.ConfidenceLevel.LOW)])
    assert capped.level.value == "low" and capped.value == 0.95 and capped.kind.value == "system"


def test_analyze_api_rejects_bad_upload_with_trace(client):
    r = client.post("/analyze", data={"query": "Describe this scene"}, files=[("files", ("notes.txt", b"hi", "text/plain"))])
    assert r.status_code == 415
    body = r.json()
    assert body["error"]["code"] == "unsupported_format" and body["execution_trace"]


def test_analyze_api_missing_query(client, rgb_png):
    with rgb_png.open("rb") as f:
        r = client.post("/analyze", data={"query": "   "}, files=[("files", ("a.png", f, "image/png"))])
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_query"
