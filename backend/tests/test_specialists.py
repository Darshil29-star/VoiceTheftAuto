"""End-to-end tests for all 5 specialist modules and the /analyze API endpoint."""

from __future__ import annotations

import asyncio
import numpy as np
import pytest

from app.controllers.agent_controller import AgentController
from app.core.config import get_settings
from app.core.storage import StoredFile
from app.schemas.common import ImageRole, TaskType
from app.specialists import get_specialist


def _controller(**overrides):
    settings = get_settings().model_copy(update={"demo_mode": True, **overrides})
    return AgentController(settings=settings)


def test_captioning_specialist_demo_mode(rgb_png):
    controller = _controller()
    files = [StoredFile(rgb_png, "optical.png", "optical")]
    status, resp = asyncio.run(controller.analyze("Describe this scene.", files, task_override=TaskType.CAPTIONING))
    
    assert status == 200
    assert resp.status == "ok"
    assert resp.task == TaskType.CAPTIONING
    assert len(resp.visualizations) >= 1
    assert any(e.label == "image_statistics" for e in resp.evidence)
    assert resp.execution_trace


def test_vqa_count_question(rgb_png):
    controller = _controller()
    files = [StoredFile(rgb_png, "optical.png", "optical")]
    status, resp = asyncio.run(controller.analyze("How many buildings are in this area?", files, task_override=TaskType.VQA))
    
    assert status == 200
    assert resp.status == "ok"
    assert resp.task == TaskType.VQA
    assert resp.execution_trace


def test_vqa_presence_question(rgb_png):
    controller = _controller()
    files = [StoredFile(rgb_png, "optical.png", "optical")]
    status, resp = asyncio.run(controller.analyze("Is there water present?", files, task_override=TaskType.VQA))
    
    assert status == 200
    assert resp.status == "ok"
    assert resp.task == TaskType.VQA


def test_vqa_open_ended_question_fallback(rgb_png):
    controller = _controller(demo_mode=True)
    files = [StoredFile(rgb_png, "optical.png", "optical")]
    status, resp = asyncio.run(controller.analyze("What general activity is visible here?", files, task_override=TaskType.VQA))
    
    assert status == 200
    assert resp.status == "ok"
    assert resp.task == TaskType.VQA
    assert resp.demo_fallback


def test_change_analysis_specialist(rgb_png, tmp_path):
    import cv2
    t1_path = rgb_png
    # Create modified T2 image with high difference patch
    arr = cv2.imread(str(rgb_png))
    arr[50:120, 50:120] = [255, 255, 255]
    t2_path = tmp_path / "after.png"
    cv2.imwrite(str(t2_path), arr)

    controller = _controller()
    files = [
        StoredFile(t1_path, "before.png", "before"),
        StoredFile(t2_path, "after.png", "after"),
    ]
    status, resp = asyncio.run(controller.analyze("What changed between these images?", files, task_override=TaskType.CHANGE_ANALYSIS))

    assert status == 200
    assert resp.status == "ok"
    assert resp.task == TaskType.CHANGE_ANALYSIS
    assert len(resp.visualizations) >= 2
    assert any(e.label == "change_statistics" for e in resp.evidence)
    assert resp.confidence.kind.value == "estimated"
    assert resp.confidence.value is not None


def test_optical_sar_fusion_specialist(geotiff_rgb, geotiff_sar):
    controller = _controller()
    files = [
        StoredFile(geotiff_rgb, "optical.tif", "optical"),
        StoredFile(geotiff_sar, "sar.tif", "sar"),
    ]
    status, resp = asyncio.run(controller.analyze("What can be identified by combining optical and SAR?", files, task_override=TaskType.OPTICAL_SAR_FUSION))

    assert status == 200
    assert resp.status == "ok"
    assert resp.task == TaskType.OPTICAL_SAR_FUSION
    assert len(resp.visualizations) >= 3
    assert any(e.label == "multimodal_classification" for e in resp.evidence)
    assert resp.confidence.kind.value == "estimated"
    assert resp.confidence.value is not None


def test_api_models_endpoint(client):
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body and "task_routing" in body
    assert "grounding" in body["task_routing"]
    assert "captioning" in body["task_routing"]
    assert "vqa" in body["task_routing"]
    assert "change_analysis" in body["task_routing"]
    assert "optical_sar_fusion" in body["task_routing"]


def test_api_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "device" in body
