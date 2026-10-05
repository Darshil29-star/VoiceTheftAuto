from __future__ import annotations

import pytest

from app.core.device import resolve_device
from app.core.errors import ModelUnavailable
from app.models.registry import ModelRegistry, get_registry


def test_device_cpu_requested_resolves_to_cpu():
    info = resolve_device("cpu")
    assert info.resolved == "cpu"


def test_device_auto_never_requires_cuda():
    info = resolve_device("auto")
    assert info.resolved in {"cpu", "cuda"}
    if not info.cuda_device_name:
        assert info.resolved == "cpu"


def test_registry_routes_every_task():
    reg = get_registry()
    for task in ("captioning", "grounding", "vqa", "change_analysis", "optical_sar_fusion", "task_classification"):
        assert reg.model_for(task) is not None, task


def test_registry_rejects_dangling_reference(tmp_path):
    bad = tmp_path / "registry.yaml"
    bad.write_text(
        "models:\n  a: {name: A, version: '1', backend: algorithm, kind: algorithm}\n"
        "tasks:\n  captioning: {primary: does-not-exist}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        ModelRegistry.from_yaml(bad)


def test_unknown_model_is_typed_error():
    with pytest.raises(ModelUnavailable):
        get_registry().get("nope")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["device"] == "cpu"
    assert "demo_mode" in body


def test_models_endpoint_lists_registry(client):
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    ids = {m["id"] for m in body["models"]}
    assert {"florence2-base", "grounding-dino-tiny", "change-classical-v1", "fusion-rules-v1"} <= ids
    assert body["task_routing"]["grounding"]["primary"] == "grounding-dino-tiny"
