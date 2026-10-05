"""Test configuration. Isolates runtime directories and forces CPU so tests are deterministic."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="geosense-test-"))
os.environ.setdefault("ENVIRONMENT", "test")
os.environ["DEVICE"] = "cpu"
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")
os.environ["PROCESSED_DIR"] = str(_TMP / "processed")
os.environ.setdefault("DEMO_MODE", "false")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


def _write_png(path: Path, arr: np.ndarray) -> Path:
    import cv2

    cv2.imwrite(str(path), arr)
    return path


@pytest.fixture
def rgb_png(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    arr = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    return _write_png(tmp_path / "scene.png", arr)


@pytest.fixture
def geotiff_rgb(tmp_path: Path) -> Path:
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "optical.tif"
    data = (np.random.default_rng(1).random((3, 128, 128)) * 3000).astype(np.uint16)
    with rasterio.open(
        path, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint16",
        crs="EPSG:32646", transform=from_origin(500000, 2900000, 10, 10),
    ) as dst:
        dst.write(data)
        dst.update_tags(ACQUISITION_DATE="2024-03-01T04:40:00Z")
    return path


@pytest.fixture
def geotiff_sar(tmp_path: Path) -> Path:
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "s1_vv_vh.tif"
    data = (np.random.default_rng(2).random((2, 128, 128)) * 0.3).astype(np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", width=128, height=128, count=2, dtype="float32",
        crs="EPSG:32646", transform=from_origin(500000, 2900000, 10, 10),
    ) as dst:
        dst.write(data)
        dst.set_band_description(1, "VV")
        dst.set_band_description(2, "VH")
    return path
