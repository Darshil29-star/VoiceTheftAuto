from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.core.errors import IncompatibleImages, InsufficientImages, MissingImage, UnsupportedFormat
from app.geospatial.raster import read_raster, to_display_rgb
from app.schemas.common import ImageRole, TaskType
from app.validators.image_validator import inspect_file, validate_for_task

REPO = Path(__file__).resolve().parents[2]
LEVIR = REPO / "data" / "samples" / "change" / "levir_cdplus"


def _geotiff(path: Path, origin=(500000, 2900000), size=128, bands=3, dtype="uint16", ts=None, descs=None):
    data = (np.random.default_rng(3).random((bands, size, size)) * 1000).astype(dtype)
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=bands, dtype=dtype,
                       crs="EPSG:32646", transform=from_origin(*origin, 10, 10)) as dst:
        dst.write(data)
        if ts:
            dst.update_tags(ACQUISITION_DATE=ts)
        for i, d in enumerate(descs or [], 1):
            dst.set_band_description(i, d)
    return path


# --------------------------------------------------------------------------- single files
def test_valid_png(rgb_png):
    img = inspect_file(rgb_png, 0)
    assert img.info.format == "PNG"
    assert (img.info.width, img.info.height, img.info.bands) == (256, 256, 3)
    assert img.info.role == ImageRole.OPTICAL
    assert not img.info.is_georeferenced


def test_geotiff_metadata(geotiff_rgb):
    info = inspect_file(geotiff_rgb, 0).info
    assert info.format == "GTiff"
    assert info.crs == "EPSG:32646"
    assert info.gsd_m == pytest.approx(10.0)
    assert info.is_georeferenced
    assert info.timestamp.startswith("2024-03-01")


def test_sar_detected_from_band_descriptions(geotiff_sar):
    info = inspect_file(geotiff_sar, 0).info
    assert info.role == ImageRole.SAR
    assert info.role_source == "metadata"


def test_user_role_overrides_detection(geotiff_sar):
    assert inspect_file(geotiff_sar, 0, user_role=ImageRole.OPTICAL).info.role == ImageRole.OPTICAL


def test_invalid_image_content(tmp_path):
    bad = tmp_path / "fake.png"
    bad.write_text("this is not an image")
    with pytest.raises(UnsupportedFormat):
        inspect_file(bad, 0)


def test_unsupported_extension(tmp_path):
    f = tmp_path / "scene.bmp"
    f.write_bytes(b"BM....")
    with pytest.raises(UnsupportedFormat):
        inspect_file(f, 0)


def test_missing_image(tmp_path):
    with pytest.raises(MissingImage):
        inspect_file(tmp_path / "nope.tif", 0)


# --------------------------------------------------------------------------- task rules
def test_no_images_for_task():
    with pytest.raises(MissingImage):
        validate_for_task(TaskType.CAPTIONING, [])


def test_change_needs_two_images(rgb_png):
    with pytest.raises(InsufficientImages):
        validate_for_task(TaskType.CHANGE_ANALYSIS, [inspect_file(rgb_png, 0)])


def test_fusion_needs_optical_and_sar(tmp_path):
    a = inspect_file(_geotiff(tmp_path / "a.tif"), 0)
    b = inspect_file(_geotiff(tmp_path / "b.tif"), 1)
    with pytest.raises(IncompatibleImages):
        validate_for_task(TaskType.OPTICAL_SAR_FUSION, [a, b])


def test_fusion_orders_optical_then_sar(geotiff_rgb, geotiff_sar):
    sar, opt = inspect_file(geotiff_sar, 0), inspect_file(geotiff_rgb, 1)
    ordered, compat, _ = validate_for_task(TaskType.OPTICAL_SAR_FUSION, [sar, opt])
    assert [i.info.role for i in ordered] == [ImageRole.OPTICAL, ImageRole.SAR]
    assert compat["overlap_fraction"] == pytest.approx(1.0)


def test_fusion_rejects_non_overlapping_footprints(tmp_path, geotiff_rgb):
    far = _geotiff(tmp_path / "far_s1.tif", origin=(600000, 2800000), bands=2, dtype="float32", descs=["VV", "VH"])
    with pytest.raises(IncompatibleImages) as exc:
        validate_for_task(TaskType.OPTICAL_SAR_FUSION, [inspect_file(geotiff_rgb, 0), inspect_file(far, 1)])
    assert "overlap" in exc.value.message


def test_change_rejects_optical_plus_sar(geotiff_rgb, geotiff_sar):
    with pytest.raises(IncompatibleImages):
        validate_for_task(TaskType.CHANGE_ANALYSIS, [inspect_file(geotiff_rgb, 0), inspect_file(geotiff_sar, 1)])


def test_change_reorders_by_timestamp(tmp_path):
    late = inspect_file(_geotiff(tmp_path / "x.tif", ts="2024-06-01"), 0)
    early = inspect_file(_geotiff(tmp_path / "y.tif", ts="2020-06-01"), 1)
    ordered, compat, warnings = validate_for_task(TaskType.CHANGE_ANALYSIS, [late, early])
    assert ordered[0].info.filename == "y.tif"
    assert any("timestamp" in w for w in warnings)


def test_change_rejects_different_aspect_ratio_without_georef(tmp_path, rgb_png):
    wide = tmp_path / "wide.png"
    cv2.imwrite(str(wide), np.zeros((100, 300, 3), np.uint8))
    with pytest.raises(IncompatibleImages):
        validate_for_task(TaskType.CHANGE_ANALYSIS, [inspect_file(rgb_png, 0), inspect_file(wide, 1)])


# --------------------------------------------------------------------------- raster reading
def test_read_raster_downsamples_and_keeps_scale(tmp_path):
    big = tmp_path / "big.png"
    cv2.imwrite(str(big), np.full((1000, 2000, 3), 128, np.uint8))
    data = read_raster(big, max_side=500)
    assert data.width == 500 and data.height == 250
    assert data.scale == pytest.approx(4.0)
    assert to_display_rgb(data).shape == (250, 500, 3)


def test_display_rgb_from_uint16_geotiff(geotiff_rgb):
    rgb = to_display_rgb(read_raster(geotiff_rgb))
    assert rgb.dtype == np.uint8 and rgb.shape == (128, 128, 3)


@pytest.mark.skipif(not LEVIR.exists(), reason="sample data not fetched")
def test_real_levir_pair_is_compatible():
    t1 = inspect_file(LEVIR / "pair00_t1.png", 0)
    t2 = inspect_file(LEVIR / "pair00_t2.png", 1)
    ordered, compat, _ = validate_for_task(TaskType.CHANGE_ANALYSIS, [t1, t2])
    assert compat["compatible"] and compat["same_shape"]


# --------------------------------------------------------------------------- API
def test_upload_endpoint(client, geotiff_rgb, geotiff_sar):
    with geotiff_rgb.open("rb") as a, geotiff_sar.open("rb") as b:
        r = client.post("/upload", files=[("files", ("optical.tif", a, "image/tiff")), ("files", ("s1_vv_vh.tif", b, "image/tiff"))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["upload_id"]) == 32
    assert [i["role"] for i in body["images"]] == ["optical", "sar"]


def test_validate_endpoint_reports_task_error_without_500(client, rgb_png):
    with rgb_png.open("rb") as f:
        r = client.post("/validate", files=[("files", ("scene.png", f, "image/png"))], data={"task": "change_analysis"})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert body["errors"][0]["code"] == "insufficient_images"


def test_validate_endpoint_by_upload_id(client, geotiff_rgb, geotiff_sar):
    with geotiff_rgb.open("rb") as a, geotiff_sar.open("rb") as b:
        up = client.post("/upload", files=[("files", ("o.tif", a, "image/tiff")), ("files", ("s.tif", b, "image/tiff"))]).json()
    r = client.post("/validate", data={"upload_id": up["upload_id"], "task": "optical_sar_fusion"})
    assert r.json()["valid"] is True


def test_upload_rejects_unsupported_type(client):
    r = client.post("/upload", files=[("files", ("notes.txt", b"hello", "text/plain"))])
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_format"
    assert "Traceback" not in r.text
