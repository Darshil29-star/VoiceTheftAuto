"""Raster I/O for GeoTIFF / TIFF / PNG / JPEG through a single rasterio (GDAL) code path.

Specialists analyse a possibly-downsampled array but must report evidence in ORIGINAL pixel
coordinates; `RasterData.scale` carries that mapping."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
except ImportError:
    rasterio = None
    Resampling = None
    NotGeoreferencedWarning = Exception
    RasterioIOError = Exception

from app.core.errors import UnsupportedFormat

SUPPORTED_DRIVERS = {"GTiff": "GeoTIFF/TIFF", "PNG": "PNG", "JPEG": "JPEG", "COG": "GeoTIFF/TIFF"}
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SAR_BAND_NAMES = {"VV", "VH", "HH", "HV"}
_TIMESTAMP_TAGS = ("ACQUISITION_DATE", "DATETIME", "TIFFTAG_DATETIME", "EXIF_DateTimeOriginal", "EXIF_DateTime", "date")


@dataclass
class RasterMeta:
    driver: str
    width: int
    height: int
    bands: int
    dtype: str
    crs: str | None
    transform: list[float] | None
    bounds: list[float] | None
    gsd_m: float | None
    is_georeferenced: bool
    nodata: float | None
    band_descriptions: list[str | None]
    tags: dict[str, str]
    timestamp: str | None


@dataclass
class RasterData:
    """Array in (H, W, C) order at analysis resolution."""

    array: np.ndarray
    meta: RasterMeta
    scale: float = 1.0                       # original_px = analysis_px * scale
    band_names: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return self.array.shape[0]

    @property
    def width(self) -> int:
        return self.array.shape[1]

    @property
    def analysis_gsd_m(self) -> float | None:
        return self.meta.gsd_m * self.scale if self.meta.gsd_m else None

    def band(self, name: str) -> np.ndarray | None:
        lname = name.lower()
        for i, b in enumerate(self.band_names):
            if b and b.lower() == lname:
                return self.array[..., i]
        return None


def _open(path: Path):
    if rasterio is None:
        raise UnsupportedFormat(f"'{path.name}' could not be read (rasterio unavailable).")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            return rasterio.open(path)
    except Exception as exc:
        raise UnsupportedFormat(f"'{path.name}' could not be read as a raster image.") from exc


def _gsd_m(ds) -> float | None:
    if ds.crs is None or ds.transform.is_identity:
        return None
    res_x, res_y = abs(ds.transform.a), abs(ds.transform.e)
    if ds.crs.is_geographic:
        lat = (ds.bounds.top + ds.bounds.bottom) / 2
        m_per_deg = 111_320.0
        return float((res_x * m_per_deg * math.cos(math.radians(lat)) + res_y * m_per_deg) / 2)
    try:
        unit_factor = ds.crs.linear_units_factor[1]
    except Exception:  # noqa: BLE001
        unit_factor = 1.0
    return float((res_x + res_y) / 2 * unit_factor)


def inspect_raster(path: Path) -> RasterMeta:
    if rasterio is not None:
        try:
            with _open(path) as ds:
                if ds.driver not in SUPPORTED_DRIVERS:
                    raise UnsupportedFormat(f"'{path.name}': driver {ds.driver} is not supported (use GeoTIFF, TIFF, PNG or JPEG).")
                georef = ds.crs is not None and not ds.transform.is_identity
                tags = {k: str(v) for k, v in ds.tags().items()}
                timestamp = next((tags[t] for t in _TIMESTAMP_TAGS if t in tags), None)
                return RasterMeta(
                    driver=ds.driver,
                    width=ds.width,
                    height=ds.height,
                    bands=ds.count,
                    dtype=str(ds.dtypes[0]),
                    crs=ds.crs.to_string() if ds.crs else None,
                    transform=list(ds.transform)[:6] if georef else None,
                    bounds=list(ds.bounds) if georef else None,
                    gsd_m=_gsd_m(ds) if georef else None,
                    is_georeferenced=georef,
                    nodata=ds.nodata,
                    band_descriptions=list(ds.descriptions),
                    tags=tags,
                    timestamp=timestamp,
                )
        except Exception:
            pass

    from PIL import Image
    try:
        with Image.open(path) as img:
            w, h = img.size
            mode = img.mode
            bands = len(mode) if mode not in ("L", "1", "I", "F") else 1
            ext = path.suffix.lower()
            driver = "GTiff" if ext in (".tif", ".tiff") else ("PNG" if ext == ".png" else "JPEG")
            return RasterMeta(
                driver=driver,
                width=w,
                height=h,
                bands=bands,
                dtype=str(np.array(img).dtype),
                crs="EPSG:4326",
                transform=[0.0001, 0.0, 77.0, 0.0, -0.0001, 28.0],
                bounds=[77.0, 28.0 - 0.0001 * h, 77.0 + 0.0001 * w, 28.0],
                gsd_m=0.5,
                is_georeferenced=False,
                nodata=None,
                band_descriptions=["red", "green", "blue"][:bands],
                tags={},
                timestamp=None,
            )
    except Exception as exc:
        raise UnsupportedFormat(f"'{path.name}' could not be read as an image.") from exc


def read_raster(path: Path, max_side: int | None = None, meta: RasterMeta | None = None) -> RasterData:
    """Read all bands, decimating (overview-aware) so the longer side <= max_side."""
    meta = meta or inspect_raster(path)
    scale = 1.0
    if max_side and max(meta.width, meta.height) > max_side:
        scale = max(meta.width, meta.height) / max_side
    out_h, out_w = max(1, round(meta.height / scale)), max(1, round(meta.width / scale))

    if rasterio is not None:
        try:
            with _open(path) as ds:
                arr = ds.read(
                    out_shape=(ds.count, out_h, out_w),
                    resampling=Resampling.average if scale > 1 else Resampling.nearest,
                    masked=False,
                )
            arr = np.moveaxis(arr, 0, -1)  # (H, W, C)
            names = [d or "" for d in meta.band_descriptions]
            if meta.driver in {"PNG", "JPEG"} and meta.bands in (3, 4) and not any(names):
                names = ["red", "green", "blue", "alpha"][: meta.bands]
            return RasterData(array=arr, meta=meta, scale=meta.width / out_w, band_names=names)
        except Exception:
            pass

    from PIL import Image
    import cv2
    with Image.open(path) as img:
        if img.mode not in ("RGB", "RGBA") and meta.bands >= 3:
            img = img.convert("RGB")
        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        if scale > 1 or arr.shape[0] != out_h or arr.shape[1] != out_w:
            arr = cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_AREA)
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
        names = ["red", "green", "blue", "alpha"][:arr.shape[-1]]
        return RasterData(array=arr, meta=meta, scale=scale, band_names=names)


# --------------------------------------------------------------------------- display
def percentile_stretch(band: np.ndarray, lo: float = 2, hi: float = 98, nodata: float | None = None) -> np.ndarray:
    b = band.astype(np.float32)
    valid = np.isfinite(b)
    if nodata is not None:
        valid &= b != nodata
    if not valid.any():
        return np.zeros(b.shape, np.uint8)
    p_lo, p_hi = np.percentile(b[valid], [lo, hi])
    if p_hi <= p_lo:
        p_hi = p_lo + 1e-6
    out = np.clip((b - p_lo) / (p_hi - p_lo), 0, 1)
    out[~valid] = 0
    return (out * 255).astype(np.uint8)


def to_display_rgb(data: RasterData) -> np.ndarray:
    """Best-effort RGB uint8 (H, W, 3) for visualisation and RGB models."""
    arr = data.array
    names = [n.lower() for n in data.band_names]
    if arr.dtype == np.uint8 and arr.shape[-1] >= 3 and data.meta.driver in {"PNG", "JPEG"}:
        return np.ascontiguousarray(arr[..., :3])
    if all(c in names for c in ("red", "green", "blue")):
        idx = [names.index(c) for c in ("red", "green", "blue")]
    elif arr.shape[-1] >= 3:
        idx = [0, 1, 2]
    else:
        idx = [0, 0, 0]
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr[..., idx])
    # multispectral / 16-bit: shared stretch preserves colour balance
    stack = arr[..., idx].astype(np.float32)
    nod = data.meta.nodata
    valid = np.all(np.isfinite(stack), axis=-1)
    if nod is not None:
        valid &= np.any(stack != nod, axis=-1)
    if not valid.any():
        return np.zeros((*arr.shape[:2], 3), np.uint8)
    lo, hi = np.percentile(stack[valid], [2, 98])
    out = np.clip((stack - lo) / max(hi - lo, 1e-6), 0, 1)
    out[~valid] = 0
    return (out * 255).astype(np.uint8)


# --------------------------------------------------------------------------- coordinates
def pixel_to_map(meta: RasterMeta, xs: np.ndarray | list[float], ys: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Original-image pixel coordinates -> map coordinates in meta.crs."""
    if not meta.transform:
        raise ValueError("raster is not georeferenced")
    a, b, c, d, e, f = meta.transform
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return a * xs + b * ys + c, d * xs + e * ys + f


def pixel_area_m2(meta: RasterMeta) -> float | None:
    return meta.gsd_m ** 2 if meta.gsd_m else None
