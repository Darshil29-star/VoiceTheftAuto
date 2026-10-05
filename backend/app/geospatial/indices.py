"""Deterministic spectral / colour indices. Every function documents its formula and assumptions."""

from __future__ import annotations

import numpy as np

from app.geospatial.raster import RasterData

EPS = 1e-6


def _f(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32)


def find_band(data: RasterData, *names: str) -> np.ndarray | None:
    for n in names:
        b = data.band(n)
        if b is not None:
            return b
    return None


def has_nir(data: RasterData) -> bool:
    return find_band(data, "nir", "b08", "b8", "nir08") is not None


def ndvi(data: RasterData) -> np.ndarray | None:
    """NDVI = (NIR − Red) / (NIR + Red). Needs a NIR band. Range [−1, 1]; > ~0.35 typically vegetation."""
    nir, red = find_band(data, "nir", "b08", "b8"), find_band(data, "red", "b04", "b4")
    if nir is None or red is None:
        return None
    nir, red = _f(nir), _f(red)
    return (nir - red) / (nir + red + EPS)


def ndwi(data: RasterData) -> np.ndarray | None:
    """McFeeters NDWI = (Green − NIR) / (Green + NIR). > 0 typically open water."""
    nir, green = find_band(data, "nir", "b08", "b8"), find_band(data, "green", "b03", "b3")
    if nir is None or green is None:
        return None
    nir, green = _f(nir), _f(green)
    return (green - nir) / (green + nir + EPS)


def excess_green(rgb: np.ndarray) -> np.ndarray:
    """ExG = 2g − r − b on chromatic coordinates (RGB only). A rough greenness proxy, NOT NDVI:
    affected by illumination, shadows and green roofs."""
    rgb = _f(rgb)
    s = rgb.sum(axis=-1) + EPS
    r, g, b = rgb[..., 0] / s, rgb[..., 1] / s, rgb[..., 2] / s
    return 2 * g - r - b
