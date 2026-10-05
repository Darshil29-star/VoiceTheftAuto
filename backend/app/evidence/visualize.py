"""Visualization artifacts. Base images are saved once; masks/change maps are saved as
transparent RGBA overlays so the frontend can toggle layers. Boxes and polygons are NOT burned
into images — the frontend draws them from structured evidence."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import cv2
import numpy as np

from app.schemas.common import Visualization

MAX_DISPLAY_SIDE = 1600

# canonical colours shared with the frontend legend (hex)
LABEL_COLOURS = {
    "building": "#ff3b30",
    "road": "#b0b0b0",
    "water": "#1e90ff",
    "barren": "#c8a165",
    "forest": "#2e9e44",
    "vegetation": "#34c759",
    "agriculture": "#e6d200",
    "change": "#ff2d95",
    "built-up": "#ff3b30",
    "bare": "#c8a165",
    "other": "#8e8e93",
}


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def colour_for(label: str) -> str:
    return LABEL_COLOURS.get(label.lower(), "#ffcc00")


def _fit(arr: np.ndarray, interpolation=cv2.INTER_AREA) -> np.ndarray:
    h, w = arr.shape[:2]
    s = max(h, w) / MAX_DISPLAY_SIDE
    if s <= 1:
        return arr
    return cv2.resize(arr, (round(w / s), round(h / s)), interpolation=interpolation)


def save_rgb(ctx, name: str, rgb: np.ndarray, title: str, kind: str = "base", image_index: int | None = None,
             group: str = "main", default_visible: bool = True, legend: dict[str, str] | None = None) -> Visualization:
    img = _fit(rgb)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    try:
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        path = ctx.output_dir / f"{name}.jpg"
        cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        url = f"{ctx.url_prefix}/{path.name}"
    except Exception:
        url = ""

    if os.environ.get("VERCEL"):
        _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        url = f"data:image/jpeg;base64,{base64.b64encode(enc).decode('ascii')}"

    return Visualization(id=name, kind=kind, title=title, url=url, image_index=image_index,
                         width=img.shape[1], height=img.shape[0], group=group, default_visible=default_visible, legend=legend)


def save_overlay(ctx, name: str, rgba: np.ndarray, title: str, kind: str = "overlay", image_index: int | None = None,
                 group: str = "main", default_visible: bool = True, legend: dict[str, str] | None = None) -> Visualization:
    img = _fit(rgba, interpolation=cv2.INTER_NEAREST)
    bgra = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    try:
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        path = ctx.output_dir / f"{name}.png"
        cv2.imwrite(str(path), bgra)
        url = f"{ctx.url_prefix}/{path.name}"
    except Exception:
        url = ""

    if os.environ.get("VERCEL"):
        _, enc = cv2.imencode(".png", bgra)
        url = f"data:image/png;base64,{base64.b64encode(enc).decode('ascii')}"

    return Visualization(id=name, kind=kind, title=title, url=url, image_index=image_index,
                         width=img.shape[1], height=img.shape[0], group=group, default_visible=default_visible, legend=legend)


def label_overlay(label_map: np.ndarray, colours: dict[int, str], alpha: int = 140) -> np.ndarray:
    """Integer label map -> RGBA overlay (label 0 / unmapped = transparent)."""
    h, w = label_map.shape
    out = np.zeros((h, w, 4), np.uint8)
    for lab, hx in colours.items():
        m = label_map == lab
        if m.any():
            out[m, :3] = hex_to_rgb(hx)
            out[m, 3] = alpha
    return out


def mask_overlay(mask: np.ndarray, hx: str, alpha: int = 150) -> np.ndarray:
    return label_overlay(mask.astype(np.uint8), {1: hx}, alpha)


def heatmap_rgb(values: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    v = values.astype(np.float32)
    lo = np.nanpercentile(v, 2) if vmin is None else vmin
    hi = np.nanpercentile(v, 98) if vmax is None else vmax
    norm = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    norm = np.nan_to_num(norm)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
