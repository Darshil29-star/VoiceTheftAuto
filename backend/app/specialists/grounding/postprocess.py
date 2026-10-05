"""Mask / box post-processing: connected regions, polygons, scores, and coordinate scaling."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Region:
    bbox: list[float]              # ORIGINAL image pixels
    polygon: list[list[float]]     # ORIGINAL image pixels (outer ring)
    area_px: float                 # ORIGINAL image pixels
    score: float | None            # mean model probability inside the region (None for algorithmic masks)


def regions_from_mask(mask: np.ndarray, to_original: float, score_map: np.ndarray | None = None,
                      min_area_px: int = 40, max_regions: int = 500, simplify_px: float = 1.5,
                      open_kernel: int = 3) -> tuple[list[Region], int]:
    """Split a binary mask (analysis resolution) into connected regions.

    Returns (regions sorted by area desc, number of regions dropped by max_regions)."""
    m = mask.astype(np.uint8)
    if open_kernel > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_kernel, open_kernel), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    order = sorted(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    order = [i for i in order if stats[i, cv2.CC_STAT_AREA] >= min_area_px]
    dropped = max(0, len(order) - max_regions)
    regions: list[Region] = []
    s = to_original
    for i in order[:max_regions]:
        x, y, w, h, area = stats[i]
        roi = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, simplify_px, True)
        poly = [[float((px + x) * s), float((py + y) * s)] for px, py in c.reshape(-1, 2)]
        if len(poly) < 3:
            poly = [[x * s, y * s], [(x + w) * s, y * s], [(x + w) * s, (y + h) * s], [x * s, (y + h) * s]]
        score = None
        if score_map is not None:
            vals = score_map[y:y + h, x:x + w][roi.astype(bool)].astype(np.float32)
            vals = vals[np.isfinite(vals)]  # NaN marks pixels that carry no model score (e.g. morphological fill)
            score = float(vals.mean()) if vals.size else None
        regions.append(Region(
            bbox=[float(x * s), float(y * s), float((x + w) * s), float((y + h) * s)],
            polygon=poly,
            area_px=float(area * s * s),
            score=score,
        ))
    return regions, dropped


def size_filter(regions: list[Region], which: str | None) -> tuple[list[Region], str | None]:
    """'large' keeps the top quartile by area, 'small' the bottom quartile. Returns a note describing the rule."""
    if not which or len(regions) < 4:
        return regions, None
    areas = np.array([r.area_px for r in regions])
    if which in ("large", "big", "major"):
        thr = float(np.percentile(areas, 75))
        return [r for r in regions if r.area_px >= thr], f"'{which}' interpreted as the largest 25% of regions (area ≥ {thr:,.0f} px)"
    if which in ("small", "tiny", "minor"):
        thr = float(np.percentile(areas, 25))
        return [r for r in regions if r.area_px <= thr], f"'{which}' interpreted as the smallest 25% of regions (area ≤ {thr:,.0f} px)"
    return regions, None


def nms(boxes: list[list[float]], scores: list[float], iou: float) -> list[int]:
    if not boxes:
        return []
    import torch
    from torchvision.ops import nms as tv_nms

    keep = tv_nms(torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32), iou)
    return [int(i) for i in keep]
