"""Evidence aggregation: normalises specialist evidence into a consistent, verifiable form.

- stable ids
- pixel geometry clamped to the ORIGINAL image bounds; invalid geometry is dropped (and reported)
- area in pixels, plus map coordinates / m² when the source image is georeferenced
"""

from __future__ import annotations

import numpy as np

from app.geospatial.raster import pixel_to_map
from app.schemas.common import Evidence, EvidenceType, GeoRef
from app.validators.image_validator import ValidatedImage

_SPATIAL = {EvidenceType.SPATIAL_REGION, EvidenceType.MASK, EvidenceType.CHANGE_MAP, EvidenceType.IMAGE_CROP}


def _polygon_area(poly: list[list[float]]) -> float:
    if len(poly) < 3:
        return 0.0
    xs = np.array([p[0] for p in poly])
    ys = np.array([p[1] for p in poly])
    return float(0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))


def aggregate(evidence: list[Evidence], images: list[ValidatedImage]) -> tuple[list[Evidence], list[str]]:
    out: list[Evidence] = []
    notes: list[str] = []
    dropped = 0
    for n, ev in enumerate(evidence, 1):
        ev = ev.model_copy(deep=True)
        ev.id = ev.id or f"ev-{n:03d}"
        img = images[ev.image_index] if ev.image_index is not None and 0 <= ev.image_index < len(images) else None

        if ev.type in _SPATIAL and img is not None:
            W, H = img.info.width, img.info.height
            if ev.bbox is not None:
                x0, y0, x1, y1 = ev.bbox
                x0, x1 = sorted((min(max(x0, 0), W), min(max(x1, 0), W)))
                y0, y1 = sorted((min(max(y0, 0), H), min(max(y1, 0), H)))
                if x1 - x0 < 1 or y1 - y0 < 1:
                    dropped += 1
                    continue
                ev.bbox = [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]
            if ev.polygon is not None:
                ev.polygon = [[round(min(max(x, 0), W), 1), round(min(max(y, 0), H), 1)] for x, y in ev.polygon]
                if len(ev.polygon) < 3:
                    ev.polygon = None
            if ev.area_px is None:
                if ev.polygon:
                    ev.area_px = round(_polygon_area(ev.polygon), 1)
                elif ev.bbox:
                    ev.area_px = round((ev.bbox[2] - ev.bbox[0]) * (ev.bbox[3] - ev.bbox[1]), 1)

            meta = img.meta
            if meta.is_georeferenced and ev.geo is None and (ev.bbox or ev.polygon):
                geo = GeoRef(crs=meta.crs)
                if ev.bbox:
                    xs, ys = pixel_to_map(meta, [ev.bbox[0], ev.bbox[2]], [ev.bbox[1], ev.bbox[3]])
                    geo.bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
                if ev.polygon:
                    xs, ys = pixel_to_map(meta, [p[0] for p in ev.polygon], [p[1] for p in ev.polygon])
                    geo.polygon = [[round(float(x), 3), round(float(y), 3)] for x, y in zip(xs, ys)]
                if ev.area_px and meta.gsd_m:
                    geo.area_m2 = round(ev.area_px * meta.gsd_m ** 2, 1)
                ev.geo = geo
        out.append(ev)
    if dropped:
        notes.append(f"{dropped} evidence item(s) with degenerate geometry were discarded.")
    return out, notes


def summarize(evidence: list[Evidence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ev in evidence:
        counts[ev.type.value] = counts.get(ev.type.value, 0) + 1
    return counts
