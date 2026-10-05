"""Grounding engine: routes each parsed target to the right backend and returns structured outcomes.

Reused by the grounding specialist, VQA (presence / count / where questions), captioning (land-cover
evidence) and change analysis (semantic attribution)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.errors import ModelUnavailable
from app.geospatial import indices
from app.schemas.common import ModelRef, TraceStatus
from app.specialists.base import AnalysisContext
from app.specialists.grounding import backends
from app.specialists.grounding.postprocess import Region, regions_from_mask, size_filter
from app.specialists.grounding.query_parser import GroundingTarget
from app.specialists.grounding.vocabulary import Concept


@dataclass
class TargetOutcome:
    label: str
    concept: Concept | None
    method: str                              # landcover-segmentation | composite-landcover | spectral-index | open-vocabulary-detection | florence-phrase-grounding
    method_kind: str                         # model | algorithm | rule
    regions: list[Region]
    total_regions: int
    coverage_fraction: float | None          # fraction of image pixels
    mask: np.ndarray | None                  # analysis-resolution bool mask for overlays
    countable: bool
    models: list[ModelRef] = field(default_factory=list)
    scores_available: bool = True
    in_domain: bool = True
    notes: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def scores(self) -> list[float]:
        return [r.score for r in self.regions if r.score is not None and np.isfinite(r.score)]

    def area_weighted_score(self) -> float | None:
        pairs = [(r.score, r.area_px) for r in self.regions if r.score is not None and np.isfinite(r.score)]
        if not pairs:
            return None
        total = sum(a for _, a in pairs)
        return sum(s * a for s, a in pairs) / total if total else None


MAX_REGIONS_PER_TARGET = 200


def ground_targets(ctx: AnalysisContext, targets: list[GroundingTarget], size: str | None = None,
                   image_index: int = 0, with_masks: bool = True) -> list[TargetOutcome]:
    outcomes = []
    for t in targets:
        with ctx.trace.step("GROUNDING", f"Grounding '{t.label}'") as st:
            out = _ground_one(ctx, t, image_index, with_masks)
            if size:
                out.regions, note = size_filter(out.regions, size)
                if note:
                    out.notes.append(note)
            st.message = (f"'{t.label}' via {out.method}: {len(out.regions)} region(s)"
                          + (f", coverage {out.coverage_fraction:.1%}" if out.coverage_fraction is not None else ""))
            st.details.update(method=out.method, models=[m.id for m in out.models], **out.extra)
            if not out.in_domain or not out.scores_available:
                st.status = TraceStatus.WARNING
        outcomes.append(out)
    return outcomes


def _ground_one(ctx: AnalysisContext, t: GroundingTarget, image_index: int, with_masks: bool) -> TargetOutcome:
    c = t.concept
    data = ctx.raster(image_index)
    if c is not None and c.spectral and indices.has_nir(data):
        return _spectral(ctx, t, image_index)
    if c is not None and c.kind in ("landcover", "composite"):
        return _landcover(ctx, t, image_index)
    return _objects(ctx, t, image_index, with_masks)


# --------------------------------------------------------------------------- land cover
def _landcover(ctx: AnalysisContext, t: GroundingTarget, image_index: int) -> TargetOutcome:
    c = t.concept
    lc = backends.run_landcover(ctx, image_index)
    data = ctx.raster(image_index)
    min_px = int(lc.spec.params.get("min_region_px", 40))
    prob = lc.prob.astype(np.float32)
    extra = dict(tiles=lc.tiles, rescale=round(lc.rescale, 3), gsd_m=lc.gsd_m, gsd_source=lc.gsd_source,
                 inference_s=round(lc.seconds, 2))

    union = np.zeros(lc.labels.shape, bool)
    for cls in c.landcover_classes:
        union |= lc.mask(cls)
    coverage = float(union.mean())

    if c.kind == "composite" and c.name == "urban area":
        gsd = (lc.gsd_m or 0.5) * data.scale
        k = int(np.clip(round(20 / gsd), 3, 41)) | 1  # ~20 m closing joins buildings and streets into zones
        zone = cv2.morphologyEx(union.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)).astype(bool)
        score_map = np.where(union, prob, np.nan)
        min_zone = max(min_px, int(2500 / (gsd * gsd)))  # ≥ ~0.25 ha
        regions, dropped = regions_from_mask(zone, data.scale, score_map, min_area_px=min_zone,
                                             max_regions=MAX_REGIONS_PER_TARGET, simplify_px=2.5, open_kernel=1)
        mask, method, kind = zone, "composite-landcover", "rule"
        coverage = float(zone.mean())
    else:
        regions, dropped = regions_from_mask(union, data.scale, prob, min_area_px=min_px, max_regions=MAX_REGIONS_PER_TARGET)
        mask = union
        method, kind = ("composite-landcover", "rule") if c.kind == "composite" else ("landcover-segmentation", "model")

    notes = list(c.notes) + lc.notes
    if dropped:
        notes.append(f"Only the {MAX_REGIONS_PER_TARGET} largest of {len(regions) + dropped} regions are listed as evidence.")
    return TargetOutcome(label=c.name, concept=c, method=method, method_kind=kind, regions=regions,
                         total_regions=len(regions) + dropped, coverage_fraction=coverage, mask=mask,
                         countable=c.countable, models=[lc.spec.ref()], in_domain=lc.in_domain, notes=notes, extra=extra)


# --------------------------------------------------------------------------- spectral
def _spectral(ctx: AnalysisContext, t: GroundingTarget, image_index: int) -> TargetOutcome:
    c = t.concept
    data = ctx.raster(image_index)
    if c.spectral == "ndwi":
        idx, thr, name = indices.ndwi(data), 0.0, "NDWI"
    else:
        idx = indices.ndvi(data)
        thr, name = (0.5, "NDVI") if c.name == "forest" else (0.35, "NDVI")
    valid = np.isfinite(idx)
    if data.meta.nodata is not None:
        valid &= np.all(data.array != data.meta.nodata, axis=-1)
    mask = (idx > thr) & valid
    # estimated per-pixel certainty: distance from threshold, saturating at 0.3 index units
    certainty = np.clip(np.abs(idx - thr) / 0.3, 0, 1).astype(np.float32)
    regions, dropped = regions_from_mask(mask, data.scale, certainty, min_area_px=16, max_regions=MAX_REGIONS_PER_TARGET)
    notes = [f"Deterministic {name} > {thr:g} threshold (no ML model). Region scores are an estimated margin from "
             "the threshold, not a probability."]
    if c.name == "forest":
        notes.append("'Dense vegetation' interpreted as NDVI > 0.5.")
    return TargetOutcome(label=c.name, concept=c, method="spectral-index", method_kind="algorithm", regions=regions,
                         total_regions=len(regions) + dropped, coverage_fraction=float(mask[valid].mean()) if valid.any() else 0.0,
                         mask=mask, countable=c.countable, notes=notes,
                         extra=dict(index=name, threshold=thr, index_mean=round(float(np.nanmean(idx[valid])), 3) if valid.any() else None))


# --------------------------------------------------------------------------- discrete objects
def _objects(ctx: AnalysisContext, t: GroundingTarget, image_index: int, with_masks: bool) -> TargetOutcome:
    c = t.concept
    data = ctx.raster(image_index)
    prompt = (c.detector_prompt if c and c.detector_prompt else t.phrase)
    label = c.name if c else t.label
    notes: list[str] = [] if c else [f"'{t.phrase}' is not in the remote-sensing vocabulary; open-vocabulary detection was used as-is."]
    try:
        det = backends.run_detector(ctx, prompt, small_object=bool(c and c.small_object), image_index=image_index)
        dets, models, method, scores_ok = det.detections, [det.spec.ref()], "open-vocabulary-detection", True
        extra = dict(prompt=prompt, tiles=det.tiles, inference_s=round(det.seconds, 2), filtered_group_boxes=det.filtered_group_boxes)
        if det.filtered_group_boxes:
            notes.append(f"{det.filtered_group_boxes} oversized 'group' box(es) spanning many objects were discarded.")
    except ModelUnavailable:
        dets, spec = backends.run_florence_phrase_grounding(ctx, t.phrase, image_index)
        models, method, scores_ok = [spec.ref()], "florence-phrase-grounding", False
        extra = dict(prompt=t.phrase)
        notes.append("Primary detector unavailable; Florence-2 phrase grounding used. It gives no per-box scores and "
                     "is unreliable for many small objects.")

    regions: list[Region] = []
    mask = np.zeros((data.height, data.width), bool)
    sam = backends.refine_with_sam(ctx, [d.bbox for d in dets], image_index) if with_masks and dets else None
    s = data.scale
    for i, d in enumerate(dets):
        x0, y0, x1, y1 = d.bbox
        poly = [[x0 * s, y0 * s], [x1 * s, y0 * s], [x1 * s, y1 * s], [x0 * s, y1 * s]]
        area = (x1 - x0) * (y1 - y0) * s * s
        if sam and i < len(sam[0]):
            m = sam[0][i]
            mask |= m
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs:
                cmax = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.5, True).reshape(-1, 2)
                if len(cmax) >= 3:
                    poly = [[float(px * s), float(py * s)] for px, py in cmax]
                    area = float(m.sum() * s * s)
        else:
            mask[int(y0):int(y1), int(x0):int(x1)] = True
        regions.append(Region(bbox=[x0 * s, y0 * s, x1 * s, y1 * s], polygon=poly, area_px=area,
                              score=d.score if scores_ok else None))
    if sam:
        models.append(sam[2].ref())
        extra["mask_refiner_mean_iou"] = round(float(np.mean(sam[1])), 3) if sam[1] else None
    regions.sort(key=lambda r: (r.score or 0), reverse=True)
    return TargetOutcome(label=label, concept=c, method=method, method_kind="model", regions=regions,
                         total_regions=len(regions), coverage_fraction=float(mask.mean()), mask=mask, countable=True,
                         models=models, scores_available=scores_ok, notes=notes, extra=extra)
