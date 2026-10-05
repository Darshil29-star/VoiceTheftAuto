"""Grounding specialist (flagship): query → concepts → backend → regions → coordinates → evidence →
visualisation → structured, evidence-bound response."""

from __future__ import annotations

from app.core.errors import UnsupportedQuery
from app.evidence import confidence as conf
from app.evidence.response import fmt_area, fmt_pct, plural
from app.evidence.visualize import colour_for, mask_overlay, save_overlay, save_rgb
from app.geospatial.raster import to_display_rgb
from app.schemas.common import (
    ConfidenceLevel,
    Evidence,
    EvidenceType,
    ImageRole,
    ModelRef,
    TaskType,
    Visualization,
)
from app.specialists.base import AnalysisContext, Specialist, SpecialistOutput
from app.specialists.grounding.engine import TargetOutcome, ground_targets
from app.specialists.grounding.query_parser import parse

_METHOD_LABEL = {
    "landcover-segmentation": "land-cover segmentation",
    "composite-landcover": "land-cover classes combined by a documented rule",
    "spectral-index": "a deterministic spectral index",
    "open-vocabulary-detection": "open-vocabulary object detection",
    "florence-phrase-grounding": "Florence-2 phrase grounding",
}
_PLURAL = {"building": "building regions", "urban area": "urban zones", "road": "road segments", "water": "water bodies",
           "forest": "dense-vegetation patches", "vegetation": "vegetated patches", "agriculture": "agricultural parcels",
           "barren": "barren patches", "vehicle": "vehicles", "ship": "ships", "airplane": "airplanes",
           "storage tank": "storage tanks", "swimming pool": "swimming pools", "sports field": "sports fields",
           "bridge": "bridges", "windmill": "wind turbines", "solar panel": "solar-panel areas"}


def outcome_evidence(ctx: AnalysisContext, outcomes: list[TargetOutcome], image_index: int = 0,
                     source_prefix: str = "grounding") -> tuple[list[Evidence], list[Visualization]]:
    """Standard evidence + overlays for grounding outcomes (reused by other specialists)."""
    evidence: list[Evidence] = []
    vis: list[Visualization] = []
    gsd, gsd_src = ctx.gsd(image_index)
    for oi, out in enumerate(outcomes):
        colour = colour_for(out.label)
        method = "algorithm" if out.method_kind in ("algorithm",) else ("rule" if out.method_kind == "rule" else "model")
        src = out.models[0].id if out.models else f"{source_prefix}:{out.method}"
        for ri, r in enumerate(out.regions):
            rc = None
            if r.score is not None:
                rc = (conf.estimated(r.score, src, "Margin from spectral threshold (estimated).")
                      if out.method == "spectral-index"
                      else conf.from_model_score(r.score, src, "Mean model probability inside the region (uncalibrated)."
                                                 if out.method != "open-vocabulary-detection" else "Detector box score (uncalibrated)."))
            evidence.append(Evidence(
                id=f"{out.label.replace(' ', '_')}-{ri + 1:03d}", type=EvidenceType.SPATIAL_REGION, label=out.label,
                source=src, method=method, image_index=image_index, bbox=r.bbox, polygon=r.polygon, area_px=r.area_px,
                confidence=rc, visualization_id=f"mask-{oi}-{out.label.replace(' ', '_')}",
            ))
        area_m2 = (sum(r.area_px for r in out.regions) * gsd * gsd) if gsd else None
        evidence.append(Evidence(
            id=f"{out.label.replace(' ', '_')}-summary", type=EvidenceType.STATISTIC, label=out.label, source=src,
            method=method, image_index=image_index,
            value={"method": out.method, "regions": out.total_regions, "coverage_fraction": out.coverage_fraction,
                   "estimated_area_m2": round(area_m2, 1) if area_m2 else None, "gsd_m": gsd, "gsd_source": gsd_src,
                   "mean_score": round(out.area_weighted_score(), 4) if out.area_weighted_score() is not None else None,
                   **out.extra},
            description="; ".join(out.notes) or None,
        ))
        if out.mask is not None and out.mask.any():
            vis.append(save_overlay(ctx, f"mask-{oi}-{out.label.replace(' ', '_')}", mask_overlay(out.mask, colour),
                                    f"{out.label} ({_METHOD_LABEL.get(out.method, out.method)})", kind="mask",
                                    image_index=image_index, group="grounding", legend={out.label: colour}))
    return evidence, vis


def outcome_sentence(out: TargetOutcome, gsd: float | None, gsd_src: str) -> str:
    noun = _PLURAL.get(out.label, f"'{out.label}' detections")
    n = out.total_regions
    area_px = sum(r.area_px for r in out.regions)
    area = fmt_area(area_px, area_px * gsd * gsd if gsd else None)
    area_note = f" (area estimate uses {gsd_src} GSD {gsd:g} m)" if gsd and gsd_src != "georeferencing" else ""
    if n == 0:
        return f"No {noun} were found using {_METHOD_LABEL.get(out.method, out.method)}."
    if out.method in ("open-vocabulary-detection", "florence-phrase-grounding"):
        score = (f" (mean detector score {sum(out.scores) / len(out.scores):.2f})" if out.scores else "")
        return f"Detected {plural(n, noun.rstrip('s') if n == 1 else noun, noun)}{score}."
    cov = f"{fmt_pct(out.coverage_fraction)} of the image" if out.coverage_fraction is not None else "part of the image"
    if out.countable:
        return f"Located {n} {noun} covering {cov} (≈ {area}){area_note}."
    return f"{out.label.capitalize()} covers {cov} (≈ {area}) across {n} connected area(s){area_note}."


def outcome_confidence(ctx: AnalysisContext, outcomes: list[TargetOutcome], image_index: int = 0):
    scored = [(o.area_weighted_score(), o) for o in outcomes if o.area_weighted_score() is not None]
    methods = {o.method for o in outcomes}
    if not scored:
        if any(o.total_regions for o in outcomes):
            base = conf.unavailable("grounding", "Confidence unavailable from model (the selected method gives no per-region score).")
        else:
            base = conf.unavailable("grounding", "No regions were found; the model provides no score for absence.")
    elif methods == {"spectral-index"}:
        base = conf.estimated(sum(s for s, _ in scored) / len(scored), "spectral-index",
                              "Estimated from how far pixel index values lie from the threshold.")
    else:
        v = sum(s for s, _ in scored) / len(scored)
        base = conf.from_model_score(v, ",".join(sorted({m.id for _, o in scored for m in o.models})),
                                     "Mean of area-weighted region probabilities / detector scores (uncalibrated).")
    factors: list[tuple[str, ConfidenceLevel]] = []
    if any(not o.in_domain for o in outcomes):
        factors.append(("image resolution is outside the model's training domain", ConfidenceLevel.LOW))
    info = ctx.images[image_index].info
    if info.role == ImageRole.SAR:
        factors.append(("the input is SAR but the model is trained on optical imagery", ConfidenceLevel.LOW))
    gsd, src = ctx.gsd(image_index)
    if src.startswith("assumed") and any(o.method != "open-vocabulary-detection" for o in outcomes):
        factors.append((f"ground resolution is assumed ({gsd:g} m), not known", ConfidenceLevel.MEDIUM))
    for o in outcomes:
        if o.method == "open-vocabulary-detection" and o.regions:
            smallest = min(min(r.bbox[2] - r.bbox[0], r.bbox[3] - r.bbox[1]) for r in o.regions)
            if smallest < 12:
                factors.append((f"some '{o.label}' objects are under 12 px across, near the resolution limit", ConfidenceLevel.MEDIUM))
    return conf.with_input_factors(base, factors)


class GroundingSpecialist(Specialist):
    task = TaskType.GROUNDING

    def run(self, ctx: AnalysisContext) -> SpecialistOutput:
        with ctx.trace.step("GROUNDING", "Parsing grounding query") as st:
            parsed = parse(ctx.query)
            if not parsed.targets:
                raise UnsupportedQuery(
                    "Tell me what to locate, e.g. buildings, roads, water bodies, dense vegetation, farmland, vehicles or ships."
                )
            st.message = "Targets: " + ", ".join(
                f"{t.label} ({t.concept.kind if t.concept else 'open vocabulary'})" for t in parsed.targets)
            st.details.update(size_filter=parsed.size_filter, attributes=parsed.attributes)

        data = ctx.raster(0)
        base = save_rgb(ctx, "base-0", to_display_rgb(data), ctx.images[0].info.filename, kind="base", image_index=0,
                        group="grounding")
        outcomes = ground_targets(ctx, parsed.targets, parsed.size_filter)
        evidence, overlays = outcome_evidence(ctx, outcomes)

        gsd, gsd_src = ctx.gsd(0)
        sentences = [outcome_sentence(o, gsd, gsd_src) for o in outcomes]
        caveats: list[str] = []
        for o in outcomes:
            caveats += [n for n in o.notes if n not in caveats]
        if parsed.attributes:
            caveats.append(f"Attribute(s) {parsed.attributes} were noted but not used to filter results.")

        models: list[ModelRef] = []
        for o in outcomes:
            models += [m for m in o.models if m.id not in {x.id for x in models}]
        primary = models[0] if models else ModelRef(id="grounding-rules", name="Rule-based grounding", version="1.0",
                                                    backend="algorithm", kind="algorithm")
        detections = [
            {"label": o.label, "bbox": [round(v, 1) for v in r.bbox],
             "score": round(r.score, 4) if r.score is not None else None, "method": o.method}
            for o in outcomes for r in o.regions[:100]
        ]
        result = {
            "task": "grounding",
            "query": ctx.query,
            "targets": [{
                "label": o.label, "method": o.method, "models": [m.id for m in o.models], "regions": o.total_regions,
                "coverage_fraction": o.coverage_fraction, "countable": o.countable,
                "mean_score": o.area_weighted_score(), "in_domain": o.in_domain, "notes": o.notes, **o.extra,
            } for o in outcomes],
            "detections": detections,
            "gsd_m": gsd, "gsd_source": gsd_src,
            "coordinate_space": "original image pixels" + (" + map coordinates" if ctx.images[0].info.is_georeferenced else ""),
        }
        return SpecialistOutput(
            answer=" ".join(sentences),
            confidence=outcome_confidence(ctx, outcomes),
            model=primary,
            models_used=models,
            evidence=evidence,
            visualizations=[base, *overlays],
            result=result,
            caveats=caveats,
        )

    def fallback(self, ctx: AnalysisContext) -> SpecialistOutput:
        """Deterministic, domain-aware spatial detection fallback when neural weights are offline."""
        import cv2
        import numpy as np
        from app.geospatial import indices
        from app.schemas.common import Confidence, ConfidenceKind, ConfidenceLevel, Evidence, EvidenceType, ModelRef

        parsed = parse(ctx.query)
        targets = [t.label for t in parsed.targets] if parsed.targets else ["object"]

        data = ctx.raster(0)
        rgb = to_display_rgb(data)
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1. Precise water body estimation (spectral NDWI if NIR exists, or RGB water index)
        if indices.has_nir(data):
            ndwi_map = indices.ndwi(data)
            water_mask = (ndwi_map > 0.0) & np.isfinite(ndwi_map)
        else:
            r_f = rgb[:, :, 0].astype(float)
            g_f = rgb[:, :, 1].astype(float)
            b_f = rgb[:, :, 2].astype(float)
            water_idx = (b_f - r_f) / (b_f + r_f + 1e-5)
            water_mask = (water_idx > 0.12) & (b_f > 35) & (b_f >= g_f - 12) & (r_f < 140)
            dark_water = (gray < 40) & (r_f <= b_f + 5) & (r_f <= g_f + 5)
            water_mask = water_mask | dark_water
            water_mask = cv2.morphologyEx(
                water_mask.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            ).astype(bool)

        water_pct = float(water_mask.mean())

        # Vectorized Non-Maximum Suppression (NMS)
        def _nms(boxes: list[list[float]], scores: list[float], iou_thresh: float = 0.35) -> list[int]:
            if not boxes:
                return []
            b = np.array(boxes, dtype=float)
            s = np.array(scores, dtype=float)
            x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            order = s.argsort()[::-1]
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                w_box = np.maximum(0.0, xx2 - xx1)
                h_box = np.maximum(0.0, yy2 - yy1)
                inter = w_box * h_box
                ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
                inds = np.where(ovr <= iou_thresh)[0]
                order = order[inds + 1]
            return keep

        caveats: list[str] = [
            "Inference executed using domain-aware spatial morphology and spectral filtering fallback because neural weights are offline."
        ]

        regions_by_target: dict[str, list[dict]] = {}
        counts: dict[str, int] = {}

        # 2. Specialized feature extraction for each queried concept
        for target in targets:
            tl = target.lower()
            is_marine = any(k in tl for k in ("ship", "boat", "vessel", "craft", "cargo", "carrier", "ferry", "barge", "submarine"))
            is_tank = any(k in tl for k in ("tank", "silo", "reservoir", "storage", "fuel tank", "oil tank"))
            is_water = any(k in tl for k in ("water", "river", "lake", "ocean", "sea", "pond", "canal", "stream", "bay"))
            is_building = any(k in tl for k in ("building", "house", "structure", "facility", "warehouse", "residential", "rooftop", "roof", "home"))
            is_road = any(k in tl for k in ("road", "street", "highway", "runway", "path", "lane", "motorway"))
            is_vehicle = any(k in tl for k in ("vehicle", "car", "truck", "bus", "automobile"))
            is_vegetation = any(k in tl for k in ("vegetation", "forest", "tree", "greenery", "agriculture", "farm", "field", "crop"))

            target_candidates: list[dict] = []

            # A. Marine targets (ships, boats, vessels) - strictly require navigable water!
            if is_marine:
                if water_pct < 0.015:
                    caveats.append(f"0 '{target}' detected: no navigable water bodies or maritime surface present in scene ({water_pct:.2%} water coverage).")
                    regions_by_target[target] = []
                    counts[target] = 0
                    continue

                # When water exists, detect vessels exclusively within or immediately bordering the water zone
                water_dil = cv2.dilate(water_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=2)
                marine_zone = cv2.bitwise_and(gray, gray, mask=water_dil)
                _, th_vessel = cv2.threshold(marine_zone, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cnts, _ = cv2.findContours(th_vessel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    area = cv2.contourArea(c)
                    if 60 <= area <= (h * w * 0.15):
                        bx, by, bw, bh = cv2.boundingRect(c)
                        aspect = max(bw, bh) / max(min(bw, bh), 1)
                        if 1.6 <= aspect <= 8.5:
                            # Higher score for elongated, well-defined vessel shapes
                            score = float(min(0.95, 0.65 + (aspect / 15.0)))
                            target_candidates.append({
                                "bbox": [float(bx), float(by), float(bx + bw), float(by + bh)],
                                "score": score,
                                "area": float(area),
                            })
                keep_idx = _nms([c["bbox"] for c in target_candidates], [c["score"] for c in target_candidates], iou_thresh=0.35)
                regions_by_target[target] = [target_candidates[i] for i in keep_idx]
                counts[target] = len(regions_by_target[target])

            # B. Storage tanks, silos, round fuel reservoirs - multi-scale circular morphology
            elif is_tank:
                tank_raw = []
                for ksize in [21, 35, 51, 65]:
                    for mode in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
                        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, ksize, 3)
                        opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                        cnts, _ = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                        for c in cnts:
                            area = cv2.contourArea(c)
                            if 70 <= area <= min(40000.0, h * w * 0.05):
                                perim = cv2.arcLength(c, True)
                                if perim > 0:
                                    circ = 4 * np.pi * area / (perim * perim)
                                    bx, by, bw, bh = cv2.boundingRect(c)
                                    aspect = max(bw, bh) / max(min(bw, bh), 1)
                                    if circ >= 0.48 and aspect <= 1.32:
                                        # Center displacement check
                                        M = cv2.moments(c)
                                        if M["m00"] != 0:
                                            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                                            dist_center = np.hypot(cx - (bx + bw / 2), cy - (by + bh / 2))
                                            if dist_center <= max(bw, bh) * 0.22:
                                                score = float(min(0.95, circ * (1.0 / aspect)))
                                                tank_raw.append({
                                                    "bbox": [float(bx), float(by), float(bx + bw), float(by + bh)],
                                                    "score": score,
                                                    "area": float(area),
                                                })
                keep_idx = _nms([c["bbox"] for c in tank_raw], [c["score"] for c in tank_raw], iou_thresh=0.35)
                # Keep top 150 cleanest tanks
                sorted_tanks = sorted([tank_raw[i] for i in keep_idx], key=lambda x: x["score"], reverse=True)[:150]
                regions_by_target[target] = sorted_tanks
                counts[target] = len(sorted_tanks)

            # C. Water bodies
            elif is_water:
                water_cnts, _ = cv2.findContours(water_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in water_cnts:
                    area = cv2.contourArea(c)
                    if area >= 120:
                        bx, by, bw, bh = cv2.boundingRect(c)
                        target_candidates.append({
                            "bbox": [float(bx), float(by), float(bx + bw), float(by + bh)],
                            "score": 0.85,
                            "area": float(area),
                        })
                keep_idx = _nms([c["bbox"] for c in target_candidates], [c["score"] for c in target_candidates], iou_thresh=0.40)
                regions_by_target[target] = [target_candidates[i] for i in keep_idx]
                counts[target] = len(regions_by_target[target])

            # D. Buildings / infrastructure
            elif is_building:
                # Mask out water pixels
                dry_gray = gray.copy()
                if water_pct > 0.01:
                    dry_gray[water_mask] = 0
                _, th_bld = cv2.threshold(cv2.GaussianBlur(dry_gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cnts, _ = cv2.findContours(th_bld, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    area = cv2.contourArea(c)
                    if 80 <= area <= (h * w * 0.30):
                        bx, by, bw, bh = cv2.boundingRect(c)
                        aspect = max(bw, bh) / max(min(bw, bh), 1)
                        hull = cv2.convexHull(c)
                        hull_area = cv2.contourArea(hull)
                        solidity = float(area) / max(hull_area, 1.0)
                        if 1.0 <= aspect <= 3.8 and solidity >= 0.60:
                            target_candidates.append({
                                "bbox": [float(bx), float(by), float(bx + bw), float(by + bh)],
                                "score": 0.72,
                                "area": float(area),
                            })
                keep_idx = _nms([c["bbox"] for c in target_candidates], [c["score"] for c in target_candidates], iou_thresh=0.35)
                sorted_blds = sorted([target_candidates[i] for i in keep_idx], key=lambda x: x["area"], reverse=True)[:100]
                regions_by_target[target] = sorted_blds
                counts[target] = len(sorted_blds)

            # E. Roads, highways, runways
            elif is_road:
                edges = cv2.Canny(blur, 40, 140)
                cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    area = cv2.contourArea(c)
                    bx, by, bw, bh = cv2.boundingRect(c)
                    aspect = max(bw, bh) / max(min(bw, bh), 1)
                    if aspect >= 3.5 and (bw * bh) >= 300:
                        target_candidates.append({
                            "bbox": [float(bx), float(by), float(bx + bw), float(by + bh)],
                            "score": 0.68,
                            "area": float(bw * bh),
                        })
                keep_idx = _nms([c["bbox"] for c in target_candidates], [c["score"] for c in target_candidates], iou_thresh=0.35)
                regions_by_target[target] = [target_candidates[i] for i in keep_idx][:50]
                counts[target] = len(regions_by_target[target])

            # F. Vehicles
            elif is_vehicle:
                _, th_veh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cnts, _ = cv2.findContours(th_veh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    area = cv2.contourArea(c)
                    if 20 <= area <= 600:
                        bx, by, bw, bh = cv2.boundingRect(c)
                        aspect = max(bw, bh) / max(min(bw, bh), 1)
                        if 1.2 <= aspect <= 3.5:
                            target_candidates.append({
                                "bbox": [float(bx), float(by), float(bx + bw), float(by + bh)],
                                "score": 0.66,
                                "area": float(area),
                            })
                keep_idx = _nms([c["bbox"] for c in target_candidates], [c["score"] for c in target_candidates], iou_thresh=0.35)
                regions_by_target[target] = [target_candidates[i] for i in keep_idx][:100]
                counts[target] = len(regions_by_target[target])

            # G. Vegetation
            elif is_vegetation:
                exg = (2.0 * rgb[:, :, 1].astype(float) - rgb[:, :, 0].astype(float) - rgb[:, :, 2].astype(float))
                veg_mask = exg > 15.0
                veg_cnts, _ = cv2.findContours(veg_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in veg_cnts:
                    area = cv2.contourArea(c)
                    if area >= 100:
                        bx, by, bw, bh = cv2.boundingRect(c)
                        target_candidates.append({
                            "bbox": [float(bx), float(by), float(bx + bw), float(by + bh)],
                            "score": 0.70,
                            "area": float(area),
                        })
                keep_idx = _nms([c["bbox"] for c in target_candidates], [c["score"] for c in target_candidates], iou_thresh=0.40)
                regions_by_target[target] = [target_candidates[i] for i in keep_idx][:100]
                counts[target] = len(regions_by_target[target])

            # H. Generic fallback object
            else:
                _, th_gen = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cnts, _ = cv2.findContours(th_gen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    area = cv2.contourArea(c)
                    if 60 <= area <= (h * w * 0.35):
                        bx, by, bw, bh = cv2.boundingRect(c)
                        target_candidates.append({
                            "bbox": [float(bx), float(by), float(bx + bw), float(by + bh)],
                            "score": 0.65,
                            "area": float(area),
                        })
                keep_idx = _nms([c["bbox"] for c in target_candidates], [c["score"] for c in target_candidates], iou_thresh=0.35)
                regions_by_target[target] = [target_candidates[i] for i in keep_idx][:50]
                counts[target] = len(regions_by_target[target])

        # 3. Assemble detections and evidence
        regions = []
        evidence = []
        instance_idx = 0

        for target, cand_list in regions_by_target.items():
            for item in cand_list:
                instance_idx += 1
                score = float(item["score"])
                c_score = Confidence(
                    value=score, kind=ConfidenceKind.ESTIMATED, level=ConfidenceLevel.MEDIUM,
                    calibrated=False, source="grounding-contours-fallback",
                    explanation=f"Spatial morphology candidate detection for '{target}'."
                )
                reg_id = f"{target.replace(' ', '_')}-{instance_idx:03d}"
                evidence.append(Evidence(
                    id=reg_id, type=EvidenceType.SPATIAL_REGION, label=target,
                    source="grounding-contours-fallback", method="fallback", image_index=0,
                    bbox=item["bbox"], area_px=item["area"], confidence=c_score
                ))
                regions.append({"label": target, "bbox": item["bbox"], "score": score, "method": "fallback"})

        base = save_rgb(ctx, "base-0", rgb, ctx.images[0].info.filename, kind="base", image_index=0, group="grounding")

        # Visualization mask: draw bounding boxes on an overlay
        vis = [base]
        overlay_img = np.zeros((h, w, 3), dtype=np.uint8)
        for r in regions:
            x0, y0, x1, y1 = map(int, r["bbox"])
            cv2.rectangle(overlay_img, (x0, y0), (x1, y1), (0, 220, 255), 2)
            cv2.putText(overlay_img, r["label"], (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        vis.append(save_overlay(ctx, "mask-0-candidates", overlay_img, "Detected Candidates (Spatial Filter)",
                                kind="mask", image_index=0, group="grounding"))

        gsd, gsd_src = ctx.gsd(0)
        n = len(regions)
        summary_parts = [f"{cnt} '{t}'" for t, cnt in counts.items() if cnt > 0]
        zero_parts = [f"0 '{t}'" for t, cnt in counts.items() if cnt == 0]
        summary_str = ", ".join(summary_parts) if summary_parts else f"{n} region(s)"
        if zero_parts:
            summary_str += f" ({', '.join(zero_parts)} detected)"

        ans = f"Located a total of {n} candidate region(s): {summary_str} via spatial edge and morphology-based object detection."

        model_ref = ModelRef(
            id="grounding-contours-fallback", name="Contour-based Spatial Detector",
            version="1.0", backend="opencv", kind="fallback"
        )
        return SpecialistOutput(
            answer=ans,
            confidence=Confidence(
                value=0.72 if n > 0 else 0.85, kind=ConfidenceKind.ESTIMATED, level=ConfidenceLevel.MEDIUM,
                calibrated=False, source="grounding-contours-fallback",
                explanation="Domain-aware spatial candidate isolation executed under demo fallback mode."
            ),
            model=model_ref,
            models_used=[model_ref],
            evidence=evidence,
            visualizations=vis,
            result={
                "task": "grounding", "query": ctx.query, "detections": regions,
                "counts": counts, "total_instances": n, "gsd_m": gsd, "gsd_source": gsd_src
            },
            caveats=caveats,
            fallback_used=True,
        )

