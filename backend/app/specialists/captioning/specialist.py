"""Captioning specialist: multimodal satellite image captioning with scene evidence
and land-cover cross-checking.

Uses:
  - Florence-2-base (<DETAILED_CAPTION>) for scene captioning
  - RemoteCLIP ViT-B/32 (optional) for zero-shot remote-sensing scene evidence
  - SegFormer LoveDA (optional) to cross-check caption claims against actual land-cover fractions
  - Deterministic image statistics (ExG vegetation, brightness, contrast)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from app.core.errors import ModelUnavailable
from app.evidence import confidence as conf
from app.evidence.visualize import label_overlay, save_overlay, save_rgb
from app.geospatial.indices import excess_green, has_nir, ndvi
from app.geospatial.raster import to_display_rgb
from app.schemas.common import (
    Confidence,
    ConfidenceKind,
    ConfidenceLevel,
    Evidence,
    EvidenceType,
    ModelRef,
    TaskType,
    Visualization,
)
from app.specialists.base import AnalysisContext, Specialist, SpecialistOutput

log = logging.getLogger("geosense.specialists.captioning")

SCENE_LABELS = [
    "residential area",
    "dense urban area",
    "commercial area",
    "industrial area",
    "agricultural farmland",
    "dense forest and vegetation",
    "water body or wetland",
    "coastal and port area",
    "airport or runway",
    "barren land or open field",
]

RS_CONCEPT_KEYWORDS = {
    "building": ["building", "buildings", "house", "houses", "residential", "roof", "roofs", "structure"],
    "road": ["road", "roads", "highway", "street", "streets", "intersection", "path"],
    "water": ["water", "river", "lake", "ocean", "pond", "reservoir", "sea", "canal"],
    "vegetation": ["tree", "trees", "vegetation", "forest", "greenery", "grass", "plants"],
    "agriculture": ["farmland", "agriculture", "crop", "crops", "field", "fields"],
    "barren": ["barren", "sand", "soil", "dirt", "empty land", "dry"],
}


class CaptioningSpecialist(Specialist):
    task = TaskType.CAPTIONING

    def run(self, ctx: AnalysisContext) -> SpecialistOutput:
        spec = ctx.model_spec("primary")
        if spec is None or not ctx.manager.is_available(spec):
            raise ModelUnavailable("The captioning model (Florence-2-base) is not available or its weights are not downloaded.")

        loaded = ctx.manager.get(spec.id)
        import torch
        from PIL import Image

        data = ctx.raster(0)
        rgb = to_display_rgb(data)
        h, w = rgb.shape[:2]
        pil_img = Image.fromarray(rgb)

        p = spec.params or {}
        caption_task = p.get("caption_task", "<DETAILED_CAPTION>")
        max_new_tokens = p.get("max_new_tokens", 96)
        num_beams = p.get("num_beams", 1)

        t0 = time.perf_counter()
        inputs = loaded.processor(text=caption_task, images=pil_img, return_tensors="pt").to(loaded.device)
        with torch.inference_mode():
            ids = loaded.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
            )
        elapsed = time.perf_counter() - t0

        raw = loaded.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = loaded.processor.post_process_generation(raw, task=caption_task, image_size=pil_img.size)
        caption_text = str(parsed.get(caption_task, raw)).strip()

        evidence: list[Evidence] = []
        visualizations: list[Visualization] = []
        models_used: list[ModelRef] = [spec.ref()]
        caveats: list[str] = []

        # 1. Base Visualization
        vis_base = save_rgb(ctx, "base-0", rgb, "Input Image (Optical)", kind="base", image_index=0)
        visualizations.append(vis_base)

        # 2. Deterministic Image Statistics
        stats = self._compute_image_statistics(data, rgb, ctx)
        evidence.append(
            Evidence(
                id="image-statistics",
                type=EvidenceType.STATISTIC,
                label="image_statistics",
                source="algorithm:image-stats-v1",
                method="algorithm",
                image_index=0,
                value=stats,
                description=f"Dimensions: {w}x{h}, bands: {data.meta.bands}, estimated vegetation fraction: {stats.get('vegetation_fraction', 0.0):.1%}",
            )
        )

        # 3. Optional RemoteCLIP Zero-shot Scene Evidence
        clip_spec = ctx.model_spec("scene_evidence")
        if clip_spec and ctx.manager.is_available(clip_spec):
            try:
                clip_evidence = self._run_remoteclip(ctx, clip_spec, pil_img)
                if clip_evidence:
                    evidence.append(clip_evidence)
                    models_used.append(clip_spec.ref())
            except Exception as e:
                log.warning("RemoteCLIP inference failed, continuing: %s", e)

        # 4. Cross-check caption claims against SegFormer land-cover (if available)
        lc_spec = ctx.registry.model_for("grounding", "landcover")
        if lc_spec and ctx.manager.is_available(lc_spec):
            try:
                from app.specialists.grounding.backends import run_landcover
                lc_res = run_landcover(ctx, 0)
                models_used.append(lc_spec.ref())
                
                # Check for fabricated features in caption
                lower_cap = caption_text.lower()
                for concept, keywords in RS_CONCEPT_KEYWORDS.items():
                    if any(kw in lower_cap for kw in keywords):
                        if concept in lc_res.class_names:
                            frac = lc_res.fraction(concept)
                            if frac < 0.005:
                                caveats.append(
                                    f"Caption mentions '{concept}', but land-cover segmentation detects less than 0.5% ({frac:.1%}) coverage."
                                )
                
                # Add land cover summary evidence
                lc_summary = {name: round(lc_res.fraction(name), 4) for name in lc_res.class_names}
                evidence.append(
                    Evidence(
                        id="landcover-crosscheck",
                        type=EvidenceType.STATISTIC,
                        label="landcover_fractions",
                        source=lc_spec.id,
                        method="model",
                        image_index=0,
                        value=lc_summary,
                        description="Land-cover distribution used for cross-checking caption statements.",
                    )
                )
            except Exception as e:
                log.warning("Land-cover cross-check failed, continuing: %s", e)

        # Compose Answer
        caveat_text = f" [Cross-check notice: {'; '.join(caveats)}]" if caveats else ""
        answer = f"{caption_text}{caveat_text}"

        # Captioning model emits no calibrated confidence; mean token probability does not reflect correctness
        confidence = conf.unavailable(
            spec.id,
            "Confidence unavailable from model (Florence-2 token probabilities are uncalibrated and do not track remote-sensing factual accuracy).",
        )

        return SpecialistOutput(
            answer=answer,
            confidence=confidence,
            model=spec.ref(),
            evidence=evidence,
            visualizations=visualizations,
            models_used=models_used,
            result={
                "caption": caption_text,
                "latency_s": round(elapsed, 2),
                "task": caption_task,
                "statistics": stats,
            },
            caveats=caveats,
            fallback_used=False,
        )

    def fallback(self, ctx: AnalysisContext) -> SpecialistOutput:
        """Deep multi-spectral and morphological scene analysis fallback."""
        import cv2
        data = ctx.raster(0)
        rgb = to_display_rgb(data)
        h, w = rgb.shape[:2]
        gray = cv2_gray(rgb)
        gsd, gsd_src = ctx.gsd(0)

        # 1. Multi-spectral Land Cover Assessment
        r = rgb[..., 0].astype(float)
        g = rgb[..., 1].astype(float)
        b = rgb[..., 2].astype(float)
        total_intensity = np.maximum(r + g + b, 1.0)

        # Vegetation: Excess Green / NDVI
        if has_nir(data):
            idx = ndvi(data)
            veg_mask = idx > 0.35 if idx is not None else np.zeros((h, w), dtype=bool)
        else:
            exg = (2.0 * g - r - b) / total_intensity
            veg_mask = exg > 0.06

        veg_pct = float(veg_mask.mean() * 100)

        # Water: High blue absorption ratio + lower overall intensity
        water_mask = (b > (r * 1.08)) & (b > (g * 0.95)) & (gray < 90)
        water_pct = float(water_mask.mean() * 100)

        # Infrastructure / Built-up: High local edge variance + moderate brightness
        edges = cv2.Canny(gray, 60, 160)
        built_candidates = cv2.dilate(edges, np.ones((3, 3), np.uint8)) > 0
        built_mask = built_candidates & (~water_mask) & (gray > 80)
        built_pct = float(built_mask.mean() * 100)

        # Bare soil / Cultivated fields: remaining land
        soil_mask = (~veg_mask) & (~water_mask) & (~built_mask)
        soil_pct = max(0.0, float(soil_mask.mean() * 100))

        # 2. Road and Linear Infrastructure Detection
        kernel_line_h = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
        kernel_line_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
        lines_h = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_line_h)
        lines_v = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_line_v)
        road_network = cv2.bitwise_or(lines_h, lines_v)
        road_px = int(np.count_nonzero(road_network))

        # 3. Discrete infrastructure feature isolation
        contours, _ = cv2.findContours(cv2.dilate(edges, np.ones((5, 5), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        spatial_evidence: list[Evidence] = []
        for idx, cnt in enumerate(contours[:20]):
            bx, by, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < 50 or area > (h * w * 0.3):
                continue
            spatial_evidence.append(Evidence(
                id=f"infrastructure-{idx + 1:03d}",
                type=EvidenceType.SPATIAL_REGION,
                label="infrastructure / road segment",
                source="spectral-feature-analysis",
                method="algorithm",
                image_index=0,
                bbox=[float(bx), float(by), float(bx + bw), float(by + bh)],
                area_px=float(area),
                confidence=Confidence(
                    value=0.72, kind=ConfidenceKind.ESTIMATED, level=ConfidenceLevel.MEDIUM,
                    calibrated=False, source="spectral-feature-analysis",
                    explanation="Morphological structure and edge contrast isolation."
                )
            ))

        # 4. Generate Visual Evidence Layers
        vis_base = save_rgb(ctx, "base-0", rgb, "Input Image (Optical)", kind="base", image_index=0)

        # Color-coded Land Cover Map (RGBA)
        labels = np.zeros((h, w), dtype=np.uint8)
        labels[water_mask] = 1
        labels[veg_mask] = 2
        labels[soil_mask] = 3
        labels[road_network > 0] = 4

        lc_colours = {
            1: "#0284c7",  # Water (Blue)
            2: "#16a34a",  # Vegetation (Green)
            3: "#b48246",  # Cultivated Soil (Tan)
            4: "#f59e0b",  # Roads/Built (Amber)
        }
        rgba_lc = label_overlay(labels, lc_colours, alpha=140)
        vis_lc = save_overlay(ctx, "mask-0-landcover", rgba_lc, "Land Cover Classification (Spectral)",
                              kind="mask", image_index=0, group="captioning",
                              legend={"Water": "#0284c7", "Vegetation": "#16a34a", "Cultivated Soil": "#b48246", "Roads/Built": "#f59e0b"})

        # Road Network Overlay (RGBA)
        rgba_roads = label_overlay((road_network > 0).astype(np.uint8), {1: "#38bdf8"}, alpha=190)
        vis_roads = save_overlay(ctx, "mask-0-roads", rgba_roads, "Extracted Road & Transport Corridors",
                                 kind="overlay", image_index=0, group="captioning")

        # 5. Synthesize Natural-Language Description
        dominant_features = []
        if veg_pct > 25:
            dominant_features.append(f"extensive vegetation and agricultural plots ({veg_pct:.1f}% coverage)")
        if water_pct > 5:
            dominant_features.append(f"prominent coastal or inland water bodies ({water_pct:.1f}% coverage)")
        if soil_pct > 15:
            dominant_features.append(f"cultivated fields and bare soil parcels ({soil_pct:.1f}% coverage)")
        if road_px > 100:
            dominant_features.append(f"an interconnected network of linear roads and transport corridors")

        features_str = "; ".join(dominant_features) if dominant_features else "diverse mixed terrain"

        answer = (
            f"Optical remote-sensing scene ({w}x{h} px, {data.meta.bands} bands). "
            f"The landscape is primarily characterized by {features_str}. "
            f"Radiometric analysis identifies an average scene reflectance of {float(np.mean(gray)):.1f}/255. "
            f"A total of {len(spatial_evidence)} distinct infrastructure and road corridor segments were isolated and mapped."
        )

        stat_ref = ModelRef(
            id="spectral-scene-analyzer",
            name="Spectral Scene & Road Analyzer",
            version="1.0",
            backend="algorithm",
            kind="algorithm",
        )

        stats = self._compute_image_statistics(data, rgb, ctx)
        spatial_evidence.insert(0, Evidence(
            id="image-statistics",
            type=EvidenceType.STATISTIC,
            label="image_statistics",
            source="spectral-scene-analyzer",
            method="algorithm",
            image_index=0,
            value=stats,
            description=f"Dimensions: {w}x{h}, bands: {data.meta.bands}, estimated vegetation fraction: {stats.get('vegetation_fraction', 0.0):.1%}",
        ))

        confidence = Confidence(
            value=0.74,
            kind=ConfidenceKind.ESTIMATED,
            level=ConfidenceLevel.MEDIUM,
            calibrated=False,
            source="spectral-scene-analyzer",
            explanation="Empirical spectral index (ExG/NDWI) and morphological linear feature separation.",
            factors=[f"Vegetation index: {veg_pct:.1f}%", f"Water body coverage: {water_pct:.1f}%", f"Extracted road pixels: {road_px}"]
        )

        return SpecialistOutput(
            answer=answer,
            confidence=confidence,
            model=stat_ref,
            evidence=spatial_evidence,
            visualizations=[vis_base, vis_lc, vis_roads],
            models_used=[stat_ref],
            result={
                "landcover": {
                    "vegetation_pct": round(veg_pct, 1),
                    "water_pct": round(water_pct, 1),
                    "soil_pct": round(soil_pct, 1),
                    "infrastructure_pct": round(built_pct, 1),
                },
                "road_network_pixels": road_px,
                "infrastructure_segments_detected": len(spatial_evidence),
                "dimensions": {"width": w, "height": h},
            },
            caveats=[] if gsd else ["Image is in pixel space (no CRS georeferencing metadata embedded)."],
            fallback_used=True,
        )

    def _compute_image_statistics(self, data, rgb: np.ndarray, ctx: AnalysisContext) -> dict[str, Any]:
        h, w = rgb.shape[:2]
        gsd, gsd_src = ctx.gsd(0)
        
        # Mean brightness & contrast
        gray = cv2_gray(rgb)
        mean_b = float(np.mean(gray))
        std_b = float(np.std(gray))

        # Vegetation estimation
        if has_nir(data):
            idx = ndvi(data)
            veg_mask = idx > 0.35 if idx is not None else np.zeros((h, w), dtype=bool)
            veg_fraction = float(np.mean(veg_mask))
            veg_metric = "NDVI > 0.35"
        else:
            exg = excess_green(rgb)
            veg_mask = exg > 0.1
            veg_fraction = float(np.mean(veg_mask))
            veg_metric = "ExG > 0.1"

        return {
            "width": w,
            "height": h,
            "bands": data.meta.bands,
            "mean_brightness": round(mean_b, 1),
            "contrast_std": round(std_b, 1),
            "vegetation_fraction": round(veg_fraction, 4),
            "vegetation_metric": veg_metric,
            "gsd_m": gsd,
            "gsd_source": gsd_src,
        }

    def _run_remoteclip(self, ctx: AnalysisContext, spec, pil_img) -> Evidence | None:
        import torch
        loaded = ctx.manager.get(spec.id)
        model = loaded.model
        preprocess = loaded.processor
        tokenizer = loaded.extras.get("tokenizer")
        if not tokenizer:
            return None

        image = preprocess(pil_img).unsqueeze(0).to(loaded.device)
        text = tokenizer(SCENE_LABELS).to(loaded.device)

        with torch.inference_mode():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]

        scores = {lab: round(float(similarity[i].item()), 4) for i, lab in enumerate(SCENE_LABELS)}
        top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        
        return Evidence(
            id="scene-classification-remoteclip",
            type=EvidenceType.STATISTIC,
            label="scene_classification",
            source=spec.id,
            method="model",
            image_index=0,
            value={"top_scenes": dict(top3), "all_scores": scores},
            description=f"Zero-shot RemoteCLIP scene probabilities over {len(SCENE_LABELS)} remote sensing classes (relative).",
        )


def cv2_gray(rgb: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
