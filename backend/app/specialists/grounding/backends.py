"""Grounding backends. Each returns results in ANALYSIS-resolution pixel coordinates of ctx.raster(i).

landcover  — semantic segmentation (SegFormer-LoveDA), GSD-aware rescaling + overlapping tiles
detector   — open-vocabulary detection (Grounding DINO), tiled for large inputs, cross-tile NMS
sam        — box-prompted mask refinement (SlimSAM)
florence   — Florence-2 phrase grounding (alternate; weak on dense small objects, see BENCHMARKS.md)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.errors import ModelUnavailable
from app.geospatial.raster import to_display_rgb
from app.models.registry import ModelSpec
from app.specialists.base import AnalysisContext
from app.specialists.grounding.postprocess import nms


def _tiles(h: int, w: int, tile: int, overlap: int) -> list[tuple[int, int, int, int]]:
    if h <= tile and w <= tile:
        return [(0, 0, w, h)]
    step = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    return [(x, y, min(x + tile, w), min(y + tile, h)) for y in ys for x in xs]


# --------------------------------------------------------------------------- land cover
@dataclass
class LandcoverResult:
    labels: np.ndarray                    # (h, w) uint8 canonical-class index into class_names, 255 = none
    prob: np.ndarray                      # (h, w) float16 probability of the assigned label
    class_names: list[str]                # e.g. ["building", "road", ...]
    spec: ModelSpec
    rescale: float
    gsd_m: float | None
    gsd_source: str
    in_domain: bool
    tiles: int
    seconds: float
    notes: list[str] = field(default_factory=list)

    def mask(self, name: str) -> np.ndarray:
        return self.labels == self.class_names.index(name)

    def fraction(self, name: str) -> float:
        return float(self.mask(name).mean())


def run_landcover(ctx: AnalysisContext, image_index: int = 0) -> LandcoverResult:
    key = f"landcover:{image_index}"
    if key in ctx.cache:
        return ctx.cache[key]
    spec = ctx.registry.model_for("grounding", "landcover")
    if spec is None:
        raise ModelUnavailable("No land-cover segmentation model is configured (tasks.grounding.landcover).")
    loaded = ctx.manager.get(spec.id)
    import torch

    p = spec.params
    data = ctx.raster(image_index)
    rgb = to_display_rgb(data)
    h, w = rgb.shape[:2]
    gsd, gsd_src = ctx.gsd(image_index)
    analysis_gsd = gsd * data.scale if gsd else None
    notes: list[str] = []

    rescale = 1.0
    if analysis_gsd:
        rescale = min(max(analysis_gsd / p.get("target_gsd_m", 0.3), 0.5), p.get("max_rescale", 2.0))
    max_side = p.get("max_working_side", 3072)
    if max(h, w) * rescale > max_side:
        new = max_side / max(h, w)
        notes.append(f"Rescale reduced from {rescale:.2f} to {new:.2f} to bound CPU time on a large image.")
        rescale = new
    in_domain = analysis_gsd is None or analysis_gsd <= p.get("domain_max_gsd_m", 2.0)
    if not in_domain:
        notes.append(
            f"Image GSD ≈ {analysis_gsd:.1f} m is much coarser than the model's training data (~0.3 m); "
            "individual objects are not resolvable and land-cover labels are unreliable."
        )

    work = cv2.resize(rgb, (round(w * rescale), round(h * rescale)), interpolation=cv2.INTER_CUBIC) if rescale != 1 else rgb
    H, W = work.shape[:2]
    id2label = {int(k): v for k, v in loaded.extras["id2label"].items()}
    n_cls = len(id2label)
    acc = np.zeros((n_cls, H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    tiles = _tiles(H, W, p.get("tile_size", 512), p.get("tile_overlap", 64))
    t0 = time.perf_counter()
    proc = loaded.processor
    for x0, y0, x1, y1 in tiles:
        crop = work[y0:y1, x0:x1]
        inputs = proc(images=crop, return_tensors="pt", do_resize=False)
        with torch.inference_mode():
            logits = loaded.model(pixel_values=inputs["pixel_values"].to(loaded.device)).logits
            probs = torch.nn.functional.interpolate(logits, size=crop.shape[:2], mode="bilinear", align_corners=False)
            probs = probs.softmax(1)[0].float().cpu().numpy()
        acc[:, y0:y1, x0:x1] += probs
        cnt[y0:y1, x0:x1] += 1
    acc /= np.maximum(cnt, 1)
    seconds = time.perf_counter() - t0

    class_map = {int(k): v for k, v in p.get("class_map", {}).items()}
    class_names = list(dict.fromkeys(class_map.values()))
    raw_label = acc.argmax(0)
    raw_prob = acc.max(0)
    labels = np.full((H, W), 255, np.uint8)
    for model_id, name in class_map.items():
        labels[raw_label == model_id] = class_names.index(name)
    if (H, W) != (h, w):
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        raw_prob = cv2.resize(raw_prob, (w, h), interpolation=cv2.INTER_LINEAR)

    result = LandcoverResult(labels=labels, prob=raw_prob.astype(np.float16), class_names=class_names, spec=spec,
                             rescale=rescale, gsd_m=gsd, gsd_source=gsd_src, in_domain=in_domain, tiles=len(tiles),
                             seconds=seconds, notes=notes)
    ctx.cache[key] = result
    return result


# --------------------------------------------------------------------------- open-vocabulary detection
@dataclass
class Detection:
    bbox: list[float]      # analysis pixels
    score: float
    phrase: str


@dataclass
class DetectorResult:
    detections: list[Detection]
    spec: ModelSpec
    tiles: int
    seconds: float
    filtered_group_boxes: int


def run_detector(ctx: AnalysisContext, prompt: str, small_object: bool, image_index: int = 0) -> DetectorResult:
    spec = ctx.registry.model_for("grounding", "primary")
    if spec is None:
        raise ModelUnavailable("No open-vocabulary detector configured (tasks.grounding.primary).")
    loaded = ctx.manager.get(spec.id)
    import torch
    from PIL import Image

    p = spec.params
    rgb = to_display_rgb(ctx.raster(image_index))
    h, w = rgb.shape[:2]
    tile = p.get("tile_size", 640)
    tiles = _tiles(h, w, tile, p.get("tile_overlap", 96)) if max(h, w) > p.get("tile_when_larger_than", 900) else [(0, 0, w, h)]
    boxes, scores, phrases = [], [], []
    filtered = 0
    t0 = time.perf_counter()
    text = prompt if prompt.strip().endswith(".") else prompt.strip() + " ."
    for x0, y0, x1, y1 in tiles:
        crop = Image.fromarray(rgb[y0:y1, x0:x1])
        inputs = loaded.processor(images=crop, text=text.lower(), return_tensors="pt").to(loaded.device)
        with torch.inference_mode():
            out = loaded.model(**inputs)
        res = loaded.processor.post_process_grounded_object_detection(
            out, inputs["input_ids"], threshold=p.get("box_threshold", 0.25), text_threshold=p.get("text_threshold", 0.2),
            target_sizes=[(y1 - y0, x1 - x0)],
        )[0]
        tile_area = (x1 - x0) * (y1 - y0)
        labels = res.get("text_labels", res.get("labels"))
        for b, s, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
            frac = ((b[2] - b[0]) * (b[3] - b[1])) / tile_area
            # zero-shot detectors often emit "group" boxes spanning many objects
            if frac > 0.6 or (small_object and frac > 0.15):
                filtered += 1
                continue
            boxes.append([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0])
            scores.append(float(s))
            phrases.append(str(lab))
    keep = nms(boxes, scores, p.get("nms_iou", 0.5))[: p.get("max_detections", 300)]
    dets = [Detection(boxes[i], scores[i], phrases[i]) for i in keep]
    return DetectorResult(dets, spec, len(tiles), time.perf_counter() - t0, filtered)


# --------------------------------------------------------------------------- SAM masks
def refine_with_sam(ctx: AnalysisContext, boxes: list[list[float]], image_index: int = 0) -> tuple[list[np.ndarray], list[float], ModelSpec] | None:
    """Box-prompted masks (analysis resolution). Returns None if no mask model is available."""
    spec = ctx.registry.model_for("grounding", "mask_refiner")
    if spec is None or not boxes or not ctx.manager.is_available(spec):
        return None
    loaded = ctx.manager.get(spec.id)
    import torch

    rgb = to_display_rgb(ctx.raster(image_index))
    boxes = boxes[: spec.params.get("max_boxes", 60)]
    inputs = loaded.processor(rgb, input_boxes=[boxes], return_tensors="pt").to(loaded.device)
    with torch.inference_mode():
        out = loaded.model(**inputs, multimask_output=False)
    masks = loaded.processor.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )[0]  # (n_boxes, 1, H, W)
    ious = out.iou_scores.reshape(-1).cpu().tolist()
    return [masks[i, 0].numpy().astype(bool) for i in range(masks.shape[0])], [float(v) for v in ious], spec


# --------------------------------------------------------------------------- Florence-2 phrase grounding
def run_florence_phrase_grounding(ctx: AnalysisContext, phrase: str, image_index: int = 0) -> tuple[list[Detection], ModelSpec]:
    spec = next((s for s in ctx.registry.alternates_for("grounding") if s.backend == "florence2"), None)
    if spec is None:
        raise ModelUnavailable("Florence-2 grounding alternate is not configured.")
    loaded = ctx.manager.get(spec.id)
    import torch
    from PIL import Image

    img = Image.fromarray(to_display_rgb(ctx.raster(image_index)))
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    inputs = loaded.processor(text=task + phrase, images=img, return_tensors="pt").to(loaded.device)
    with torch.inference_mode():
        ids = loaded.model.generate(**inputs, max_new_tokens=256, num_beams=1, do_sample=False)
    raw = loaded.processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = loaded.processor.post_process_generation(raw, task=task, image_size=img.size)[task]
    # Florence-2 emits no per-box score
    return [Detection(b, float("nan"), lab) for b, lab in zip(parsed.get("bboxes", []), parsed.get("labels", []))], spec
