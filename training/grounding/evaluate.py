"""Evaluation script for Visual Grounding on Remote Sensing imagery."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import yaml
import torch
from transformers import AutoProcessor, Florence2ForConditionalGeneration

from training.grounding.dataset import RemoteSensingGroundingDataset
from training.grounding.preprocessing import compute_iou, denormalize_bbox

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("geosense.eval.grounding")


def evaluate(config_path: str = "training/grounding/config.yaml", checkpoint_dir: str | None = None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = checkpoint_dir or cfg["training"].get("output_dir", "models/checkpoints/grounding-finetuned")
    if not Path(ckpt).exists():
        ckpt = cfg["model"]["name"]

    processor = AutoProcessor.from_pretrained(ckpt)
    model = Florence2ForConditionalGeneration.from_pretrained(ckpt).to(device).eval()

    val_dataset = RemoteSensingGroundingDataset(
        manifest_path=cfg["data"].get("val_manifest", "data/annotations/grounding_val.json"),
        image_root=cfg["data"]["image_root"],
    )

    if len(val_dataset) == 0:
        log.warning("No grounding validation samples found.")
        return

    task = cfg["model"].get("task_prefix", "<CAPTION_TO_PHRASE_GROUNDING>")
    total_samples = min(len(val_dataset), 50)
    ious = []
    hits_at_50 = 0

    log.info("Evaluating grounding accuracy over %d samples...", total_samples)
    for i in range(total_samples):
        sample = val_dataset[i]
        img = sample["image"]
        query = sample["query"]
        gt_boxes = sample["bboxes"]
        orig_w, orig_h = sample["orig_size"]

        prompt = f"{task}{query}"
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=128, num_beams=1)
        raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(raw, task=task, image_size=img.size).get(task, {})

        pred_boxes_norm = parsed.get("bboxes", [])
        pred_boxes = [denormalize_bbox(b, orig_w, orig_h) for b in pred_boxes_norm]

        sample_max_iou = 0.0
        for pred in pred_boxes:
            for gt in gt_boxes:
                iou = compute_iou(pred, gt)
                if iou > sample_max_iou:
                    sample_max_iou = iou

        ious.append(sample_max_iou)
        if sample_max_iou >= 0.5:
            hits_at_50 += 1

    mean_iou = sum(ious) / max(len(ious), 1)
    rec_50 = hits_at_50 / max(total_samples, 1)

    log.info("Visual Grounding Evaluation Results:")
    log.info("  Mean IoU     : %.4f", mean_iou)
    log.info("  Precision@0.5: %.2f%%", rec_50 * 100)

    out_file = Path("docs/grounding_evaluation_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump({"mean_iou": mean_iou, "precision_at_0.5": rec_50, "evaluated_samples": total_samples}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/grounding/config.yaml")
    args = parser.parse_args()
    evaluate(args.config)
