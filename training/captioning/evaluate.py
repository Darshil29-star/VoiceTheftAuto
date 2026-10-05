"""Evaluation script for Remote Sensing image captioning."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import yaml
import torch
from transformers import AutoProcessor, Florence2ForConditionalGeneration

from training.captioning.dataset import RemoteSensingCaptionDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("geosense.eval.captioning")


def evaluate(config_path: str = "training/captioning/config.yaml", checkpoint_dir: str | None = None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = checkpoint_dir or cfg["training"].get("output_dir", "models/checkpoints/florence-2-caption-finetuned")
    
    if not Path(ckpt).exists():
        ckpt = cfg["model"]["name"]
        log.info("Checkpoint not found at %s; evaluating base model %s", ckpt, cfg['model']['name'])

    log.info("Loading model from %s on %s", ckpt, device)
    processor = AutoProcessor.from_pretrained(ckpt)
    model = Florence2ForConditionalGeneration.from_pretrained(ckpt).to(device).eval()

    val_dataset = RemoteSensingCaptionDataset(
        manifest_path=cfg["data"].get("val_manifest", "data/annotations/caption_val.json"),
        image_root=cfg["data"]["image_root"],
        task_prefix=cfg["model"].get("task_prefix", "<DETAILED_CAPTION>"),
    )

    if len(val_dataset) == 0:
        log.warning("No validation samples available.")
        return

    predictions = []
    task = cfg["model"].get("task_prefix", "<DETAILED_CAPTION>")
    log.info("Evaluating %d validation samples...", len(val_dataset))

    for i in range(min(len(val_dataset), 50)):
        sample = val_dataset[i]
        img = sample["image"]
        ref_caption = sample["caption"]

        inputs = processor(text=task, images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=96, num_beams=1)
        raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(raw, task=task, image_size=img.size)
        pred_caption = parsed.get(task, raw)

        predictions.append({
            "sample_index": i,
            "reference": ref_caption,
            "prediction": pred_caption,
        })

    log.info("Evaluation sample results:")
    for p in predictions[:5]:
        log.info("REF : %s", p["reference"])
        log.info("PRED: %s", p["prediction"])
        log.info("-" * 40)

    out_file = Path("docs/caption_evaluation_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    log.info("Results saved to %s", out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/captioning/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)
