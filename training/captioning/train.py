"""Training script for Florence-2 Remote Sensing captioning fine-tuning.

Explicitly invoked via CLI:
  python training/captioning/train.py --config training/captioning/config.yaml

Supports:
  - CPU-first mode & optional CUDA
  - LoRA / PEFT or parameter-efficient freezing
  - Gradient accumulation for low-memory execution
  - Portability to Google Colab, Kaggle, or cloud GPUs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Florence2ForConditionalGeneration

from training.captioning.dataset import RemoteSensingCaptionDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("geosense.train.captioning")


def train(config_path: str = "training/captioning/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = cfg["training"].get("device", "auto")
    if device_str == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str
    log.info("Using training device: %s", device)

    model_name = cfg["model"]["name"]
    log.info("Loading model and processor: %s", model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = Florence2ForConditionalGeneration.from_pretrained(model_name).to(device)

    # Apply LoRA if requested and available
    if cfg["model"].get("use_lora", False):
        try:
            from peft import LoraConfig, get_peft_model
            lora_cfg = cfg["model"].get("lora", {})
            peft_config = LoraConfig(
                r=lora_cfg.get("r", 8),
                lora_alpha=lora_cfg.get("lora_alpha", 16),
                target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=lora_cfg.get("lora_dropout", 0.05),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
            log.info("PEFT/LoRA applied successfully. Trainable params: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))
        except ImportError:
            log.warning("peft library not found; freezing vision backbone and training language projection layers only.")
            for param in model.vision_tower.parameters():
                param.requires_grad = False

    train_data = RemoteSensingCaptionDataset(
        manifest_path=cfg["data"]["train_manifest"],
        image_root=cfg["data"]["image_root"],
        processor=processor,
        task_prefix=cfg["model"].get("task_prefix", "<DETAILED_CAPTION>"),
        target_size=tuple(cfg["data"].get("image_size", (384, 384))),
    )

    if len(train_data) == 0:
        log.warning("No training samples found in manifest or image directory. Please prepare dataset.")
        return

    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        return {"input_ids": input_ids.to(device), "pixel_values": pixel_values.to(device), "labels": labels.to(device)}

    batch_size = cfg["training"].get("batch_size", 2)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"].get("learning_rate", 1e-5)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )

    grad_accum = cfg["training"].get("gradient_accumulation_steps", 4)
    epochs = cfg["training"].get("epochs", 3)
    out_dir = Path(cfg["training"].get("output_dir", "models/checkpoints/florence-2-caption-finetuned"))
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting training loop: %d epochs, %d batches/epoch", epochs, len(loader))
    model.train()
    step = 0
    for epoch in range(epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        for i, batch in enumerate(loader):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum

            if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step % 10 == 0:
                    log.info("Epoch %d/%d | Step %d | Batch loss: %.4f", epoch + 1, epochs, step, loss.item() * grad_accum)

        log.info("Epoch %d completed. Average loss: %.4f", epoch + 1, total_loss / max(len(loader), 1))

    # Save fine-tuned checkpoint
    log.info("Saving checkpoint to %s", out_dir)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    log.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Florence-2 on Remote Sensing captions")
    parser.add_argument("--config", default="training/captioning/config.yaml", help="Path to config yaml")
    args = parser.parse_args()
    train(args.config)
