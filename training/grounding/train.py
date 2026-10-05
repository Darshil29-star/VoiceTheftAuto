"""Training script for Visual Grounding fine-tuning on Remote Sensing datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Florence2ForConditionalGeneration

from training.grounding.dataset import RemoteSensingGroundingDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("geosense.train.grounding")


def train(config_path: str = "training/grounding/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = cfg["training"].get("device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else device_str
    log.info("Using training device: %s", device)

    model_name = cfg["model"]["name"]
    log.info("Loading base grounding model: %s", model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = Florence2ForConditionalGeneration.from_pretrained(model_name).to(device)

    # Optional LoRA configuration
    if cfg["model"].get("use_lora", False):
        try:
            from peft import LoraConfig, get_peft_model
            lora_cfg = cfg["model"].get("lora", {})
            peft_config = LoraConfig(
                r=lora_cfg.get("r", 8),
                lora_alpha=lora_cfg.get("lora_alpha", 16),
                target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
            log.info("PEFT/LoRA adapter initialized.")
        except ImportError:
            log.warning("peft package not installed; fine-tuning language head layers.")
            for param in model.vision_tower.parameters():
                param.requires_grad = False

    dataset = RemoteSensingGroundingDataset(
        manifest_path=cfg["data"]["train_manifest"],
        image_root=cfg["data"]["image_root"],
        processor=processor,
        task_prefix=cfg["model"].get("task_prefix", "<CAPTION_TO_PHRASE_GROUNDING>"),
    )

    if len(dataset) == 0:
        log.warning("No grounding training data found.")
        return

    def collate_fn(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]).to(device),
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]).to(device),
        }

    loader = DataLoader(dataset, batch_size=cfg["training"].get("batch_size", 2), shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"].get("learning_rate", 2e-5)),
    )

    epochs = cfg["training"].get("epochs", 3)
    out_dir = Path(cfg["training"].get("output_dir", "models/checkpoints/grounding-finetuned"))
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting Grounding training: %d epochs", epochs)
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            out = model(**batch, labels=batch["input_ids"])
            loss = out.loss
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        log.info("Epoch %d/%d completed | Avg loss: %.4f", ep + 1, epochs, ep_loss / max(len(loader), 1))

    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    log.info("Grounding checkpoint saved to %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/grounding/config.yaml")
    args = parser.parse_args()
    train(args.config)
