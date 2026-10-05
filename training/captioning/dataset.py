"""Dataset loader for Remote Sensing image-caption pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset
from PIL import Image

from training.captioning.preprocessing import preprocess_image


class RemoteSensingCaptionDataset(Dataset):
    """Loads image-caption pairs from JSON manifest.
    
    Expected JSON schema:
    [
      {"image": "relative/path/to/image.png", "caption": "An aerial view of a harbor with cargo ships..."},
      ...
    ]
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        processor: Any = None,
        task_prefix: str = "<DETAILED_CAPTION>",
        target_size: tuple[int, int] = (384, 384),
        max_length: int = 96,
    ):
        self.image_root = Path(image_root)
        self.processor = processor
        self.task_prefix = task_prefix
        self.target_size = target_size
        self.max_length = max_length
        self.samples: list[dict[str, str]] = []

        path = Path(manifest_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                self.samples = json.load(f)
        else:
            # Fallback: scan image_root if manifest does not exist
            for img_file in self.image_root.glob("*.jpg"):
                self.samples.append({
                    "image": img_file.name,
                    "caption": f"Satellite view of {img_file.stem.replace('_', ' ')}."
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        img_path = self.image_root / item["image"]
        
        if img_path.exists():
            img = preprocess_image(str(img_path), self.target_size)
        else:
            img = Image.new("RGB", self.target_size, color=(128, 128, 128))

        caption = item.get("caption", "Remote sensing scene.")

        if self.processor is not None:
            inputs = self.processor(
                text=self.task_prefix,
                images=img,
                return_tensors="pt"
            )
            # Remove batch dim
            item_data = {k: v.squeeze(0) for k, v in inputs.items()}
            
            # Target labels
            target = self.processor.tokenizer(
                caption,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
            )
            labels = target.input_ids.squeeze(0)
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            item_data["labels"] = labels
            item_data["raw_caption"] = caption
            return item_data

        return {"image": img, "caption": caption}
