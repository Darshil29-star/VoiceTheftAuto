"""Dataset loader for Remote Sensing visual grounding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset
from PIL import Image

from training.grounding.preprocessing import normalize_bbox


class RemoteSensingGroundingDataset(Dataset):
    """Loads image-query-bounding box annotations.
    
    Supports DIOR-RSVG and custom GeoSense annotations format:
    [
      {
        "image": "dior_001.jpg",
        "query": "locate the storage tanks",
        "bboxes": [[120, 140, 200, 220], [300, 310, 380, 390]],
        "labels": ["storage tank", "storage tank"]
      },
      ...
    ]
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        processor: Any = None,
        task_prefix: str = "<CAPTION_TO_PHRASE_GROUNDING>",
        target_size: tuple[int, int] = (640, 640),
    ):
        self.image_root = Path(image_root)
        self.processor = processor
        self.task_prefix = task_prefix
        self.target_size = target_size
        self.samples: list[dict[str, Any]] = []

        path = Path(manifest_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                self.samples = json.load(f)
        else:
            # Fallback mock/sample generator if user dataset not yet annotated
            for img_file in self.image_root.glob("*.jpg"):
                self.samples.append({
                    "image": img_file.name,
                    "query": "locate buildings",
                    "bboxes": [[50, 50, 200, 200]],
                    "labels": ["building"],
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        img_path = self.image_root / item["image"]
        
        if img_path.exists():
            img = Image.open(str(img_path)).convert("RGB")
        else:
            img = Image.new("RGB", self.target_size, color=(128, 128, 128))

        orig_w, orig_h = img.size
        query = item.get("query", "locate objects")
        bboxes = item.get("bboxes", [])
        labels = item.get("labels", [])

        # Normalize boxes to Florence-2 [0, 1000] space
        norm_boxes = [normalize_bbox(b, orig_w, orig_h) for b in bboxes]

        if self.processor is not None:
            prompt = f"{self.task_prefix}{query}"
            inputs = self.processor(text=prompt, images=img, return_tensors="pt")
            item_data = {k: v.squeeze(0) for k, v in inputs.items()}
            item_data["raw_query"] = query
            item_data["bboxes"] = torch.tensor(bboxes, dtype=torch.float32)
            item_data["labels"] = labels
            return item_data

        return {
            "image": img,
            "query": query,
            "bboxes": bboxes,
            "labels": labels,
            "orig_size": (orig_w, orig_h),
        }
