"""Dataset loader for BIFOLD-BigEarthNetv2-0/BigEarthNet.txt multimodal training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset
from PIL import Image


class BigEarthNetDataset(Dataset):
    """Loads image-instruction pairs from BigEarthNet.txt annotations.
    
    Expected JSON schema per entry:
    {
      "id": 1,
      "patch_id": "...",
      "type": "captioning" | "binary" | "mcq" | "bounding box",
      "category": "presence" | "area" | "count" | ...,
      "input": "Question or prompt instruction",
      "output": "Reference answer or description",
      "image": "images/patch_xxx.jpg"
    }
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        processor: Any = None,
        tokenizer: Any = None,
        target_size: tuple[int, int] = (224, 224),
        max_length: int = 128,
    ):
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.processor = processor
        self.tokenizer = tokenizer
        self.target_size = target_size
        self.max_length = max_length
        self.samples: list[dict[str, Any]] = []

        if self.manifest_path.exists():
            with self.manifest_path.open("r", encoding="utf-8") as f:
                self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def _format_prompt(self, item: dict[str, Any]) -> str:
        task_type = item.get("type", "captioning")
        user_input = item.get("input", "")

        if task_type == "captioning":
            return f"<DETAILED_CAPTION>{user_input}" if user_input else "<DETAILED_CAPTION>"
        elif task_type in ("binary", "mcq"):
            return f"<VQA>{user_input}"
        elif task_type == "bounding box":
            return f"<CAPTION_TO_PHRASE_GROUNDING>{user_input}"
        return f"<VQA>{user_input}"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        img_rel = item.get("image", "")
        img_path = self.image_root / img_rel

        if img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                if img.size != self.target_size:
                    img = img.resize(self.target_size, Image.BILINEAR)
            except Exception:
                img = Image.new("RGB", self.target_size, color=(60, 100, 60))
        else:
            img = Image.new("RGB", self.target_size, color=(60, 100, 60))

        prompt = self._format_prompt(item)
        target = str(item.get("output", ""))

        if self.processor is not None:
            # Florence-2 style processor
            try:
                inputs = self.processor(
                    text=prompt,
                    images=img,
                    return_tensors="pt"
                )
                item_data = {k: v.squeeze(0) for k, v in inputs.items()}

                # Tokenize output target as labels
                tok = self.tokenizer or getattr(self.processor, "tokenizer", None)
                if tok is not None:
                    labels = tok(
                        target,
                        max_length=self.max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids.squeeze(0)
                    # Mask padding tokens for CrossEntropyLoss
                    pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else -100
                    labels[labels == pad_token_id] = -100
                    item_data["labels"] = labels
                else:
                    item_data["labels"] = torch.zeros(self.max_length, dtype=torch.long)

                item_data["raw_prompt"] = prompt
                item_data["raw_target"] = target
                item_data["type"] = item.get("type", "unknown")
                return item_data
            except Exception:
                pass

        # Fallback raw dictionary
        return {
            "image": img,
            "prompt": prompt,
            "target": target,
            "type": item.get("type", "unknown"),
            "category": item.get("category", ""),
        }
