"""Preprocessing utilities for Remote Sensing image captioning."""

from __future__ import annotations

from typing import Any
import numpy as np
from PIL import Image


def preprocess_image(image_input: str | np.ndarray | Image.Image, target_size: tuple[int, int] = (384, 384)) -> Image.Image:
    """Load and normalize remote sensing image to RGB PIL format."""
    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    else:
        img = image_input

    if img.mode != "RGB":
        img = img.convert("RGB")
    
    if target_size:
        img = img.resize(target_size, Image.Resampling.BILINEAR)
    return img


def format_florence_caption_batch(
    processor: Any,
    images: list[Image.Image],
    captions: list[str],
    task_prefix: str = "<DETAILED_CAPTION>",
    max_length: int = 96,
) -> dict[str, Any]:
    """Tokenize and format image-caption pairs for Florence-2 training."""
    prompts = [task_prefix] * len(images)
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    
    # Target tokens (captions)
    targets = processor.tokenizer(
        captions,
        return_tensors="pt",
        padding=True,
        max_length=max_length,
        truncation=True,
    )
    labels = targets.input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs
