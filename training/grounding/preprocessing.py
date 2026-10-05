"""Preprocessing utilities for Remote Sensing visual grounding datasets."""

from __future__ import annotations

from typing import Any
import numpy as np
from PIL import Image


def normalize_bbox(bbox: list[float], orig_w: int, orig_h: int) -> list[float]:
    """Convert [x1, y1, x2, y2] pixel coordinates to normalized [0, 1000] scale for Florence-2."""
    x1, y1, x2, y2 = bbox
    return [
        round((x1 / orig_w) * 1000, 1),
        round((y1 / orig_h) * 1000, 1),
        round((x2 / orig_w) * 1000, 1),
        round((y2 / orig_h) * 1000, 1),
    ]


def denormalize_bbox(bbox: list[float], orig_w: int, orig_h: int) -> list[int]:
    """Convert normalized [0, 1000] bounding boxes back to pixel coordinates."""
    x1, y1, x2, y2 = bbox
    return [
        int(round((x1 / 1000.0) * orig_w)),
        int(round((y1 / 1000.0) * orig_h)),
        int(round((x2 / 1000.0) * orig_w)),
        int(round((y2 / 1000.0) * orig_h)),
    ]


def compute_iou(boxA: list[float], boxB: list[float]) -> float:
    """Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area
