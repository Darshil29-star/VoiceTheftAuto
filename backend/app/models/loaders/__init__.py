"""Loader registry: backend key (from registry.yaml) -> function(spec, source, device) -> LoadedModel.

Each loader imports its heavy dependencies lazily so the API can start without them."""

from __future__ import annotations

from typing import Callable

from app.models.loaders.hf import (
    load_florence2,
    load_grounding_dino,
    load_remoteclip,
    load_sam,
    load_segformer,
    load_smolvlm,
)

LOADERS: dict[str, Callable] = {
    "florence2": load_florence2,
    "grounding_dino": load_grounding_dino,
    "sam": load_sam,
    "segformer": load_segformer,
    "smolvlm": load_smolvlm,
    "remoteclip": load_remoteclip,
}
