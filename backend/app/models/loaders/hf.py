"""Hugging Face / open_clip loaders. All CPU-compatible; CUDA is used only if resolved."""

from __future__ import annotations

from pathlib import Path

from app.models.manager import LoadedModel
from app.models.registry import ModelSpec


def _dtype(spec: ModelSpec, device: str):
    import torch

    if device == "cuda" and spec.dtype == "float16":
        return torch.float16
    return torch.float32  # CPU inference stays in float32 for numerical reliability


def load_florence2(spec: ModelSpec, source: str, device: str) -> LoadedModel:
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    dtype = _dtype(spec, device)
    model = Florence2ForConditionalGeneration.from_pretrained(source, dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(source)
    return LoadedModel(spec=spec, device=device, model=model, processor=processor, extras={"dtype": str(dtype)})


def load_grounding_dino(spec: ModelSpec, source: str, device: str) -> LoadedModel:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model = AutoModelForZeroShotObjectDetection.from_pretrained(source).to(device).eval()
    processor = AutoProcessor.from_pretrained(source)
    return LoadedModel(spec=spec, device=device, model=model, processor=processor)


def load_sam(spec: ModelSpec, source: str, device: str) -> LoadedModel:
    from transformers import SamModel, SamProcessor

    model = SamModel.from_pretrained(source).to(device).eval()
    processor = SamProcessor.from_pretrained(source)
    return LoadedModel(spec=spec, device=device, model=model, processor=processor)


def load_smolvlm(spec: ModelSpec, source: str, device: str) -> LoadedModel:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(source, dtype=_dtype(spec, device)).to(device).eval()
    processor = AutoProcessor.from_pretrained(source)
    return LoadedModel(spec=spec, device=device, model=model, processor=processor)


def load_remoteclip(spec: ModelSpec, source: str, device: str) -> LoadedModel:
    import open_clip
    import torch

    arch = spec.params.get("arch", "ViT-B-32")
    src = Path(source)
    ckpt = next(src.rglob(f"RemoteCLIP-{arch}.pt")) if src.is_dir() else src
    model, _, preprocess = open_clip.create_model_and_transforms(arch)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(arch)
    return LoadedModel(spec=spec, device=device, model=model, processor=preprocess, extras={"tokenizer": tokenizer})


def load_segformer(spec: ModelSpec, source: str, device: str) -> LoadedModel:
    # SegformerImageProcessor is used directly: older checkpoints declare the removed
    # "SegformerFeatureExtractor" type, which AutoImageProcessor no longer resolves.
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    model = SegformerForSemanticSegmentation.from_pretrained(source).to(device).eval()
    processor = SegformerImageProcessor.from_pretrained(source)
    return LoadedModel(spec=spec, device=device, model=model, processor=processor,
                       extras={"id2label": dict(model.config.id2label)})
