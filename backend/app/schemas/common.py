"""Shared schemas. Every specialist speaks these types, so the frontend never needs to know
which specialist produced a piece of evidence."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    VQA = "vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    CHANGE_ANALYSIS = "change_analysis"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"


class ImageRole(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"
    BEFORE = "before"
    AFTER = "after"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- confidence
class ConfidenceKind(str, Enum):
    MODEL = "model"          # a score emitted by the model itself (may be uncalibrated)
    ESTIMATED = "estimated"  # derived by a documented heuristic from outputs/statistics
    SYSTEM = "system"        # combination of the above plus input-quality factors
    UNAVAILABLE = "unavailable"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNAVAILABLE = "unavailable"


class Confidence(BaseModel):
    value: float | None = Field(None, ge=0.0, le=1.0)
    kind: ConfidenceKind = ConfidenceKind.UNAVAILABLE
    level: ConfidenceLevel = ConfidenceLevel.UNAVAILABLE
    calibrated: bool = False
    source: str = ""
    explanation: str = "Confidence unavailable from model"
    factors: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- evidence
class EvidenceType(str, Enum):
    TEXT = "text"
    SPATIAL_REGION = "spatial_region"  # bbox and/or polygon (+ optional mask)
    MASK = "mask"
    CHANGE_MAP = "change_map"
    IMAGE_CROP = "image_crop"
    METADATA = "metadata"
    MODEL_OUTPUT = "model_output"
    STATISTIC = "statistic"


class GeoRef(BaseModel):
    crs: str
    bbox: list[float] | None = None                 # [minx, miny, maxx, maxy] in crs units
    polygon: list[list[float]] | None = None        # [[x, y], ...] in crs units
    area_m2: float | None = None


class Evidence(BaseModel):
    id: str
    type: EvidenceType
    label: str
    source: str                                      # specialist / model id that produced it
    method: Literal["model", "algorithm", "rule", "fallback"] = "model"
    image_index: int | None = None
    bbox: list[float] | None = None                  # [x0, y0, x1, y1] pixels, ORIGINAL image space
    polygon: list[list[float]] | None = None         # [[x, y], ...] pixels, ORIGINAL image space
    area_px: float | None = None
    geo: GeoRef | None = None
    visualization_id: str | None = None              # e.g. mask layer
    value: Any = None
    description: str | None = None
    confidence: Confidence | None = None


class Visualization(BaseModel):
    id: str
    kind: Literal["base", "overlay", "mask", "change_map", "fusion", "comparison", "index"]
    title: str
    url: str
    image_index: int | None = None
    width: int
    height: int
    default_visible: bool = True
    group: str = "main"                              # frontend groups panels by this
    legend: dict[str, str] | None = None             # label -> css colour


# --------------------------------------------------------------------------- trace
class TraceStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"
    FALLBACK = "fallback"


class TraceStep(BaseModel):
    stage: str
    status: TraceStatus = TraceStatus.OK
    message: str
    duration_ms: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ModelRef(BaseModel):
    id: str
    name: str
    version: str
    backend: str
    kind: Literal["ml", "algorithm", "fallback"] = "ml"


# --------------------------------------------------------------------------- images
class ImageInfo(BaseModel):
    index: int
    filename: str
    format: str                                      # GTiff / PNG / JPEG
    size_bytes: int
    width: int
    height: int
    bands: int
    dtype: str
    crs: str | None = None
    transform: list[float] | None = None             # affine a,b,c,d,e,f
    bounds: list[float] | None = None
    gsd_m: float | None = None
    is_georeferenced: bool = False
    timestamp: str | None = None
    band_descriptions: list[str | None] = Field(default_factory=list)
    nodata: float | None = None
    role: ImageRole = ImageRole.UNKNOWN
    role_source: Literal["user", "metadata", "heuristic", "default"] = "default"
    warnings: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    valid: bool
    images: list[ImageInfo]
    errors: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    compatibility: dict[str, Any] = Field(default_factory=dict)
