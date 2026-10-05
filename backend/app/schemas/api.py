"""API request / response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.common import (
    Confidence,
    Evidence,
    ImageInfo,
    ModelRef,
    TaskType,
    TraceStep,
    ValidationReport,
    Visualization,
)


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ClassificationInfo(BaseModel):
    task: TaskType | None
    scores: dict[str, float]
    rationale: list[str]
    decided_by: Literal["query", "query+inputs", "override", "clarification"]
    ambiguous: bool = False


class Clarification(BaseModel):
    question: str
    options: list[TaskType]


class AnalyzeResponse(BaseModel):
    request_id: str
    status: Literal["ok", "clarification_needed", "error"]
    task: TaskType | None = None
    query: str
    answer: str | None = None
    confidence: Confidence | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    visualizations: list[Visualization] = Field(default_factory=list)
    model: ModelRef | None = None
    models_used: list[ModelRef] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)   # task-specific structured output
    images: list[ImageInfo] = Field(default_factory=list)
    classification: ClassificationInfo | None = None
    clarification: Clarification | None = None
    warnings: list[str] = Field(default_factory=list)       # input caveats + method limitations
    confidence_note: str | None = None                        # "LOW CONFIDENCE" / "Confidence unavailable from model"
    execution_trace: list[TraceStep] = Field(default_factory=list)
    processing_time: float = 0.0
    device: str = "cpu"
    demo_fallback: bool = False
    error: ErrorBody | None = None


class UploadResponse(BaseModel):
    upload_id: str
    images: list[ImageInfo]
    warnings: list[str] = Field(default_factory=list)


class ValidateResponse(ValidationReport):
    pass


class ModelStatus(BaseModel):
    id: str
    name: str
    version: str
    backend: str
    kind: str
    tasks: list[str]
    enabled: bool
    available: bool
    loaded: bool
    license: str | None = None
    size_mb: float | None = None
    provides_scores: bool = False
    note: str | None = None


class ModelsResponse(BaseModel):
    device: dict[str, Any]
    max_loaded_models: int
    task_routing: dict[str, dict[str, Any]]
    models: list[ModelStatus]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    device: str
    device_reason: str
    demo_mode: bool
    models_available: int
    models_enabled: int
    loaded_models: list[str]
