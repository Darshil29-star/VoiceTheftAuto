"""Common specialist interface. Every specialist consumes an AnalysisContext and returns a
SpecialistOutput built from the shared schemas, so the controller, evidence layer and frontend
are independent of which model produced a result."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.trace import Trace
from app.geospatial.raster import RasterData, read_raster
from app.models.manager import ModelManager
from app.models.registry import ModelRegistry, ModelSpec
from app.schemas.common import Confidence, Evidence, ModelRef, TaskType, Visualization
from app.validators.image_validator import ValidatedImage


@dataclass
class AnalysisContext:
    request_id: str
    query: str
    task: TaskType
    images: list[ValidatedImage]
    settings: Settings
    registry: ModelRegistry
    manager: ModelManager
    trace: Trace
    output_dir: Path
    url_prefix: str
    compatibility: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    user_gsd_m: float | None = None                          # user-supplied metadata for non-georeferenced imagery
    cache: dict[str, Any] = field(default_factory=dict)      # per-request results shared between sub-steps
    _rasters: dict[int, RasterData] = field(default_factory=dict)

    def raster(self, i: int = 0) -> RasterData:
        """Lazily read (and cache) image i at analysis resolution."""
        if i not in self._rasters:
            img = self.images[i]
            self._rasters[i] = read_raster(img.path, self.settings.max_image_side, img.meta)
        return self._rasters[i]

    def gsd(self, i: int = 0) -> tuple[float | None, str]:
        """Ground sample distance of the ORIGINAL image and where it came from."""
        meta = self.images[i].meta
        if meta.gsd_m:
            return meta.gsd_m, "georeferencing"
        if self.user_gsd_m:
            return self.user_gsd_m, "user metadata"
        if self.settings.assumed_gsd_m:
            return self.settings.assumed_gsd_m, "assumed (ASSUMED_GSD_M)"
        return None, "unknown"

    def model_spec(self, role: str = "primary", task: str | None = None) -> ModelSpec | None:
        return self.registry.model_for(task or self.task.value, role)


@dataclass
class SpecialistOutput:
    answer: str
    confidence: Confidence
    model: ModelRef
    evidence: list[Evidence] = field(default_factory=list)
    visualizations: list[Visualization] = field(default_factory=list)
    models_used: list[ModelRef] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    fallback_used: bool = False


class Specialist(ABC):
    task: TaskType

    @abstractmethod
    def run(self, ctx: AnalysisContext) -> SpecialistOutput:
        """Genuine analysis. Raise ModelUnavailable if a required model cannot be loaded."""

    def fallback(self, ctx: AnalysisContext) -> SpecialistOutput | None:
        """DEMO_MODE only. Must never imitate model output; return None if no honest fallback exists."""
        return None
