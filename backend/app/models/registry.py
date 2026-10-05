"""Model registry: the single source of truth for which model serves which task."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.errors import ModelUnavailable
from app.schemas.common import ModelRef


class ModelSpec(BaseModel):
    id: str
    name: str
    version: str
    backend: str
    kind: Literal["ml", "algorithm"] = "ml"
    repo_id: str | None = None
    local_dir: str | None = None
    allow_patterns: list[str] | None = None
    license: str | None = None
    size_mb: float | None = None
    tasks: list[str] = Field(default_factory=list)
    input_type: str = "rgb"
    supported_formats: list[str] = Field(default_factory=list)
    provides_scores: bool = False
    device: str = "auto"
    dtype: str = "float32"
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    def path(self, checkpoint_dir: Path) -> Path | None:
        return checkpoint_dir / self.local_dir if self.local_dir else None

    def weights_present(self, checkpoint_dir: Path) -> bool:
        if self.kind == "algorithm":
            return True
        p = self.path(checkpoint_dir)
        if p is None or not p.exists():
            return False
        for f in p.rglob("*"):
            if f.is_file() and f.suffix in {".safetensors", ".bin", ".pt", ".pth"}:
                if f.stat().st_size > 1024:  # Git LFS pointers are ~130 bytes
                    return True
        return False

    def ref(self) -> ModelRef:
        return ModelRef(
            id=self.id,
            name=self.name,
            version=self.version,
            backend=self.backend,
            kind="algorithm" if self.kind == "algorithm" else "ml",
        )


class ModelRegistry:
    def __init__(self, models: dict[str, ModelSpec], tasks: dict[str, dict[str, Any]], source: Path | None = None):
        self.models = models
        self.tasks = tasks
        self.source = source

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelRegistry":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        models = {mid: ModelSpec(id=mid, **cfg) for mid, cfg in (raw.get("models") or {}).items()}
        tasks = raw.get("tasks") or {}
        reg = cls(models, tasks, path)
        reg._check()
        return reg

    def _check(self) -> None:
        for task, route in self.tasks.items():
            for key, value in route.items():
                ids = value if isinstance(value, list) else [value]
                for mid in ids:
                    if isinstance(mid, str) and mid in {"grounding"}:
                        continue  # routes to another specialist, not a model id
                    if isinstance(mid, str) and mid not in self.models:
                        raise ValueError(f"registry: tasks.{task}.{key} references unknown model '{mid}'")

    # ------------------------------------------------------------------ lookups
    def get(self, model_id: str) -> ModelSpec:
        try:
            return self.models[model_id]
        except KeyError as exc:
            raise ModelUnavailable(f"Model '{model_id}' is not in the registry.") from exc

    def route(self, task: str) -> dict[str, Any]:
        return self.tasks.get(task, {})

    def model_for(self, task: str, role: str = "primary") -> ModelSpec | None:
        mid = self.route(task).get(role)
        if not mid or not isinstance(mid, str) or mid not in self.models:
            return None
        spec = self.models[mid]
        return spec if spec.enabled else None

    def alternates_for(self, task: str) -> list[ModelSpec]:
        return [self.models[m] for m in self.route(task).get("alternates", []) if self.models[m].enabled]


@lru_cache
def get_registry() -> ModelRegistry:
    return ModelRegistry.from_yaml(get_settings().registry_path)
