"""Central configuration. Every path and limit comes from environment variables (or .env)."""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# geosense-ai/ (repo root) — used only to resolve relative defaults
REPO_ROOT = Path(__file__).resolve().parents[3]
_IS_VERCEL = bool(os.environ.get("VERCEL"))
_TMP_DATA = Path(tempfile.gettempdir()) / "geosense_data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "GeoSense AI"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"

    # --- compute ---------------------------------------------------------
    device: Literal["auto", "cpu", "cuda"] = "auto"
    torch_threads: int = Field(0, description="0 = let PyTorch decide")
    max_loaded_models: int = Field(3, ge=1, description="LRU bound on models held in RAM")
    allow_remote_model_download: bool = Field(
        False, description="If false, models must already exist under CHECKPOINT_DIR"
    )
    inference_timeout_s: float = 180.0

    # --- behaviour -------------------------------------------------------
    demo_mode: bool = Field(
        True if _IS_VERCEL else False, description="Allow clearly-labelled fallbacks when a model cannot be loaded"
    )
    max_upload_mb: int = 200
    max_images_per_request: int = 2
    max_image_side: int = Field(2048, description="Rasters are downsampled to this for analysis")
    classifier_clarify_margin: float = 0.15
    assumed_gsd_m: float | None = Field(
        0.5, description="GSD assumed for non-georeferenced PNG/JPEG when the user gives none (recorded in trace)"
    )

    # --- paths -----------------------------------------------------------
    data_dir: Path = _TMP_DATA if _IS_VERCEL else (REPO_ROOT / "data")
    upload_dir: Path = (_TMP_DATA / "uploads") if _IS_VERCEL else (REPO_ROOT / "data" / "uploads")
    processed_dir: Path = (_TMP_DATA / "processed") if _IS_VERCEL else (REPO_ROOT / "data" / "processed")
    samples_dir: Path = REPO_ROOT / "data" / "samples"
    checkpoint_dir: Path = (_TMP_DATA / "checkpoints") if _IS_VERCEL else (REPO_ROOT / "models" / "checkpoints")
    registry_path: Path = REPO_ROOT / "models" / "registry.yaml"

    # --- api -------------------------------------------------------------
    cors_origins: list[str] = ["*"]
    keep_uploads_hours: int = 24

    @field_validator(
        "data_dir", "upload_dir", "processed_dir", "samples_dir", "checkpoint_dir", "registry_path",
        mode="after",
    )
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        return v if v.is_absolute() else (REPO_ROOT / v).resolve()

    def ensure_dirs(self) -> None:
        try:
            for p in (self.upload_dir, self.processed_dir, self.checkpoint_dir):
                p.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


@lru_cache
def get_settings() -> Settings:
    return Settings()
