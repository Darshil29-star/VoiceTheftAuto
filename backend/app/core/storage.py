"""Runtime upload storage. Uploads live under UPLOAD_DIR/<upload_id>/ and are never mixed with
training data. Old uploads are purged after KEEP_UPLOADS_HOURS."""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from app.core.config import Settings, get_settings
from app.core.errors import NotFound, OversizedImage, UnsupportedFormat
from app.geospatial.raster import SUPPORTED_EXTENSIONS

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_ID = re.compile(r"^[a-f0-9]{32}$")


@dataclass
class StoredFile:
    path: Path
    original_name: str
    role: str | None


def _safe_name(name: str) -> str:
    base = Path(name or "image").name
    cleaned = _SAFE.sub("_", base).strip("._") or "image"
    return cleaned[:120]


async def save_uploads(files: list[UploadFile], roles: list[str | None], settings: Settings | None = None) -> tuple[str, list[StoredFile]]:
    settings = settings or get_settings()
    upload_id = uuid.uuid4().hex
    folder = settings.upload_dir / upload_id
    folder.mkdir(parents=True, exist_ok=True)
    limit = settings.max_upload_mb * 1024 * 1024
    stored: list[StoredFile] = []
    try:
        for i, f in enumerate(files):
            name = _safe_name(f.filename or f"image_{i}")
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise UnsupportedFormat(
                    f"'{f.filename}': unsupported file type. Supported: GeoTIFF/TIFF, PNG, JPEG.",
                    details={"filename": f.filename},
                )
            dest = folder / f"{i}_{name}"
            written = 0
            with dest.open("wb") as out:
                while chunk := await f.read(1024 * 1024):
                    written += len(chunk)
                    if written > limit:
                        raise OversizedImage(
                            f"'{f.filename}' exceeds the {settings.max_upload_mb} MB limit.",
                            details={"filename": f.filename},
                        )
                    out.write(chunk)
            stored.append(StoredFile(dest, f.filename or name, roles[i] if i < len(roles) else None))
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    (folder / "meta.json").write_text(
        json.dumps([{"file": s.path.name, "original_name": s.original_name, "role": s.role} for s in stored]),
        encoding="utf-8",
    )
    return upload_id, stored


def load_upload(upload_id: str, settings: Settings | None = None) -> list[StoredFile]:
    settings = settings or get_settings()
    if not _ID.match(upload_id or ""):
        raise NotFound("Unknown upload id.")
    folder = settings.upload_dir / upload_id
    meta = folder / "meta.json"
    if not meta.exists():
        raise NotFound("Upload not found or expired.")
    entries = json.loads(meta.read_text(encoding="utf-8"))
    return [StoredFile(folder / e["file"], e["original_name"], e.get("role")) for e in entries]


def purge_old_uploads(settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    cutoff = time.time() - settings.keep_uploads_hours * 3600
    removed = 0
    for root in (settings.upload_dir, settings.processed_dir):
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
    return removed
