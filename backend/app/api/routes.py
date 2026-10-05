"""HTTP routes. Thin: all logic lives in validators / controllers / specialists."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, File, Form, UploadFile

from app import __version__
from app.core.config import get_settings
from app.core.device import resolve_device
from app.core.errors import GeoSenseError, MissingImage, ValidationFailed
from app.core.storage import StoredFile, load_upload, save_uploads
from app.models.manager import get_model_manager
from app.models.registry import get_registry
from app.schemas.api import AnalyzeResponse, HealthResponse, ModelsResponse, ModelStatus, UploadResponse, ValidateResponse
from app.schemas.common import ImageRole, TaskType
from app.validators.image_validator import ValidatedImage, inspect_file, validate_for_task

router = APIRouter()


def _parse_role(value: str | None) -> ImageRole | None:
    if not value:
        return None
    try:
        return ImageRole(value.strip().lower())
    except ValueError as exc:
        raise ValidationFailed(f"Unknown image role '{value}'. Use one of: {[r.value for r in ImageRole]}.") from exc


def _roles_list(roles: list[str] | None) -> list[str | None]:
    """Accept repeated form fields or a single comma-separated value."""
    if not roles:
        return []
    if len(roles) == 1 and "," in roles[0]:
        roles = roles[0].split(",")
    return [r.strip() or None for r in roles]


def inspect_stored(stored: list[StoredFile]) -> list[ValidatedImage]:
    return [inspect_file(s.path, i, display_name=s.original_name, user_role=_parse_role(s.role)) for i, s in enumerate(stored)]


async def resolve_inputs(files: list[UploadFile] | None, upload_id: str | None, roles: list[str] | None) -> tuple[str, list[StoredFile]]:
    role_values = _roles_list(roles)
    files = [f for f in (files or []) if f is not None and (f.filename or "")]
    if files:
        return await save_uploads(files, role_values)
    if upload_id:
        stored = load_upload(upload_id)
        for i, r in enumerate(role_values):  # roles supplied now override roles stored at upload
            if i < len(stored) and r:
                stored[i].role = r
        return upload_id, stored
    raise MissingImage("Upload at least one image (files) or reference a previous upload (upload_id).")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    registry = get_registry()
    manager = get_model_manager()
    dev = resolve_device(settings.device, settings.torch_threads)
    ml = [m for m in registry.models.values() if m.kind == "ml" and m.enabled]
    available = [m for m in ml if manager.is_available(m)]
    return HealthResponse(
        status="ok" if dev.torch_available else "degraded",
        version=__version__,
        device=dev.resolved,
        device_reason=dev.reason,
        demo_mode=settings.demo_mode,
        models_available=len(available),
        models_enabled=len(ml),
        loaded_models=manager.loaded_ids(),
    )


@router.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    settings = get_settings()
    registry = get_registry()
    manager = get_model_manager()
    loaded = set(manager.loaded_ids())
    statuses = []
    for spec in registry.models.values():
        available = manager.is_available(spec)
        note = None
        if spec.kind == "ml" and not available:
            note = manager.last_failure(spec.id) or "weights not downloaded"
        statuses.append(
            ModelStatus(
                id=spec.id,
                name=spec.name,
                version=spec.version,
                backend=spec.backend,
                kind=spec.kind,
                tasks=spec.tasks,
                enabled=spec.enabled,
                available=available,
                loaded=spec.id in loaded,
                license=spec.license,
                size_mb=spec.size_mb,
                provides_scores=spec.provides_scores,
                note=note,
            )
        )
    return ModelsResponse(
        device=asdict(resolve_device(settings.device, settings.torch_threads)),
        max_loaded_models=settings.max_loaded_models,
        task_routing=registry.tasks,
        models=statuses,
    )


@router.post("/upload", response_model=UploadResponse)
async def upload(
    files: list[UploadFile] = File(..., description="1-2 images: GeoTIFF/TIFF, PNG or JPEG"),
    roles: list[str] | None = Form(None, description="optional per-file roles: optical, sar, before, after"),
) -> UploadResponse:
    upload_id, stored = await resolve_inputs(files, None, roles)
    images = inspect_stored(stored)
    warnings = [f"{i.info.filename}: {w}" for i in images for w in i.info.warnings]
    return UploadResponse(upload_id=upload_id, images=[i.info for i in images], warnings=warnings)


@router.post("/validate", response_model=ValidateResponse)
async def validate(
    files: list[UploadFile] | None = File(None),
    upload_id: str | None = Form(None),
    roles: list[str] | None = Form(None),
    task: TaskType | None = Form(None, description="if given, also check task-specific requirements"),
) -> ValidateResponse:
    """Always returns a report (valid=false with errors) instead of failing, so the UI can explain problems."""
    images: list[ValidatedImage] = []
    errors: list[dict] = []
    warnings: list[str] = []
    compat: dict = {}
    try:
        _, stored = await resolve_inputs(files, upload_id, roles)
        images = inspect_stored(stored)
        warnings = [f"{i.info.filename}: {w}" for i in images for w in i.info.warnings]
        if task is not None:
            _, compat, task_warnings = validate_for_task(task, images)
            warnings += task_warnings
    except GeoSenseError as exc:
        errors.append(exc.to_dict())
        compat = exc.details.get("compatibility", compat)
    return ValidateResponse(valid=not errors, images=[i.info for i in images], errors=errors, warnings=warnings, compatibility=compat)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    query: str = Form(..., description="natural-language question or instruction"),
    files: list[UploadFile] | None = File(None),
    upload_id: str | None = Form(None),
    roles: list[str] | None = Form(None, description="optional per-file roles: optical, sar, before, after"),
    task: TaskType | None = Form(None, description="optional explicit task (skips classification)"),
):
    """Main endpoint. Returns the full AnalyzeResponse (with execution trace) even on errors."""
    from fastapi.responses import JSONResponse

    from app.controllers.agent_controller import AgentController

    stored, input_error = [], None
    try:
        _, stored = await resolve_inputs(files, upload_id, roles)
    except GeoSenseError as exc:
        input_error = exc  # reported by the controller inside a full response with trace
    status_code, response = await AgentController().analyze(query, stored, task_override=task, input_error=input_error)
    return JSONResponse(status_code=status_code, content=response.model_dump(mode="json"))
