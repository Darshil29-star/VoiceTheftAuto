"""Input validation: file checks, raster metadata, modality roles, and task-specific
compatibility (image count, optical/SAR pairing, spatial + temporal compatibility)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from rasterio.warp import transform_bounds
except ImportError:
    transform_bounds = None

from app.core.config import Settings, get_settings
from app.core.errors import (
    IncompatibleImages,
    InsufficientImages,
    MissingImage,
    OversizedImage,
    UnsupportedFormat,
    ValidationFailed,
)
from app.geospatial.raster import SAR_BAND_NAMES, SUPPORTED_EXTENSIONS, RasterMeta, inspect_raster
from app.schemas.common import ImageInfo, ImageRole, TaskType, ValidationReport

MAX_INPUT_PIXELS = 400_000_000
_SAR_NAME = re.compile(r"(^|[^a-z])(sar|s1|sentinel-?1|vv|vh|hh|hv|grd|slc)([^a-z]|$)", re.I)
_BEFORE_NAME = re.compile(r"(^|[^a-z])(before|pre|t1|t0|old)([^a-z0-9]|$)", re.I)
_AFTER_NAME = re.compile(r"(^|[^a-z])(after|post|t2|new)([^a-z0-9]|$)", re.I)


@dataclass
class ValidatedImage:
    path: Path
    info: ImageInfo
    meta: RasterMeta


# --------------------------------------------------------------------------- per-file
def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    v = value.strip().replace("Z", "+00:00")
    for fmt in (None, "%Y:%m:%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.fromisoformat(v) if fmt is None else datetime.strptime(v, fmt)
        except ValueError:
            continue
    return None


def _infer_role(path: Path, meta: RasterMeta, user_role: ImageRole | None) -> tuple[ImageRole, str, list[str]]:
    warnings: list[str] = []
    if user_role and user_role != ImageRole.UNKNOWN:
        return user_role, "user", warnings

    tag_role = (meta.tags.get("IMAGE_ROLE") or "").lower()
    sensor = (meta.tags.get("SENSOR") or "").lower()
    if tag_role in {r.value for r in ImageRole} and tag_role != "unknown":
        return ImageRole(tag_role), "metadata", warnings
    if "sentinel-1" in sensor or "sar" in sensor:
        return ImageRole.SAR, "metadata", warnings
    descs = {(d or "").upper() for d in meta.band_descriptions}
    if descs & SAR_BAND_NAMES:
        return ImageRole.SAR, "metadata", warnings

    if _SAR_NAME.search(path.stem):
        warnings.append("Modality inferred as SAR from the file name; tag the role explicitly if this is wrong.")
        return ImageRole.SAR, "heuristic", warnings
    if meta.bands in (1, 2) and meta.dtype.startswith("float"):
        warnings.append("1–2 float bands suggest SAR backscatter; tag the role explicitly to confirm.")
        return ImageRole.SAR, "heuristic", warnings
    if meta.bands >= 3:
        return ImageRole.OPTICAL, "heuristic", warnings
    warnings.append("Single-band image: modality could not be determined (optical panchromatic or SAR).")
    return ImageRole.UNKNOWN, "default", warnings


def inspect_file(path: Path, index: int, display_name: str | None = None, user_role: ImageRole | None = None,
                 settings: Settings | None = None) -> ValidatedImage:
    settings = settings or get_settings()
    name = display_name or path.name
    if not path.exists():
        raise MissingImage(f"Image '{name}' was not found.")
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormat(
            f"'{name}': unsupported file type '{ext or 'none'}'. Supported: GeoTIFF/TIFF, PNG, JPEG.",
            details={"filename": name},
        )
    size = path.stat().st_size
    if size == 0:
        raise UnsupportedFormat(f"'{name}' is empty.", details={"filename": name})
    if size > settings.max_upload_mb * 1024 * 1024:
        raise OversizedImage(
            f"'{name}' is {size / 1e6:.0f} MB; the limit is {settings.max_upload_mb} MB.",
            details={"filename": name, "size_bytes": size},
        )

    meta = inspect_raster(path)
    if meta.width * meta.height > MAX_INPUT_PIXELS:
        raise OversizedImage(f"'{name}' has {meta.width}x{meta.height} pixels, above the supported maximum.")
    if meta.width < 32 or meta.height < 32:
        raise ValidationFailed(f"'{name}' is only {meta.width}x{meta.height} px; at least 32x32 is required.")

    role, role_source, warnings = _infer_role(Path(name), meta, user_role)
    if max(meta.width, meta.height) > settings.max_image_side:
        warnings.append(
            f"Image is {meta.width}x{meta.height}; analysis runs at max {settings.max_image_side} px "
            "and results are mapped back to original coordinates."
        )
    if not meta.is_georeferenced:
        warnings.append("No georeferencing (CRS/transform): results are in pixel coordinates only.")

    ts = _parse_time(meta.timestamp)
    info = ImageInfo(
        index=index,
        filename=name,
        format=meta.driver,
        size_bytes=size,
        width=meta.width,
        height=meta.height,
        bands=meta.bands,
        dtype=meta.dtype,
        crs=meta.crs,
        transform=meta.transform,
        bounds=meta.bounds,
        gsd_m=round(meta.gsd_m, 4) if meta.gsd_m else None,
        is_georeferenced=meta.is_georeferenced,
        timestamp=ts.isoformat() if ts else meta.timestamp,
        band_descriptions=meta.band_descriptions,
        nodata=meta.nodata,
        role=role,
        role_source=role_source,
        warnings=warnings,
    )
    return ValidatedImage(path=path, info=info, meta=meta)


# --------------------------------------------------------------------------- pairs
def spatial_compatibility(a: ValidatedImage, b: ValidatedImage) -> dict:
    """Decide whether two images cover the same ground and can be aligned."""
    ma, mb = a.meta, b.meta
    result: dict = {"georeferenced": ma.is_georeferenced and mb.is_georeferenced}

    if ma.is_georeferenced and mb.is_georeferenced:
        bb = transform_bounds(mb.crs, ma.crs, *mb.bounds) if ma.crs != mb.crs else mb.bounds
        ax0, ay0, ax1, ay1 = ma.bounds
        bx0, by0, bx1, by1 = bb
        ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        iy = max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = ix * iy
        area_a = (ax1 - ax0) * (ay1 - ay0)
        area_b = (bx1 - bx0) * (by1 - by0)
        overlap = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
        result.update(
            same_crs=ma.crs == mb.crs,
            overlap_fraction=round(overlap, 4),
            gsd_ratio=round((ma.gsd_m or 1) / (mb.gsd_m or 1), 3),
            alignment="reproject_to_first" if (ma.crs != mb.crs or ma.transform != mb.transform) else "identical_grid",
        )
        result["compatible"] = overlap >= 0.10
        if not result["compatible"]:
            result["reason"] = f"Image footprints overlap by only {overlap:.1%} (minimum 10%)."
        return result

    if ma.is_georeferenced != mb.is_georeferenced:
        result["warning"] = "Only one image is georeferenced; alignment falls back to pixel geometry."

    ar_a, ar_b = ma.width / ma.height, mb.width / mb.height
    ar_diff = abs(ar_a - ar_b) / max(ar_a, ar_b)
    result.update(
        same_shape=(ma.width, ma.height) == (mb.width, mb.height),
        aspect_ratio_diff=round(ar_diff, 4),
        alignment="pixel_grid" if (ma.width, ma.height) == (mb.width, mb.height) else "resize_to_first",
    )
    result["compatible"] = ar_diff <= 0.02
    if not result["compatible"]:
        result["reason"] = (
            f"Images are not georeferenced and their aspect ratios differ by {ar_diff:.1%}; "
            "they cannot be assumed to show the same area."
        )
    else:
        result["assumption"] = "No georeferencing: images are assumed to depict the same area (pixel grid)."
    return result


def temporal_info(a: ValidatedImage, b: ValidatedImage) -> dict:
    ta, tb = _parse_time(a.info.timestamp), _parse_time(b.info.timestamp)
    if not ta or not tb:
        return {"known": False}
    if (ta.tzinfo is None) != (tb.tzinfo is None):
        ta, tb = ta.replace(tzinfo=None), tb.replace(tzinfo=None)
    gap_h = (tb - ta).total_seconds() / 3600
    return {"known": True, "gap_hours": round(gap_h, 2), "first_is_earlier": gap_h >= 0}


# --------------------------------------------------------------------------- task rules
def validate_for_task(task: TaskType, images: list[ValidatedImage]) -> tuple[list[ValidatedImage], dict, list[str]]:
    """Check task-specific requirements. Returns images in the order the specialist expects,
    a compatibility report, and warnings. Raises typed errors when requirements are not met."""
    warnings: list[str] = []
    if not images:
        raise MissingImage("Upload at least one image to analyse.")

    if task in (TaskType.VQA, TaskType.CAPTIONING, TaskType.GROUNDING):
        if len(images) > 1:
            warnings.append(f"{task.value} uses one image; only '{images[0].info.filename}' is analysed.")
        img = images[0]
        if img.info.role == ImageRole.SAR:
            warnings.append("The selected models are trained on optical imagery; results on SAR are unreliable.")
        return [img], {}, warnings

    if len(images) < 2:
        need = "a before and an after image" if task == TaskType.CHANGE_ANALYSIS else "an optical and a SAR image"
        raise InsufficientImages(f"This task needs two images: {need}.", details={"task": task.value, "received": len(images)})
    if len(images) > 2:
        warnings.append("Only the first two images are used.")
    a, b = images[:2]

    if task == TaskType.CHANGE_ANALYSIS:
        roles = {a.info.role, b.info.role}
        if ImageRole.SAR in roles and ImageRole.OPTICAL in roles:
            raise IncompatibleImages(
                "Change analysis needs two images of the same modality; an optical and a SAR image were given. "
                "Did you mean optical–SAR fusion?",
            )
        if a.info.role == ImageRole.AFTER or b.info.role == ImageRole.BEFORE:
            a, b = b, a
            warnings.append("Images reordered using the before/after roles.")
        elif a.info.role != ImageRole.BEFORE and b.info.role != ImageRole.AFTER:
            by_name_a, by_name_b = _AFTER_NAME.search(a.info.filename), _BEFORE_NAME.search(b.info.filename)
            t = temporal_info(a, b)
            if t.get("known") and not t["first_is_earlier"]:
                a, b = b, a
                warnings.append("Images reordered by acquisition timestamp (earlier = before).")
            elif not t.get("known") and by_name_a and by_name_b:
                a, b = b, a
                warnings.append("Images reordered using file names (before/after).")
            elif not t.get("known"):
                warnings.append("No timestamps: the first image is treated as BEFORE and the second as AFTER.")
        compat = spatial_compatibility(a, b)
        compat["temporal"] = temporal_info(a, b)
        if a.meta.bands != b.meta.bands:
            warnings.append(f"Band counts differ ({a.meta.bands} vs {b.meta.bands}); comparison uses RGB renderings.")
        if not compat["compatible"]:
            raise IncompatibleImages(compat["reason"], details={"compatibility": compat})
        return [a, b], compat, warnings

    # ---- optical + SAR fusion
    roles = [a.info.role, b.info.role]
    if ImageRole.SAR not in roles or ImageRole.OPTICAL not in roles:
        if roles.count(ImageRole.SAR) == 1 and ImageRole.UNKNOWN in roles:
            other = b if a.info.role == ImageRole.SAR else a
            other.info.role, other.info.role_source = ImageRole.OPTICAL, "heuristic"
            warnings.append(f"'{other.info.filename}' assumed optical because the other image is SAR.")
        else:
            raise IncompatibleImages(
                "Optical–SAR fusion needs one optical and one SAR image. Detected roles: "
                f"{a.info.filename}={a.info.role.value}, {b.info.filename}={b.info.role.value}. "
                "Tag the image roles explicitly if detection is wrong.",
                details={"roles": [r.value for r in roles]},
            )
    optical = a if a.info.role == ImageRole.OPTICAL else b
    sar = b if optical is a else a
    compat = spatial_compatibility(optical, sar)
    compat["temporal"] = temporal_info(optical, sar)
    if not compat["compatible"]:
        raise IncompatibleImages(compat["reason"], details={"compatibility": compat})
    if not compat["georeferenced"]:
        warnings.append(
            "Neither image is georeferenced: optical–SAR co-registration cannot be verified and is assumed."
        )
    gap = compat["temporal"].get("gap_hours")
    if gap is not None and abs(gap) > 12 * 24:
        warnings.append(f"Optical and SAR acquisitions are {abs(gap) / 24:.0f} days apart; surface conditions may differ.")
    return [optical, sar], compat, warnings


def build_report(images: list[ValidatedImage], errors: list[dict] | None = None, compat: dict | None = None,
                 warnings: list[str] | None = None) -> ValidationReport:
    all_warnings = list(warnings or [])
    return ValidationReport(
        valid=not errors,
        images=[i.info for i in images],
        errors=errors or [],
        warnings=all_warnings,
        compatibility=compat or {},
    )
