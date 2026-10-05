"""Typed, user-safe errors. Raw stack traces never reach API clients."""

from __future__ import annotations

from typing import Any


class GeoSenseError(Exception):
    code = "internal_error"
    status_code = 500
    user_message = "An internal error occurred."

    def __init__(self, message: str | None = None, *, details: dict[str, Any] | None = None):
        super().__init__(message or self.user_message)
        self.message = message or self.user_message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ValidationFailed(GeoSenseError):
    code, status_code, user_message = "validation_failed", 422, "Input validation failed."


class MissingImage(ValidationFailed):
    code, user_message = "missing_image", "At least one image is required."


class UnsupportedFormat(ValidationFailed):
    code, status_code, user_message = "unsupported_format", 415, "Unsupported image format."


class OversizedImage(ValidationFailed):
    code, status_code, user_message = "oversized_image", 413, "Image exceeds the configured size limit."


class InsufficientImages(ValidationFailed):
    code, user_message = "insufficient_images", "This task needs more images."


class IncompatibleImages(ValidationFailed):
    code, user_message = "incompatible_images", "The provided images are not compatible for this task."


class UnsupportedQuery(GeoSenseError):
    code, status_code, user_message = "unsupported_query", 400, "The query could not be mapped to a supported task."


class ClarificationNeeded(GeoSenseError):
    code, status_code, user_message = "clarification_needed", 200, "The query is ambiguous."


class ModelUnavailable(GeoSenseError):
    code, status_code, user_message = "model_unavailable", 503, "The required model is not available."


class InferenceFailed(GeoSenseError):
    code, status_code, user_message = "inference_failed", 500, "Model inference failed."


class OutOfMemory(InferenceFailed):
    code, status_code, user_message = "out_of_memory", 507, "Not enough memory to run this analysis. Try a smaller image."


class InferenceTimeout(InferenceFailed):
    code, status_code, user_message = "timeout", 504, "Analysis timed out."


class NotFound(GeoSenseError):
    code, status_code, user_message = "not_found", 404, "Resource not found."
