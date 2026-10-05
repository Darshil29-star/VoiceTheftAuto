"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.core.config import get_settings
from app.core.errors import GeoSenseError
from app.core.logging import configure_logging

log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Multimodal remote-sensing analysis: VQA, captioning + grounding, change analysis, optical-SAR fusion.",
        redirect_slashes=False,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(GeoSenseError)
    async def _geosense_error(_: Request, exc: GeoSenseError):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.to_dict()})

    @app.exception_handler(RequestValidationError)
    async def _request_validation(_: Request, exc: RequestValidationError):
        details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "bad_request", "message": "Invalid request.", "details": {"errors": details}}},
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("unhandled_error")  # full trace stays in server logs only
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "An internal error occurred.", "details": {}}},
        )

    from app.api.routes import router

    app.include_router(router)
    app.include_router(router, prefix="/api")
    app.mount("/files", StaticFiles(directory=str(settings.processed_dir), check_dir=False), name="files")

    import os
    frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))
    if not os.path.isdir(frontend_dist):
        frontend_dist = os.path.abspath("frontend/dist")
    if os.path.isdir(frontend_dist):
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    return app


app = create_app()
