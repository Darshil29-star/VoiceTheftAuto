"""Vercel Serverless Function entry point for GeoSense AI Backend."""

import os
import sys

# Set environment flags for Vercel execution
os.environ["VERCEL"] = "1"
os.environ["DEMO_MODE"] = "true"

# Ensure backend directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
backend_dir = os.path.join(repo_root, "backend")

if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi import Request
from app.main import app


@app.middleware("http")
async def vercel_path_rewrite(request: Request, call_next):
    """In Vercel serverless with rewrites to /api/index.py,

    recover original client path from query param or headers so FastAPI routes match.
    """
    route = (
        request.query_params.get("__route")
        or request.headers.get("x-matched-path")
        or request.headers.get("x-vercel-matched-path")
        or request.headers.get("x-forwarded-uri")
    )
    if route:
        clean_path = route.split("?")[0]
        if not clean_path.endswith(".py"):
            request.scope["path"] = clean_path

    return await call_next(request)
