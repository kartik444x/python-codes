"""
app/api/v1/router.py

Aggregates all API v1 endpoints under /api/v1 prefix.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes import admin, audio, health, jobs, tts, voices

api_v1_router = APIRouter(prefix="/api/v1")

# Mount route modules
api_v1_router.include_router(health.router)
api_v1_router.include_router(voices.router)
api_v1_router.include_router(tts.router)
api_v1_router.include_router(jobs.router)
api_v1_router.include_router(audio.router)
api_v1_router.include_router(admin.router)
