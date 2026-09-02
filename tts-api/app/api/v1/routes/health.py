"""
app/api/v1/routes/health.py

Health check and readiness endpoints for Kubernetes, load balancers, and monitoring.
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_engine_instance
from app.core.config import get_settings
from app.engines.base import BaseTTSEngine
from app.models.db.session import get_db_session
from app.models.responses import HealthResponse, LivenessResponse, ReadinessResponse

router = APIRouter(tags=["Health"])

_START_TIME = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Overview",
    description="Returns high-level health status, loaded model information, and active device.",
)
async def get_health(
    engine: BaseTTSEngine = Depends(get_engine_instance),
) -> HealthResponse:
    settings = get_settings()
    uptime = time.time() - _START_TIME
    is_loaded = engine.is_loaded

    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        device=engine.get_status().device,
        version=settings.app_version,
        uptime_seconds=round(uptime, 1),
    )


@router.get(
    "/health/live",
    response_model=LivenessResponse,
    summary="Liveness Probe",
    description="Kubernetes / container liveness check. Confirms HTTP process is running.",
)
async def get_liveness() -> LivenessResponse:
    return LivenessResponse(alive=True)


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness Probe",
    description="Confirms model is loaded and database is reachable. Returns 503 if not ready to serve traffic.",
)
async def get_readiness(
    engine: BaseTTSEngine = Depends(get_engine_instance),
    db: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    checks = {}

    # 1. Check TTS Engine
    checks["model_loaded"] = engine.is_loaded

    # 2. Check Database connectivity
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        checks["database"] = False

    is_ready = checks["model_loaded"] and checks["database"]
    response_data = ReadinessResponse(
        ready=is_ready,
        model_loaded=engine.is_loaded,
        checks=checks,
    )

    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=response_data.model_dump(), status_code=status_code)
