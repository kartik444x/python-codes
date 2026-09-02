"""
app/main.py

FastAPI Application Factory & Lifecycle Management.

Startup sequence:
1. Initialize structured logging
2. Load configuration & verify directories
3. Initialize database connection pool & schema
4. Instantiate & warm up TTS ML Model
5. Register middleware & routes
6. Expose /metrics & documentation

Shutdown sequence:
1. Release model memory / CUDA cache
2. Close database pools
3. Close Redis cache connections
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.deps import get_cache_service, get_engine_instance
from app.api.v1.router import api_v1_router
from app.core.config import get_settings
from app.core.exceptions import TTSAPIError
from app.core.logging import get_logger, setup_logging
from app.core.middleware import RequestContextMiddleware
from app.models.db.session import close_db, init_db

# Setup logging before any other module logs
setup_logging()
logger = get_logger(__name__)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{get_settings().rate_limit_requests}/{get_settings().rate_limit_window}second"]
    if get_settings().rate_limit_enabled
    else [],
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application startup and graceful shutdown lifecycle context.
    """
    settings = get_settings()
    logger.info(
        "app.starting",
        app_name=settings.app_name,
        env=settings.app_env.value,
        device=settings.resolved_tts_device,
        model=settings.tts_model.value,
    )

    # 1. Ensure required directories
    settings.ensure_directories()

    # 2. Initialize Database
    try:
        await init_db()
        logger.info("app.db.ready")
    except Exception as e:
        logger.error("app.db.failed", error=str(e))
        raise e

    # 3. Load & Warm up TTS Engine
    engine = get_engine_instance()
    try:
        engine.load()
        if settings.model_warmup_on_start:
            engine.warmup()
        logger.info("app.model.ready", model=engine.model_name, version=engine.model_version)
    except Exception as e:
        logger.warning(
            "app.model.load_deferred_or_failed",
            error=str(e),
            msg="Service will start in degraded mode. Verify model weights via scripts/download_model.py.",
        )

    logger.info("app.started_successfully", host=settings.host, port=settings.port)
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("app.shutting_down")

    # 1. Unload model & release compute resources
    try:
        engine.shutdown()
        logger.info("app.model.unloaded")
    except Exception as e:
        logger.warning("app.model.unload_error", error=str(e))

    # 2. Close cache connections
    try:
        cache = get_cache_service()
        await cache.close()
        logger.info("app.cache.closed")
    except Exception as e:
        logger.warning("app.cache.close_error", error=str(e))

    # 3. Close database connections
    try:
        await close_db()
        logger.info("app.db.closed")
    except Exception as e:
        logger.warning("app.db.close_error", error=str(e))

    logger.info("app.shutdown_complete")


def create_application() -> FastAPI:
    """
    Constructs and configures the FastAPI application.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-Grade Text-to-Speech AI API powered by Kokoro-82M (Apache 2.0). "
            "High-fidelity neural speech synthesis with synchronous streaming and asynchronous queueing."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Attach Rate Limiter state
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Middleware ───────────────────────────────────────────────────────────

    # 1. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Request Tracking, Metrics, and Headers Middleware
    app.add_middleware(RequestContextMiddleware)

    # ── Exception Handlers (Sanitizing internal errors) ──────────────────────

    @app.exception_handler(TTSAPIError)
    async def handle_tts_api_error(request: Request, exc: TTSAPIError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(request_id=request_id),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        # Format human-friendly error messages from Pydantic
        errors = []
        for err in exc.errors():
            loc = " -> ".join(str(x) for x in err["loc"] if x != "body")
            msg = err["msg"]
            errors.append(f"{loc}: {msg}" if loc else msg)

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "; ".join(errors) if errors else "Invalid request body.",
                    "request_id": request_id,
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "HTTP_ERROR",
                    "message": str(exc.detail),
                    "request_id": request_id,
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.error("unhandled_server_error", error=str(exc), request_id=request_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal server error occurred. Please contact support with the request ID.",
                    "request_id": request_id,
                }
            },
        )

    # ── Routes ───────────────────────────────────────────────────────────────

    # Prometheus Metrics endpoint
    @app.get("/metrics", tags=["Observability"], include_in_schema=False)
    async def get_metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Mount API v1 routes
    app.include_router(api_v1_router)

    return app


# Application singleton
app = create_application()
