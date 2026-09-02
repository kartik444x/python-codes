"""
app/core/logging.py

Structured logging setup using structlog.
Outputs JSON in production, human-readable colored text in development.
Every log entry includes: timestamp, level, logger name, request_id (if set).

Usage:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("tts.synthesize.start", text_length=250, voice="af_sarah")
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.core.config import LogFormat, get_settings


def setup_logging() -> None:
    """
    Configure structlog and stdlib logging.
    Call this ONCE at application startup in app/main.py.
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    use_json = settings.log_format == LogFormat.JSON

    # ── Shared processors (applied to every log event) ───────────────────────
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,        # Merge request-scoped context
        structlog.stdlib.add_logger_name,               # Add "logger" key
        structlog.stdlib.add_log_level,                 # Add "level" key
        structlog.stdlib.PositionalArgumentsFormatter(), # Handle positional args
        structlog.processors.TimeStamper(fmt="iso"),    # ISO 8601 timestamp
        structlog.processors.StackInfoRenderer(),
    ]

    if use_json:
        # ── Production: JSON output ───────────────────────────────────────────
        structlog.configure(
            processors=shared_processors + [
                structlog.processors.dict_tracebacks,       # Convert exceptions to dicts
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # ── Development: colored, human-readable output ───────────────────────
        structlog.configure(
            processors=shared_processors + [
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    # ── Configure stdlib logging to route through structlog ──────────────────
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Forward uvicorn errors through our logging
    logging.getLogger("uvicorn.error").setLevel(log_level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a named structlog logger.

    Args:
        name: Usually __name__ of the calling module.

    Returns:
        A structlog BoundLogger that merges contextvars automatically.
    """
    return structlog.get_logger(name)
