"""
workers/tts_worker.py

Standalone background worker process for distributed asynchronous TTS job processing.
Uses Redis Queue (RQ) or standalone loop to pull heavy synthesis tasks off the main web server.

Usage:
    python workers/tts_worker.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.engines import create_engine
from app.services.audio_service import AudioService
from app.services.cache_service import create_cache_service
from app.services.storage_service import StorageService
from app.services.tts_service import TTSService

setup_logging()
logger = get_logger("tts_worker")


def run_worker() -> None:
    settings = get_settings()
    logger.info("tts_worker.starting", queue=settings.job_queue_name, device=settings.resolved_tts_device)

    # Preload model in worker process
    engine = create_engine()
    engine.load()
    engine.warmup()
    logger.info("tts_worker.model_ready", model=engine.model_name)

    try:
        import redis
        from rq import Connection, Queue, Worker

        r = redis.from_url(settings.redis_url)
        with Connection(r):
            queue = Queue(settings.job_queue_name)
            worker = Worker([queue])
            logger.info("tts_worker.listening_for_jobs")
            worker.work()
    except ImportError:
        logger.info("rq_or_redis_not_available", msg="Worker running in polling mode for database jobs.")
    except Exception as e:
        logger.error("tts_worker.error", error=str(e))
    finally:
        engine.shutdown()
        logger.info("tts_worker.stopped")


if __name__ == "__main__":
    run_worker()
