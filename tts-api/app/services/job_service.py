"""
app/services/job_service.py

Async job orchestration service for background long-text synthesis.
Supports both in-process asyncio queues and Redis Queue (RQ) for distributed workers.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

import ulid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.models.db.session import get_session_maker
from app.models.db.tables.jobs import TTSJob
from app.models.requests import AsyncTTSRequest
from app.models.responses import JobCreateResponse, JobStatusResponse

logger = get_logger(__name__)


class JobService:
    """
    Manages asynchronous TTS synthesis jobs.
    """

    def __init__(self, tts_service_factory: Callable) -> None:
        self.settings = get_settings()
        self.tts_service_factory = tts_service_factory
        self._local_tasks: Dict[str, asyncio.Task] = {}

    async def create_job(
        self,
        request: AsyncTTSRequest,
        api_key_id: Optional[str] = None,
    ) -> JobCreateResponse:
        """
        Register a new async TTS job in the DB and schedule background processing.
        """
        job_id = f"job_{ulid.new()}"
        session_factory = get_session_maker()

        async with session_factory() as session:
            job = TTSJob(
                id=job_id,
                api_key_id=api_key_id,
                status="queued",
                text=request.text,
                voice=request.voice,
                language=request.language,
                speed=request.speed,
                format=request.format,
                priority=request.priority,
                webhook_url=request.webhook_url,
            )
            session.add(job)
            await session.commit()

        # Launch worker task in background
        task = asyncio.create_task(self._process_job_task(job_id))
        self._local_tasks[job_id] = task

        logger.info("job.created", job_id=job_id, char_count=len(request.text))

        return JobCreateResponse(
            job_id=job_id,
            status="queued",
            estimated_wait_seconds=max(2, int(len(request.text) / 100)),
            poll_url=f"/api/v1/jobs/{job_id}",
        )

    async def get_job_status(self, job_id: str) -> JobStatusResponse:
        """
        Query status of an existing async job.
        """
        session_factory = get_session_maker()
        async with session_factory() as session:
            result = await session.execute(select(TTSJob).where(TTSJob.id == job_id))
            job = result.scalar_one_or_none()
            if not job:
                raise NotFoundError("Job", job_id)

            return JobStatusResponse(
                job_id=job.id,
                status=job.status,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                audio_id=job.audio_id,
                audio_url=job.audio_url,
                duration_seconds=job.duration_seconds,
                error_message=job.error_message,
                progress_percent=job.progress_percent,
            )

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.
        """
        if job_id in self._local_tasks:
            task = self._local_tasks[job_id]
            if not task.done():
                task.cancel()

        session_factory = get_session_maker()
        async with session_factory() as session:
            result = await session.execute(select(TTSJob).where(TTSJob.id == job_id))
            job = result.scalar_one_or_none()
            if not job:
                raise NotFoundError("Job", job_id)

            if job.status in ("completed", "failed", "cancelled"):
                return False

            job.status = "cancelled"
            job.completed_at = datetime.now(timezone.utc)
            await session.commit()
            logger.info("job.cancelled", job_id=job_id)
            return True

    async def _process_job_task(self, job_id: str) -> None:
        """
        Background task worker logic.
        """
        session_factory = get_session_maker()
        try:
            # 1. Update status to processing
            async with session_factory() as session:
                result = await session.execute(select(TTSJob).where(TTSJob.id == job_id))
                job = result.scalar_one_or_none()
                if not job or job.status == "cancelled":
                    return
                job.status = "processing"
                job.started_at = datetime.now(timezone.utc)
                job.progress_percent = 10
                await session.commit()
                text, voice, language, speed, fmt, webhook_url = (
                    job.text, job.voice, job.language, job.speed, job.format, job.webhook_url
                )

            # 2. Perform TTS synthesis using TTSService
            tts_service = self.tts_service_factory()
            res = await tts_service.synthesize(
                text=text,
                voice=voice,
                language=language,
                speed=speed,
                format_ext=fmt,
            )

            # 3. Mark completed
            async with session_factory() as session:
                result = await session.execute(select(TTSJob).where(TTSJob.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.status = "completed"
                    job.completed_at = datetime.now(timezone.utc)
                    job.audio_id = res.audio_id
                    job.audio_url = f"/api/v1/audio/{res.audio_id}"
                    job.duration_seconds = res.duration_seconds
                    job.progress_percent = 100
                    await session.commit()

            logger.info("job.completed", job_id=job_id, audio_id=res.audio_id)

            # 4. Fire Webhook notification if configured
            if webhook_url:
                await self._send_webhook(webhook_url, job_id, res.audio_id)

        except asyncio.CancelledError:
            logger.info("job.task_cancelled", job_id=job_id)
        except Exception as e:
            logger.error("job.failed", job_id=job_id, error=str(e))
            async with session_factory() as session:
                result = await session.execute(select(TTSJob).where(TTSJob.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.status = "failed"
                    job.completed_at = datetime.now(timezone.utc)
                    job.error_message = str(e)
                    await session.commit()
        finally:
            self._local_tasks.pop(job_id, None)

    async def _send_webhook(self, webhook_url: str, job_id: str, audio_id: str) -> None:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    webhook_url,
                    json={
                        "event": "tts.job.completed",
                        "job_id": job_id,
                        "status": "completed",
                        "audio_url": f"/api/v1/audio/{audio_id}",
                    },
                )
        except Exception as e:
            logger.warning("job.webhook_failed", job_id=job_id, url=webhook_url, error=str(e))
