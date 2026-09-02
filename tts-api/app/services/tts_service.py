"""
app/services/tts_service.py

High-level TTS orchestration service.
Enforces separation of concerns:
    API Route -> TTSService -> Engine (Model) -> AudioService -> StorageService
                             -> CacheService
                             -> Usage Logging

Features:
- Global concurrency limiter via asyncio.Semaphore
- Smart text chunking with sentence boundary preservation
- Cache-aside retrieval before invoking model inference
- Automated audio normalization, silence insertion, and format encoding
- Persistent audio metadata recording
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import ulid
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import EmptyTextError, ModelNotLoadedError, TextTooLongError
from app.core.logging import get_logger
from app.core.metrics import (
    TTS_ACTIVE_SYNTHESIS,
    TTS_AUDIO_DURATION_SECONDS,
    TTS_REAL_TIME_FACTOR,
    TTS_SYNTHESIS_DURATION_SECONDS,
    TTS_SYNTHESIS_TOTAL,
)
from app.engines.base import BaseTTSEngine
from app.models.db.tables.audio import AudioRecord
from app.models.db.tables.usage import UsageRecord
from app.services.audio_service import AudioService
from app.services.cache_service import BaseCacheService, generate_cache_key
from app.services.storage_service import StorageService
from app.utils.text import preprocess

logger = get_logger(__name__)


@dataclass
class SynthesisOutput:
    """Orchestrated result of a TTS request."""
    audio_id: str
    audio_bytes: bytes
    duration_seconds: float
    format_ext: str
    voice: str
    language: str
    model_name: str
    model_version: str
    was_cached: bool
    processing_time_ms: int
    char_count: int


class TTSService:
    """
    Main TTS workflow coordinator.
    """

    def __init__(
        self,
        engine: BaseTTSEngine,
        audio_service: AudioService,
        storage_service: StorageService,
        cache_service: BaseCacheService,
        concurrency_limit: int = 2,
    ) -> None:
        self.engine = engine
        self.audio_service = audio_service
        self.storage_service = storage_service
        self.cache_service = cache_service
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.settings = get_settings()

    async def synthesize(
        self,
        text: str,
        voice: str = "af_sarah",
        language: str = "en",
        speed: float = 1.0,
        format_ext: str = "wav",
        db_session: Optional[AsyncSession] = None,
        api_key_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> SynthesisOutput:
        """
        Execute full synthesis pipeline from raw text to stored audio bytes.
        """
        start_time = time.perf_counter()

        if not text or not text.strip():
            raise EmptyTextError()

        if not self.engine.is_loaded:
            raise ModelNotLoadedError()

        # 1. Text Preprocessing & Chunking
        processed = preprocess(text)
        if not processed.chunks:
            raise EmptyTextError()

        char_count = processed.total_chars
        clean_text = processed.cleaned

        # 2. Check Cache
        cache_key = generate_cache_key(
            text=clean_text,
            voice=voice,
            language=language,
            speed=speed,
            format_ext=format_ext,
            model_name=self.engine.model_name,
            model_version=self.engine.model_version,
        )

        cached_audio = await self.cache_service.get(cache_key)
        if cached_audio is not None:
            # Cache hit: compute duration and return immediately
            audio_id = str(ulid.new())
            duration = float(len(cached_audio)) / (self.engine.sample_rate * 2)  # rough estimation for wav
            proc_ms = int((time.perf_counter() - start_time) * 1000)

            # Persist to storage backend for direct URL access if needed
            await self.storage_service.save_audio(audio_id, cached_audio, format_ext)

            logger.info(
                "tts.synthesize.cache_hit",
                request_id=request_id,
                audio_id=audio_id,
                duration_ms=proc_ms,
            )

            if db_session:
                await self._record_usage(
                    db_session=db_session,
                    api_key_id=api_key_id,
                    endpoint="/api/v1/tts",
                    char_count=char_count,
                    audio_duration=duration,
                    proc_ms=proc_ms,
                    was_cached=True,
                )

            return SynthesisOutput(
                audio_id=audio_id,
                audio_bytes=cached_audio,
                duration_seconds=duration,
                format_ext=format_ext,
                voice=voice,
                language=language,
                model_name=self.engine.model_name,
                model_version=self.engine.model_version,
                was_cached=True,
                processing_time_ms=proc_ms,
                char_count=char_count,
            )

        # 3. Model Inference (Guarded by Concurrency Semaphore)
        audio_segments = []
        async with self.semaphore:
            TTS_ACTIVE_SYNTHESIS.inc()
            inference_start = time.perf_counter()
            try:
                for chunk in processed.chunks:
                    res = await self.engine.synthesize_async(
                        text=chunk.text,
                        voice_id=voice,
                        language=language,
                        speed=speed,
                    )
                    audio_segments.append(res.audio)

                TTS_SYNTHESIS_TOTAL.labels(voice=voice, language=language, status="success").inc()
            except Exception as e:
                TTS_SYNTHESIS_TOTAL.labels(voice=voice, language=language, status="error").inc()
                raise e
            finally:
                TTS_ACTIVE_SYNTHESIS.dec()
                inference_duration = time.perf_counter() - inference_start
                TTS_SYNTHESIS_DURATION_SECONDS.labels(voice=voice, language=language).observe(inference_duration)

        # 4. Audio Processing & Encoding (Normalization, Concatenation, MP3 conversion)
        audio_bytes, total_duration = self.audio_service.process_and_encode(
            audio_segments=audio_segments,
            sample_rate=self.engine.sample_rate,
            format_ext=format_ext,
        )

        # Record metrics
        TTS_AUDIO_DURATION_SECONDS.observe(total_duration)
        if total_duration > 0:
            rtf = inference_duration / total_duration
            TTS_REAL_TIME_FACTOR.observe(rtf)

        # 5. Persist Audio to Storage Backend
        audio_id = str(ulid.new())
        storage_uri = await self.storage_service.save_audio(audio_id, audio_bytes, format_ext)

        # 6. Save to Cache
        if self.settings.cache_enabled:
            await self.cache_service.set(cache_key, audio_bytes, ttl_seconds=self.settings.cache_ttl)

        proc_ms = int((time.perf_counter() - start_time) * 1000)

        # 7. Record Metadata & Usage in DB
        if db_session:
            # Save Audio Record
            record = AudioRecord(
                id=audio_id,
                api_key_id=api_key_id,
                storage_uri=storage_uri,
                format=format_ext,
                duration_seconds=total_duration,
                char_count=char_count,
                sample_rate=self.engine.sample_rate,
                voice=voice,
                language=language,
                model_name=self.engine.model_name,
                model_version=self.engine.model_version,
            )
            db_session.add(record)

            # Save Usage Record
            await self._record_usage(
                db_session=db_session,
                api_key_id=api_key_id,
                endpoint="/api/v1/tts",
                char_count=char_count,
                audio_duration=total_duration,
                proc_ms=proc_ms,
                was_cached=False,
            )

        logger.info(
            "tts.synthesize.complete",
            request_id=request_id,
            audio_id=audio_id,
            duration_seconds=round(total_duration, 2),
            proc_ms=proc_ms,
            chunks=len(processed.chunks),
            format=format_ext,
        )

        return SynthesisOutput(
            audio_id=audio_id,
            audio_bytes=audio_bytes,
            duration_seconds=total_duration,
            format_ext=format_ext,
            voice=voice,
            language=language,
            model_name=self.engine.model_name,
            model_version=self.engine.model_version,
            was_cached=False,
            processing_time_ms=proc_ms,
            char_count=char_count,
        )

    async def _record_usage(
        self,
        db_session: AsyncSession,
        api_key_id: Optional[str],
        endpoint: str,
        char_count: int,
        audio_duration: float,
        proc_ms: int,
        was_cached: bool,
    ) -> None:
        try:
            usage = UsageRecord(
                api_key_id=api_key_id,
                endpoint=endpoint,
                characters_processed=char_count,
                audio_duration_seconds=audio_duration,
                processing_time_ms=proc_ms,
                status_code=200,
                was_cached=was_cached,
            )
            db_session.add(usage)
        except Exception as e:
            logger.warning("tts.usage_record_failed", error=str(e))
