"""
app/api/deps.py

FastAPI dependency injection module:
- Database session acquisition
- API Key extraction and verification
- Service singletons (TTS, Audio, Storage, Cache, Voice, Job)
"""
from __future__ import annotations

from typing import AsyncGenerator, Optional

from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import AuthenticationError, InvalidAPIKeyError
from app.core.security import hash_api_key, validate_dev_api_key
from app.engines import create_engine
from app.engines.base import BaseTTSEngine
from app.models.db.session import get_db_session
from app.models.db.tables.api_keys import ApiKey
from app.services.audio_service import AudioService
from app.services.cache_service import BaseCacheService, create_cache_service
from app.services.job_service import JobService
from app.services.storage_service import StorageService
from app.services.tts_service import TTSService
from app.services.voice_service import VoiceService

# Global service singletons initialized during lifespan
_ENGINE: Optional[BaseTTSEngine] = None
_STORAGE_SERVICE: Optional[StorageService] = None
_CACHE_SERVICE: Optional[BaseCacheService] = None
_AUDIO_SERVICE: Optional[AudioService] = None
_TTS_SERVICE: Optional[TTSService] = None
_VOICE_SERVICE: Optional[VoiceService] = None
_JOB_SERVICE: Optional[JobService] = None

api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_engine_instance() -> BaseTTSEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine()
    return _ENGINE


def get_storage_service() -> StorageService:
    global _STORAGE_SERVICE
    if _STORAGE_SERVICE is None:
        _STORAGE_SERVICE = StorageService()
    return _STORAGE_SERVICE


def get_cache_service() -> BaseCacheService:
    global _CACHE_SERVICE
    if _CACHE_SERVICE is None:
        _CACHE_SERVICE = create_cache_service()
    return _CACHE_SERVICE


def get_audio_service() -> AudioService:
    global _AUDIO_SERVICE
    if _AUDIO_SERVICE is None:
        _AUDIO_SERVICE = AudioService()
    return _AUDIO_SERVICE


def get_tts_service() -> TTSService:
    global _TTS_SERVICE
    if _TTS_SERVICE is None:
        settings = get_settings()
        _TTS_SERVICE = TTSService(
            engine=get_engine_instance(),
            audio_service=get_audio_service(),
            storage_service=get_storage_service(),
            cache_service=get_cache_service(),
            concurrency_limit=settings.max_concurrent_synthesis,
        )
    return _TTS_SERVICE


def get_voice_service() -> VoiceService:
    global _VOICE_SERVICE
    if _VOICE_SERVICE is None:
        _VOICE_SERVICE = VoiceService(engine=get_engine_instance())
    return _VOICE_SERVICE


def get_job_service() -> JobService:
    global _JOB_SERVICE
    if _JOB_SERVICE is None:
        _JOB_SERVICE = JobService(tts_service_factory=get_tts_service)
    return _JOB_SERVICE


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header_scheme),
    db: AsyncSession = Depends(get_db_session),
) -> Optional[ApiKey]:
    """
    Authenticate request via X-API-Key header.
    Validates against hash in database or dev fallback keys.
    """
    settings = get_settings()

    if not settings.api_key_required:
        return None  # Auth disabled in development

    if not api_key:
        raise AuthenticationError()

    # Check dev/admin bypass keys first
    if validate_dev_api_key(api_key):
        return ApiKey(
            id="dev_key",
            name="Development Key",
            key_prefix=api_key[:8],
            key_hash="",
            is_active=True,
            is_admin=True,
        )

    # Lookup hashed key in database
    hashed = hash_api_key(api_key)
    result = await db.execute(select(ApiKey).where(ApiKey.key_hash == hashed))
    key_record = result.scalar_one_or_none()

    if not key_record or not key_record.is_active:
        raise InvalidAPIKeyError()

    # Update request count
    key_record.request_count += 1
    return key_record


async def require_admin_key(
    api_key_record: Optional[ApiKey] = Depends(verify_api_key),
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> None:
    """
    Ensure the caller has administrator privileges.
    """
    settings = get_settings()
    if x_admin_key and x_admin_key == settings.admin_api_key:
        return

    if api_key_record and api_key_record.is_admin:
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Admin access required.",
    )
