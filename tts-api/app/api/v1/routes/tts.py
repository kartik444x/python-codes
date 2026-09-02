"""
app/api/v1/routes/tts.py

Core Text-to-Speech API endpoints:
- POST /api/v1/tts: Synchronous synthesis returning binary audio stream or JSON metadata.
- POST /api/v1/tts/async: Asynchronous synthesis returning job tracking token.
"""
from __future__ import annotations

from typing import Optional, Union

from fastapi import APIRouter, Depends, Header, Request, Response, status
from fastapi.responses import Response as FastAPIResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_job_service,
    get_tts_service,
    verify_api_key,
)
from app.core.config import AudioFormat, get_settings
from app.core.exceptions import TextTooLongError
from app.models.db.session import get_db_session
from app.models.db.tables.api_keys import ApiKey
from app.models.requests import AsyncTTSRequest, TTSRequest
from app.models.responses import JobCreateResponse, TTSJsonResponse
from app.services.job_service import JobService
from app.services.tts_service import TTSService

router = APIRouter(prefix="/tts", tags=["Text-to-Speech"])


@router.post(
    "",
    response_model=None,
    summary="Synthesize Speech (Synchronous)",
    description=(
        "Converts input text into natural speech. Returns high-quality audio file (WAV or MP3) "
        "or JSON metadata with audio URL."
    ),
    response_description="Binary audio stream (audio/wav or audio/mpeg) or JSON metadata",
    responses={
        200: {
            "content": {
                "audio/wav": {},
                "audio/mpeg": {},
                "application/json": {},
            },
            "description": "Successfully synthesized audio.",
        },
    },
)
async def synthesize_speech(
    request: TTSRequest,
    raw_request: Request,
    api_key_record: Optional[ApiKey] = Depends(verify_api_key),
    tts_service: TTSService = Depends(get_tts_service),
    db: AsyncSession = Depends(get_db_session),
) -> Union[FastAPIResponse, TTSJsonResponse]:
    settings = get_settings()

    # Enforce synchronous max text length limit
    if len(request.text) > settings.max_text_length:
        raise TextTooLongError(length=len(request.text), max_length=settings.max_text_length)

    request_id = getattr(raw_request.state, "request_id", None)
    api_key_id = api_key_record.id if api_key_record else None

    # Synthesize via orchestrated service
    output = await tts_service.synthesize(
        text=request.text,
        voice=request.voice,
        language=request.language,
        speed=request.speed,
        format_ext=request.format,
        db_session=db,
        api_key_id=api_key_id,
        request_id=request_id,
    )

    # If caller specifically requested JSON response with audio URL
    if request.return_json:
        return TTSJsonResponse(
            request_id=request_id or "",
            audio_id=output.audio_id,
            audio_url=f"/api/v1/audio/{output.audio_id}",
            duration_seconds=round(output.duration_seconds, 2),
            format=output.format_ext,
            voice=output.voice,
            language=output.language,
            model_name=output.model_name,
            model_version=output.model_version,
            processing_time_ms=output.processing_time_ms,
        )

    # Return binary audio directly with appropriate Content-Type
    media_type = "audio/wav" if output.format_ext == "wav" else "audio/mpeg"
    filename = f"speech_{output.audio_id[:8]}.{output.format_ext}"

    headers = {
        "Content-Disposition": f'inline; filename="{filename}"',
        "X-Audio-Duration": str(round(output.duration_seconds, 2)),
        "X-Audio-ID": output.audio_id,
        "X-Cached": "true" if output.was_cached else "false",
    }

    return FastAPIResponse(
        content=output.audio_bytes,
        media_type=media_type,
        headers=headers,
    )


@router.post(
    "/async",
    response_model=JobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Synthesize Speech (Asynchronous)",
    description=(
        "Submits a long-form text-to-speech generation job to the background worker queue. "
        "Returns a job ID for status polling and optional webhook dispatch."
    ),
)
async def synthesize_speech_async(
    request: AsyncTTSRequest,
    api_key_record: Optional[ApiKey] = Depends(verify_api_key),
    job_service: JobService = Depends(get_job_service),
) -> JobCreateResponse:
    settings = get_settings()

    if len(request.text) > settings.max_text_length_async:
        raise TextTooLongError(length=len(request.text), max_length=settings.max_text_length_async)

    api_key_id = api_key_record.id if api_key_record else None
    return await job_service.create_job(request=request, api_key_id=api_key_id)
