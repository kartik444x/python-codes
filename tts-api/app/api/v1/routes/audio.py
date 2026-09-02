"""
app/api/v1/routes/audio.py

Audio retrieval endpoints:
- GET /api/v1/audio/{audio_id}: Stream or download previously synthesized audio files.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response as FastAPIResponse

from app.api.deps import get_storage_service, verify_api_key
from app.core.exceptions import NotFoundError
from app.models.db.tables.api_keys import ApiKey
from app.services.storage_service import StorageService

router = APIRouter(prefix="/audio", tags=["Audio Storage"])


@router.get(
    "/{audio_id}",
    summary="Retrieve Generated Audio",
    description="Stream or download a previously generated audio asset by audio_id.",
    responses={
        200: {
            "content": {
                "audio/wav": {},
                "audio/mpeg": {},
            },
            "description": "Audio binary data stream.",
        },
    },
)
async def get_audio_file(
    audio_id: str,
    format: str = Query("wav", description="Audio format ('wav' or 'mp3')"),
    download: bool = Query(False, description="If true, sets attachment header to trigger browser download"),
    api_key_record: Optional[ApiKey] = Depends(verify_api_key),
    storage_service: StorageService = Depends(get_storage_service),
) -> FastAPIResponse:
    fmt = format.lower().strip()
    data = await storage_service.get_audio(audio_id=audio_id, format_ext=fmt)

    media_type = "audio/wav" if fmt == "wav" else "audio/mpeg"
    disposition = "attachment" if download else "inline"
    filename = f"{audio_id}.{fmt}"

    headers = {
        "Content-Disposition": f'{disposition}; filename="{filename}"',
        "Content-Length": str(len(data)),
        "Cache-Control": "public, max-age=86400",
    }

    return FastAPIResponse(
        content=data,
        media_type=media_type,
        headers=headers,
    )
