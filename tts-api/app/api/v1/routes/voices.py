"""
app/api/v1/routes/voices.py

Voice catalog endpoints and voice registration with strict ethical/licensing safeguards.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.deps import get_voice_service, verify_api_key
from app.models.db.tables.api_keys import ApiKey
from app.models.responses import VoiceListResponse, VoiceResponse
from app.services.voice_service import VoiceService

router = APIRouter(prefix="/voices", tags=["Voices"])


class VoiceRegistrationRequest(BaseModel):
    """
    Request model for registering a custom voice profile.
    Requires explicit legal consent and attribution declarations.
    """
    voice_name: str = Field(..., min_length=2, max_length=64, description="Unique display name for the voice.")
    language_code: str = Field(..., min_length=2, max_length=10, description="BCP-47 language code (e.g. en-US).")
    gender: str = Field("neutral", description="Gender tag (male, female, neutral).")
    consent_confirmed: bool = Field(
        ...,
        description="Explicit confirmation that the voice owner gave written consent for AI voice synthesis usage."
    )
    license_type: str = Field(
        "Commercial-Permitted",
        description="License model governing the voice profile (e.g. Apache-2.0, Proprietary-Licensed)."
    )
    attribution_notes: Optional[str] = Field(
        None,
        description="Mandatory attribution details or actor credits if required by license."
    )


@router.get(
    "",
    response_model=VoiceListResponse,
    summary="List Available Voices",
    description="Retrieve catalog of supported TTS voices. Allows filtering by language or gender.",
)
async def list_voices(
    language: Optional[str] = Query(None, description="Filter by language code (e.g. 'en', 'es', 'ja')"),
    gender: Optional[str] = Query(None, description="Filter by gender ('female', 'male', 'neutral')"),
    voice_service: VoiceService = Depends(get_voice_service),
) -> VoiceListResponse:
    voices = voice_service.get_all_voices(language=language, gender=gender)
    return VoiceListResponse(voices=voices, total=len(voices))


@router.get(
    "/{voice_id}",
    response_model=VoiceResponse,
    summary="Get Voice Details",
    description="Retrieve metadata for a specific voice by ID.",
)
async def get_voice(
    voice_id: str,
    voice_service: VoiceService = Depends(get_voice_service),
) -> VoiceResponse:
    return voice_service.get_voice(voice_id)


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Register Custom Voice",
    description=(
        "Register a new voice profile. Requires explicit consent confirmation and license metadata "
        "to prevent unauthorized cloning and impersonation."
    ),
)
async def register_voice(
    payload: VoiceRegistrationRequest,
    api_key: Optional[ApiKey] = Depends(verify_api_key),
) -> dict:
    if not payload.consent_confirmed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Voice registration rejected: Explicit owner consent (consent_confirmed=true) is mandatory."
        )

    # Scaffolding for voice registration / custom cloning profiles
    return {
        "status": "registered",
        "voice_id": f"custom_{payload.voice_name.lower().replace(' ', '_')}",
        "name": payload.voice_name,
        "language_code": payload.language_code,
        "license_type": payload.license_type,
        "message": "Voice profile registered with valid consent verification.",
    }
