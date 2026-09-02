"""
app/services/voice_service.py

Voice management and querying service.
Provides listing, filtering by language and gender, and default voice lookup.
"""
from __future__ import annotations

from typing import List, Optional

from app.core.exceptions import UnsupportedVoiceError
from app.engines.base import BaseTTSEngine, VoiceInfo
from app.models.responses import VoiceResponse


class VoiceService:
    """
    Service for querying voices exposed by the underlying TTS engine.
    """

    def __init__(self, engine: BaseTTSEngine) -> None:
        self.engine = engine

    def get_all_voices(
        self,
        language: Optional[str] = None,
        gender: Optional[str] = None,
    ) -> List[VoiceResponse]:
        """
        List all available voices with optional filtering.
        """
        voices = self.engine.get_voices()
        results: List[VoiceResponse] = []

        lang_filter = language.lower().strip() if language else None
        gender_filter = gender.lower().strip() if gender else None

        for v in voices:
            if lang_filter:
                # Match against language_code (e.g. en-US) or language name (e.g. English)
                if not (lang_filter in v.language_code.lower() or lang_filter in v.language.lower() or lang_filter in v.id.lower()):
                    continue

            if gender_filter and v.gender.lower() != gender_filter:
                continue

            results.append(
                VoiceResponse(
                    id=v.id,
                    name=v.name,
                    language=v.language,
                    language_code=v.language_code,
                    gender=v.gender,
                    description=v.description,
                    is_default=v.is_default,
                    sample_rate=v.sample_rate,
                )
            )

        return results

    def get_voice(self, voice_id: str) -> VoiceResponse:
        """
        Get metadata for a single voice.
        """
        voice_info = self.engine.validate_voice(voice_id)
        return VoiceResponse(
            id=voice_info.id,
            name=voice_info.name,
            language=voice_info.language,
            language_code=voice_info.language_code,
            gender=voice_info.gender,
            description=voice_info.description,
            is_default=voice_info.is_default,
            sample_rate=voice_info.sample_rate,
        )
