"""
app/services/audio_service.py

Audio processing service:
- Normalization (peak volume control)
- Chunk concatenation with configurable inter-sentence silence
- Format conversion (WAV to MP3)
- In-memory encoding to minimize disk I/O
- Duration & sample rate calculations
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from app.core.config import AudioFormat, get_settings
from app.core.exceptions import AudioProcessingError, UnsupportedFormatError
from app.core.logging import get_logger
from app.utils.audio import (
    audio_to_wav_bytes,
    concatenate_audio,
    normalize_audio,
    wav_bytes_to_mp3_bytes,
)

logger = get_logger(__name__)


class AudioService:
    """
    Service responsible for audio transformations and format encodings.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def process_and_encode(
        self,
        audio_segments: List[np.ndarray],
        sample_rate: int,
        format_ext: str = "wav",
        silence_between: float = 0.15,
        target_peak: float = 0.95,
    ) -> Tuple[bytes, float]:
        """
        Takes raw model audio segment(s), normalizes, concatenates, and converts to requested format.

        Args:
            audio_segments: List of 1D float32 numpy arrays from the TTS engine.
            sample_rate: Audio sample rate in Hz (e.g. 24000).
            format_ext: 'wav' or 'mp3'.
            silence_between: Seconds of silence between sentence chunks.
            target_peak: Peak amplitude normalization target (0.0 to 1.0).

        Returns:
            Tuple of (encoded_audio_bytes, duration_in_seconds).
        """
        if not audio_segments:
            raise AudioProcessingError("No audio segments provided for processing.")

        try:
            # 1. Concatenate chunks if multiple
            if len(audio_segments) == 1:
                combined_audio = audio_segments[0]
            else:
                combined_audio = concatenate_audio(
                    audio_segments,
                    sample_rate=sample_rate,
                    silence_between_seconds=silence_between,
                )

            # 2. Normalize peak amplitude
            normalized = normalize_audio(combined_audio, target_peak=target_peak)

            # 3. Calculate duration
            duration = float(len(normalized)) / float(sample_rate)

            # 4. Encode to WAV bytes
            wav_bytes = audio_to_wav_bytes(normalized, sample_rate)

            # 5. Handle format conversion
            fmt = format_ext.lower().strip()
            if fmt == AudioFormat.WAV.value:
                return wav_bytes, duration
            elif fmt == AudioFormat.MP3.value:
                mp3_bytes = wav_bytes_to_mp3_bytes(wav_bytes)
                return mp3_bytes, duration
            else:
                raise UnsupportedFormatError(fmt, [f.value for f in AudioFormat])

        except UnsupportedFormatError:
            raise
        except Exception as e:
            logger.error("audio.process_error", error=str(e))
            raise AudioProcessingError(f"Audio processing failed: {e}") from e
