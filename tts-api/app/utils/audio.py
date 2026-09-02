"""
app/utils/audio.py

Audio utility functions for WAV/MP3 processing.

Responsibilities:
- Concatenate multiple numpy audio arrays into one
- Normalize audio levels (prevent clipping)
- Add silence padding between chunks
- Convert WAV bytes to MP3 (requires ffmpeg)
- Encode numpy arrays to WAV bytes
- Read WAV bytes and return numpy array + sample rate
- Compute audio duration
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


class AudioInfo(NamedTuple):
    """Metadata about an audio segment."""
    duration_seconds: float
    sample_rate: int
    channels: int
    num_samples: int


# ─── Core Audio Operations ────────────────────────────────────────────────────

def compute_duration(audio: np.ndarray, sample_rate: int) -> float:
    """Return duration of audio array in seconds."""
    return len(audio) / sample_rate


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalize audio to a target peak amplitude.
    Prevents clipping while maximizing loudness.

    Args:
        audio: 1D float32 audio array in range [-1.0, 1.0].
        target_peak: Desired peak amplitude (0.0–1.0).

    Returns:
        Normalized audio array.
    """
    if audio.size == 0:
        return audio
    peak = np.abs(audio).max()
    if peak == 0.0:
        return audio
    return (audio / peak * target_peak).astype(np.float32)


def make_silence(duration_seconds: float, sample_rate: int) -> np.ndarray:
    """
    Create a silence array of the given duration.

    Args:
        duration_seconds: Duration of silence in seconds.
        sample_rate: Audio sample rate.

    Returns:
        Zero-filled float32 array.
    """
    num_samples = int(duration_seconds * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)


def concatenate_audio(
    segments: list[np.ndarray],
    sample_rate: int,
    silence_between_seconds: float = 0.15,
) -> np.ndarray:
    """
    Concatenate multiple audio arrays with optional silence between them.

    Args:
        segments: List of 1D float32 audio arrays.
        sample_rate: Sample rate (must be the same for all segments).
        silence_between_seconds: Duration of silence to insert between segments.

    Returns:
        Concatenated audio array.

    Raises:
        ValueError: If segments list is empty.
    """
    if not segments:
        raise ValueError("No audio segments to concatenate.")
    if len(segments) == 1:
        return segments[0]

    silence = make_silence(silence_between_seconds, sample_rate)
    parts: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        if seg.ndim > 1:
            seg = seg.mean(axis=1)  # Downmix to mono if stereo
        parts.append(seg.astype(np.float32))
        if i < len(segments) - 1:
            parts.append(silence)

    return np.concatenate(parts, axis=0)


def audio_to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int,
) -> bytes:
    """
    Encode a float32 numpy array as WAV bytes (in-memory, no disk I/O).

    Args:
        audio: 1D float32 audio array, values in [-1.0, 1.0].
        sample_rate: Audio sample rate in Hz.

    Returns:
        WAV file content as bytes.
    """
    import soundfile as sf

    buf = io.BytesIO()
    # soundfile expects float64 for some subtype handling; cast explicitly
    sf.write(buf, audio.astype(np.float32), sample_rate, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


def wav_bytes_to_array(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Decode WAV bytes to a numpy array.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    import soundfile as sf

    buf = io.BytesIO(wav_bytes)
    audio, sample_rate = sf.read(buf, dtype="float32", always_2d=False)
    return audio, sample_rate


def wav_bytes_to_mp3_bytes(
    wav_bytes: bytes,
    bitrate: str = "128k",
) -> bytes:
    """
    Convert WAV bytes to MP3 bytes using pydub + ffmpeg.

    Requires `ffmpeg` to be installed as a system binary.
    On Windows: install via https://ffmpeg.org/download.html or `choco install ffmpeg`
    On Linux: `apt-get install ffmpeg`
    On macOS: `brew install ffmpeg`

    Args:
        wav_bytes: Input WAV audio as bytes.
        bitrate: MP3 bitrate string (e.g., '128k', '192k', '320k').

    Returns:
        MP3 file content as bytes.

    Raises:
        RuntimeError: If pydub or ffmpeg is not available.
    """
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise RuntimeError(
            "pydub is required for MP3 export. Install it with: pip install pydub"
        ) from e

    try:
        wav_buf = io.BytesIO(wav_bytes)
        audio_segment = AudioSegment.from_wav(wav_buf)

        mp3_buf = io.BytesIO()
        audio_segment.export(mp3_buf, format="mp3", bitrate=bitrate)
        mp3_buf.seek(0)
        return mp3_buf.read()

    except Exception as e:
        if "ffmpeg" in str(e).lower() or "avconv" in str(e).lower():
            raise RuntimeError(
                "ffmpeg is not installed or not on PATH. "
                "Install it with:\n"
                "  Windows: choco install ffmpeg  (or download from ffmpeg.org)\n"
                "  Linux:   sudo apt-get install ffmpeg\n"
                "  macOS:   brew install ffmpeg"
            ) from e
        raise RuntimeError(f"MP3 conversion failed: {e}") from e


def get_audio_info(audio: np.ndarray, sample_rate: int) -> AudioInfo:
    """Return metadata about an audio array."""
    return AudioInfo(
        duration_seconds=compute_duration(audio, sample_rate),
        sample_rate=sample_rate,
        channels=1 if audio.ndim == 1 else audio.shape[1],
        num_samples=len(audio),
    )


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg binary is available on PATH."""
    import shutil
    return shutil.which("ffmpeg") is not None


def save_wav_file(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    """
    Save audio array to a WAV file.

    Args:
        audio: Float32 numpy array.
        sample_rate: Sample rate.
        path: Destination file path.
    """
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="FLOAT")
    logger.debug("audio.saved", path=str(path), duration=compute_duration(audio, sample_rate))


def load_wav_file(path: Path) -> tuple[np.ndarray, int]:
    """
    Load a WAV file and return (audio_array, sample_rate).

    Args:
        path: Source WAV file path.

    Returns:
        Tuple of (audio_array as float32, sample_rate).
    """
    import soundfile as sf
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return audio, sample_rate
