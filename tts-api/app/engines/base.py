"""
app/engines/base.py

Abstract base class (ABC) for TTS engines.
All engine implementations (Kokoro, Piper, XTTS, etc.) must implement
this interface. The service layer only talks to BaseTTSEngine — it never
imports a concrete engine directly.

This is the "Strategy Pattern" applied to model backends:
    TTSService → BaseTTSEngine (interface)
                    ↑
                KokoroEngine  (or PiperEngine, XTTSEngine in future)

Adding a new model means:
    1. Create a new class in app/engines/ that inherits BaseTTSEngine.
    2. Implement all abstract methods.
    3. Update the engine factory in app/engines/__init__.py.
    4. Update TTS_MODEL in .env.
    Nothing else needs to change.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ─── Data Types ───────────────────────────────────────────────────────────────

@dataclass
class VoiceInfo:
    """Metadata for a single available voice."""
    id: str
    name: str
    language: str
    language_code: str          # e.g. "en-US", "es-ES"
    gender: str = "neutral"     # male | female | neutral
    description: str = ""
    is_default: bool = False
    sample_rate: int = 24000


@dataclass
class SynthesisResult:
    """Result returned from engine.synthesize()."""
    audio: np.ndarray           # Float32 numpy array, values in [-1.0, 1.0]
    sample_rate: int            # Audio sample rate in Hz
    duration_seconds: float     # Duration of generated audio
    voice_id: str               # Voice used
    language: str               # Language used
    model_name: str             # Engine/model name
    model_version: str          # Engine/model version
    chunk_index: int = 0        # Index of this chunk (for multi-chunk synthesis)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineStatus:
    """Current status of the TTS engine."""
    is_loaded: bool
    model_name: str
    model_version: str
    device: str                 # "cpu" or "cuda"
    available_voices: list[str]
    supported_languages: list[str]
    sample_rate: int


# ─── Abstract Base ────────────────────────────────────────────────────────────

class BaseTTSEngine(ABC):
    """
    Abstract base class for all TTS engine implementations.

    Thread-safety: Concrete engines may be called from a thread pool
    (via run_in_executor). Implementations must ensure model.forward()
    calls are either thread-safe or protected by an asyncio.Lock.

    The public `synthesize_async` method wraps the synchronous
    `synthesize` in a thread pool executor so the FastAPI event loop
    is never blocked during CPU/GPU-bound inference.
    """

    def __init__(self) -> None:
        self._loaded: bool = False
        self._executor: ThreadPoolExecutor | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

    # ─── Abstract interface ───────────────────────────────────────────────────

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable engine/model name."""
        ...

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Model version string. Used in cache keys and logs."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Native sample rate of this engine's output."""
        ...

    @abstractmethod
    def load(self) -> None:
        """
        Load model weights into memory and move to the target device.
        Called ONCE at application startup.
        Must set self._loaded = True on success.

        Raises:
            ModelLoadError: If loading fails for any reason.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """
        Release model resources (weights, CUDA memory, etc.).
        Called at graceful shutdown.
        """
        ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_id: str,
        language: str,
        speed: float = 1.0,
    ) -> SynthesisResult:
        """
        Synthesize speech from text. SYNCHRONOUS — blocks the calling thread.

        This method runs inside a ThreadPoolExecutor (via synthesize_async)
        so it MUST NOT call asyncio.* directly.

        Args:
            text: Preprocessed text chunk to synthesize (≤ max_chunk_size chars).
            voice_id: Voice identifier (must be in get_voices()).
            language: Language code (e.g., 'en', 'es').
            speed: Speech speed multiplier (0.5–2.0).

        Returns:
            SynthesisResult with audio array and metadata.

        Raises:
            SynthesisError: If synthesis fails.
            ModelNotLoadedError: If called before load().
        """
        ...

    @abstractmethod
    def get_voices(self) -> list[VoiceInfo]:
        """
        Return all available voices for this engine.

        Returns:
            List of VoiceInfo objects.
        """
        ...

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """
        Return list of supported language codes.

        Returns:
            e.g. ['en', 'es', 'fr', 'de']
        """
        ...

    @abstractmethod
    def warmup(self) -> None:
        """
        Run a short inference to warm up the model (JIT compile, etc.).
        Called once after load(). A warm model has lower latency for the
        first real request.
        """
        ...

    # ─── Concrete methods ─────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """True if the model has been successfully loaded."""
        return self._loaded

    def get_status(self) -> EngineStatus:
        """Return current engine status snapshot."""
        return EngineStatus(
            is_loaded=self._loaded,
            model_name=self.model_name,
            model_version=self.model_version,
            device=self._get_device(),
            available_voices=[v.id for v in self.get_voices()] if self._loaded else [],
            supported_languages=self.get_supported_languages() if self._loaded else [],
            sample_rate=self.sample_rate,
        )

    def _get_device(self) -> str:
        """Return the device the model is running on."""
        return "cpu"  # Overridden by concrete implementations

    def _get_executor(self) -> ThreadPoolExecutor:
        """
        Return the thread pool executor, creating it lazily.
        The executor is shared for all synthesis calls on this engine.
        """
        if self._executor is None:
            # 1 thread per engine instance — prevents model thrashing
            # under concurrent requests (handled at service level via semaphore)
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"tts_{self.model_name}",
            )
        return self._executor

    async def synthesize_async(
        self,
        text: str,
        voice_id: str,
        language: str,
        speed: float = 1.0,
    ) -> SynthesisResult:
        """
        Async wrapper around synthesize().
        Runs the synchronous synthesis in a thread pool so the FastAPI
        event loop is not blocked during CPU/GPU-bound inference.

        Args: Same as synthesize().

        Returns:
            SynthesisResult from the thread pool.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._get_executor(),
            lambda: self.synthesize(text, voice_id, language, speed),
        )

    def validate_voice(self, voice_id: str) -> VoiceInfo:
        """
        Validate that a voice ID is available and return its VoiceInfo.

        Raises:
            UnsupportedVoiceError: If voice_id is not in get_voices().
        """
        from app.core.exceptions import UnsupportedVoiceError
        voice_map = {v.id: v for v in self.get_voices()}
        if voice_id not in voice_map:
            raise UnsupportedVoiceError(voice_id)
        return voice_map[voice_id]

    def validate_language(self, language: str) -> str:
        """
        Validate and normalize language code.

        Raises:
            UnsupportedLanguageError: If language is not supported.
        """
        from app.core.exceptions import UnsupportedLanguageError
        supported = self.get_supported_languages()
        # Normalize: lowercase, strip region code if exact match not found
        lang_lower = language.lower()
        lang_base = lang_lower.split("-")[0].split("_")[0]

        if lang_lower in supported:
            return lang_lower
        if lang_base in supported:
            return lang_base

        raise UnsupportedLanguageError(language, supported)

    def shutdown(self) -> None:
        """
        Release resources: unload model + shutdown thread executor.
        Called at application shutdown.
        """
        try:
            self.unload()
        except Exception:
            pass
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name!r}, "
            f"version={self.model_version!r}, "
            f"loaded={self._loaded})"
        )
