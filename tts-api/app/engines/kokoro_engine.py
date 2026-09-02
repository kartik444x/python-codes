"""
app/engines/kokoro_engine.py

Kokoro-82M TTS engine implementation.

Model: hexgrad/Kokoro-82M
License: Apache 2.0 (commercial use permitted)
Sample Rate: 24,000 Hz
Languages: en (American/British), es, fr, hi, it, ja, zh, pt

This engine wraps the `kokoro` Python package and implements the
BaseTTSEngine interface. The API layer never imports this class directly —
it always uses the abstract BaseTTSEngine type.

Key design decisions:
- Model is loaded ONCE at startup via load().
- Synthesis runs synchronously in a thread pool (not the event loop).
- Language-to-prefix mapping follows Kokoro's KPipeline lang_code convention.
- Voice definitions are declared statically; Kokoro voices are fixed presets.
- Thread safety: KPipeline is not thread-safe; a single executor thread
  serializes all synthesis calls through this engine.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from app.core.exceptions import ModelLoadError, ModelNotLoadedError, SynthesisError
from app.engines.base import BaseTTSEngine, SynthesisResult, VoiceInfo

logger = logging.getLogger(__name__)


# ─── Voice Registry ───────────────────────────────────────────────────────────
# Kokoro uses voice IDs in the format: {gender_prefix}_{name}
# e.g., af_sarah = American female, Sarah
# Complete voice list sourced from hexgrad/kokoro VOICES.md (Apache 2.0)

_KOKORO_VOICES: list[VoiceInfo] = [
    # ── American English (lang_code: 'a') ─────────────────────────────────────
    VoiceInfo(id="af_sarah",   name="Sarah",   language="English (US)", language_code="en-US", gender="female",  is_default=True,  sample_rate=24000),
    VoiceInfo(id="af_bella",   name="Bella",   language="English (US)", language_code="en-US", gender="female",  sample_rate=24000),
    VoiceInfo(id="af_nicole",  name="Nicole",  language="English (US)", language_code="en-US", gender="female",  sample_rate=24000),
    VoiceInfo(id="af_sky",     name="Sky",     language="English (US)", language_code="en-US", gender="female",  sample_rate=24000),
    VoiceInfo(id="am_adam",    name="Adam",    language="English (US)", language_code="en-US", gender="male",    sample_rate=24000),
    VoiceInfo(id="am_michael", name="Michael", language="English (US)", language_code="en-US", gender="male",    sample_rate=24000),
    # ── British English (lang_code: 'b') ─────────────────────────────────────
    VoiceInfo(id="bf_emma",    name="Emma",    language="English (GB)", language_code="en-GB", gender="female",  sample_rate=24000),
    VoiceInfo(id="bf_isabella",name="Isabella",language="English (GB)", language_code="en-GB", gender="female",  sample_rate=24000),
    VoiceInfo(id="bm_george",  name="George",  language="English (GB)", language_code="en-GB", gender="male",    sample_rate=24000),
    VoiceInfo(id="bm_lewis",   name="Lewis",   language="English (GB)", language_code="en-GB", gender="male",    sample_rate=24000),
    # ── Spanish (lang_code: 'e') ──────────────────────────────────────────────
    VoiceInfo(id="ef_dora",    name="Dora",    language="Spanish",      language_code="es-ES", gender="female",  sample_rate=24000),
    VoiceInfo(id="em_alex",    name="Alex",    language="Spanish",      language_code="es-ES", gender="male",    sample_rate=24000),
    VoiceInfo(id="em_santa",   name="Santa",   language="Spanish",      language_code="es-ES", gender="male",    sample_rate=24000),
    # ── French (lang_code: 'f') ───────────────────────────────────────────────
    VoiceInfo(id="ff_siwis",   name="Siwis",   language="French",       language_code="fr-FR", gender="female",  sample_rate=24000),
    # ── Hindi (lang_code: 'h') ───────────────────────────────────────────────
    VoiceInfo(id="hf_alpha",   name="Alpha",   language="Hindi",        language_code="hi-IN", gender="female",  sample_rate=24000),
    VoiceInfo(id="hm_omega",   name="Omega",   language="Hindi",        language_code="hi-IN", gender="male",    sample_rate=24000),
    # ── Italian (lang_code: 'i') ─────────────────────────────────────────────
    VoiceInfo(id="if_sara",    name="Sara",    language="Italian",      language_code="it-IT", gender="female",  sample_rate=24000),
    VoiceInfo(id="im_nicola",  name="Nicola",  language="Italian",      language_code="it-IT", gender="male",    sample_rate=24000),
    # ── Japanese (lang_code: 'j') ────────────────────────────────────────────
    VoiceInfo(id="jf_alpha",   name="Alpha",   language="Japanese",     language_code="ja-JP", gender="female",  sample_rate=24000),
    VoiceInfo(id="jm_kumo",    name="Kumo",    language="Japanese",     language_code="ja-JP", gender="male",    sample_rate=24000),
    # ── Mandarin Chinese (lang_code: 'z') ────────────────────────────────────
    VoiceInfo(id="zf_xiaobei", name="Xiaobei", language="Chinese (Mandarin)", language_code="zh-CN", gender="female", sample_rate=24000),
    VoiceInfo(id="zm_yunjian", name="Yunjian", language="Chinese (Mandarin)", language_code="zh-CN", gender="male",   sample_rate=24000),
    # ── Brazilian Portuguese (lang_code: 'p') ────────────────────────────────
    VoiceInfo(id="pf_dora",    name="Dora",    language="Portuguese (BR)", language_code="pt-BR", gender="female", sample_rate=24000),
    VoiceInfo(id="pm_alex",    name="Alex",    language="Portuguese (BR)", language_code="pt-BR", gender="male",   sample_rate=24000),
]

# Map voice prefix → Kokoro lang_code (used by KPipeline)
# The first letter of a voice ID encodes the language
_VOICE_PREFIX_TO_LANG_CODE: dict[str, str] = {
    "a": "a",   # American English
    "b": "b",   # British English
    "e": "e",   # Spanish
    "f": "f",   # French
    "h": "h",   # Hindi
    "i": "i",   # Italian
    "j": "j",   # Japanese
    "z": "z",   # Chinese (Mandarin)
    "p": "p",   # Brazilian Portuguese
}

# User-facing language codes → Kokoro lang_code prefix
_LANGUAGE_TO_LANG_CODE: dict[str, str] = {
    "en": "a",       # Default American English
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "es-es": "e",
    "fr": "f",
    "fr-fr": "f",
    "hi": "h",
    "hi-in": "h",
    "it": "i",
    "it-it": "i",
    "ja": "j",
    "ja-jp": "j",
    "zh": "z",
    "zh-cn": "z",
    "pt": "p",
    "pt-br": "p",
}

_SUPPORTED_LANGUAGES = sorted(set(_LANGUAGE_TO_LANG_CODE.keys()))


# ─── Engine Implementation ────────────────────────────────────────────────────

class KokoroEngine(BaseTTSEngine):
    """
    TTS engine backed by Kokoro-82M (hexgrad/Kokoro-82M, Apache 2.0).

    Thread safety: KPipeline is NOT thread-safe. This class is always
    called from a single-threaded executor (max_workers=1 in BaseTTSEngine).
    Do not change the executor pool size without reviewing thread safety.
    """

    def __init__(self, device: str = "cpu", model_dir: Path | None = None) -> None:
        """
        Args:
            device: 'cpu' or 'cuda'. Resolved from settings before instantiation.
            model_dir: Directory for cached model weights. Kokoro uses HF Hub cache
                       by default; this sets the HF_HOME/cache dir.
        """
        super().__init__()
        self._device = device
        self._model_dir = model_dir
        self._pipelines: dict[str, Any] = {}  # lang_code → KPipeline instance
        self._voice_map: dict[str, VoiceInfo] = {v.id: v for v in _KOKORO_VOICES}

    # ─── Properties ──────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return "kokoro"

    @property
    def model_version(self) -> str:
        try:
            import kokoro
            return getattr(kokoro, "__version__", "0.9.x")
        except ImportError:
            return "unknown"

    @property
    def sample_rate(self) -> int:
        return 24_000

    def _get_device(self) -> str:
        return self._device

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load the Kokoro model. Creates one KPipeline per language group
        lazily (on first synthesis call for that language), but validates
        the import and initializes the primary English pipeline eagerly.
        """
        logger.info("kokoro.load.start", device=self._device)
        t0 = time.perf_counter()

        try:
            import kokoro as _kokoro_module  # noqa: F401 — verify import
        except ImportError as e:
            raise ModelLoadError(
                f"kokoro package not installed. Run: pip install kokoro\n"
                f"Original error: {e}"
            ) from e

        # Set model cache directory if specified
        if self._model_dir:
            import os
            os.environ.setdefault("HF_HOME", str(self._model_dir))

        # Pre-load English pipeline eagerly (most common use case)
        self._get_pipeline("a")  # American English

        self._loaded = True
        elapsed = time.perf_counter() - t0
        logger.info("kokoro.load.complete", elapsed_seconds=round(elapsed, 2), device=self._device)

    def unload(self) -> None:
        """Release pipeline references and CUDA memory."""
        for lang_code, pipeline in self._pipelines.items():
            try:
                del pipeline
            except Exception:
                pass
        self._pipelines.clear()

        if self._device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info("kokoro.unload.cuda_cache_cleared")
            except Exception:
                pass

        self._loaded = False
        logger.info("kokoro.unload.complete")

    def warmup(self) -> None:
        """
        Run a short synthesis to trigger PyTorch JIT compilation
        and any model initialization that happens on first forward pass.
        """
        if not self._loaded:
            raise ModelNotLoadedError()

        logger.info("kokoro.warmup.start")
        t0 = time.perf_counter()
        try:
            result = self.synthesize(
                text="Warming up.",
                voice_id="af_sarah",
                language="en",
                speed=1.0,
            )
            elapsed = time.perf_counter() - t0
            logger.info(
                "kokoro.warmup.complete",
                duration_seconds=round(result.duration_seconds, 2),
                elapsed_ms=round(elapsed * 1000),
            )
        except Exception as e:
            logger.warning("kokoro.warmup.failed", error=str(e))
            # Warmup failure is non-fatal — the engine may still work

    # ─── Core Synthesis ───────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        voice_id: str,
        language: str,
        speed: float = 1.0,
    ) -> SynthesisResult:
        """
        Synthesize speech for a single text chunk.

        Args:
            text: Preprocessed text chunk (≤ 500 chars recommended).
            voice_id: Voice identifier from _KOKORO_VOICES.
            language: Language code (e.g., 'en', 'es').
            speed: Speed multiplier (0.5–2.0).

        Returns:
            SynthesisResult with float32 audio array at 24kHz.

        Raises:
            ModelNotLoadedError: If called before load().
            SynthesisError: If synthesis fails for any reason.
        """
        if not self._loaded:
            raise ModelNotLoadedError()

        import numpy as np

        # Resolve voice and language
        voice_info = self.validate_voice(voice_id)
        resolved_language = self.validate_language(language)

        # Determine the Kokoro lang_code from the voice prefix
        # Voice prefix overrides language setting (e.g., 'bf_emma' forces British English)
        voice_prefix = voice_id[0]  # 'a', 'b', 'e', etc.
        lang_code = _VOICE_PREFIX_TO_LANG_CODE.get(voice_prefix, "a")

        pipeline = self._get_pipeline(lang_code)

        logger.debug(
            "kokoro.synthesize",
            text_length=len(text),
            voice=voice_id,
            language=resolved_language,
            lang_code=lang_code,
            speed=speed,
        )

        t0 = time.perf_counter()
        audio_segments: list[np.ndarray] = []

        try:
            # KPipeline.__call__ returns a generator of (graphemes, phonemes, audio)
            # Each iteration is a sentence-level chunk from Kokoro internally
            for _, _, audio_chunk in pipeline(text, voice=voice_id, speed=speed):
                if audio_chunk is not None and len(audio_chunk) > 0:
                    audio_segments.append(np.array(audio_chunk, dtype=np.float32))

        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error(
                "kokoro.synthesize.error",
                error=str(e),
                error_type=type(e).__name__,
                elapsed_ms=round(elapsed * 1000),
            )
            raise SynthesisError(f"Kokoro synthesis failed: {e}") from e

        if not audio_segments:
            raise SynthesisError("Kokoro returned empty audio. Check text and voice settings.")

        # Concatenate internal Kokoro segments
        if len(audio_segments) == 1:
            audio = audio_segments[0]
        else:
            audio = np.concatenate(audio_segments, axis=0)

        elapsed = time.perf_counter() - t0
        duration = len(audio) / self.sample_rate

        logger.debug(
            "kokoro.synthesize.complete",
            duration_seconds=round(duration, 2),
            elapsed_ms=round(elapsed * 1000),
            rtf=round(elapsed / duration, 3) if duration > 0 else 0,
        )

        return SynthesisResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            voice_id=voice_id,
            language=resolved_language,
            model_name=self.model_name,
            model_version=self.model_version,
        )

    # ─── Voice / Language Queries ─────────────────────────────────────────────

    def get_voices(self) -> list[VoiceInfo]:
        return list(_KOKORO_VOICES)

    def get_supported_languages(self) -> list[str]:
        return _SUPPORTED_LANGUAGES

    def get_default_voice(self, language: str = "en") -> str:
        """Return the default voice ID for a given language."""
        resolved = self.validate_language(language)
        lang_code = _LANGUAGE_TO_LANG_CODE.get(resolved, "a")

        # Find first voice matching this language code prefix
        for voice in _KOKORO_VOICES:
            if voice.id.startswith(lang_code) and voice.is_default:
                return voice.id

        # Fallback: first voice with matching prefix
        for voice in _KOKORO_VOICES:
            if voice.id.startswith(lang_code):
                return voice.id

        return "af_sarah"  # Ultimate fallback

    # ─── Internal Helpers ─────────────────────────────────────────────────────

    def _get_pipeline(self, lang_code: str) -> "KPipeline":
        """
        Get or create a KPipeline for a given language code.
        Pipelines are cached per language to avoid repeated initialization.
        """
        if lang_code not in self._pipelines:
            logger.info("kokoro.pipeline.init", lang_code=lang_code)
            try:
                from kokoro import KPipeline
                pipeline = KPipeline(lang_code=lang_code)
                self._pipelines[lang_code] = pipeline
                logger.info("kokoro.pipeline.ready", lang_code=lang_code)
            except Exception as e:
                raise ModelLoadError(
                    f"Failed to initialize Kokoro pipeline for lang_code='{lang_code}': {e}"
                ) from e
        return self._pipelines[lang_code]


# ─── Type annotation fix ──────────────────────────────────────────────────────
from typing import Any  # noqa: E402 — required for _pipelines dict annotation
