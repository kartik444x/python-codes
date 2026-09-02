"""
app/engines/__init__.py

Engine factory: resolves TTS_MODEL setting to a concrete engine instance.
The rest of the application imports get_engine() from here.

Adding a new engine:
    1. Create app/engines/my_engine.py implementing BaseTTSEngine.
    2. Add 'my_model' to the factory dict below.
    3. Set TTS_MODEL=my_model in .env.
"""
from __future__ import annotations

from app.core.config import TTSModel, get_settings
from app.engines.base import BaseTTSEngine


def create_engine() -> BaseTTSEngine:
    """
    Instantiate the correct TTS engine based on TTS_MODEL setting.
    The engine is NOT loaded yet — call engine.load() to initialize weights.

    Returns:
        An unloaded BaseTTSEngine instance.

    Raises:
        ValueError: If TTS_MODEL specifies an unknown engine.
    """
    settings = get_settings()

    if settings.tts_model == TTSModel.KOKORO:
        from app.engines.kokoro_engine import KokoroEngine
        return KokoroEngine(
            device=settings.resolved_tts_device,
            model_dir=settings.tts_model_dir,
        )

    raise ValueError(
        f"Unknown TTS model: '{settings.tts_model}'. "
        f"Supported models: {[m.value for m in TTSModel]}"
    )
