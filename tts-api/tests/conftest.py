"""
tests/conftest.py

Shared pytest fixtures and test doubles.
Uses MockTTSEngine to provide ultra-fast, deterministic test execution
without requiring GPU hardware or downloading model weights during CI/testing.
"""
from __future__ import annotations

import os
from typing import AsyncGenerator, Generator, List

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Set test environment before loading settings
os.environ["APP_ENV"] = "development"
os.environ["API_KEY_REQUIRED"] = "true"
os.environ["DEV_API_KEYS"] = "test-key-123,admin-test-key"
os.environ["ADMIN_API_KEY"] = "admin-secret-key"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["CACHE_BACKEND"] = "memory"
os.environ["STORAGE_BACKEND"] = "local"
os.environ["AUDIO_DIR"] = "./test_generated_audio"
os.environ["MODEL_WARMUP_ON_START"] = "false"

from app.api.deps import (
    get_audio_service,
    get_cache_service,
    get_engine_instance,
    get_job_service,
    get_storage_service,
    get_tts_service,
    get_voice_service,
)
from app.engines.base import BaseTTSEngine, SynthesisResult, VoiceInfo
from app.main import app
from app.models.db.base import Base
from app.models.db.session import get_db_session, get_engine, init_db
from app.services.audio_service import AudioService
from app.services.cache_service import InMemoryCacheService
from app.services.job_service import JobService
from app.services.storage_service import LocalStorageBackend, StorageService
from app.services.tts_service import TTSService
from app.services.voice_service import VoiceService


class MockTTSEngine(BaseTTSEngine):
    """
    Fast test double generating synthetic 24kHz tone audio.
    """

    @property
    def model_name(self) -> str:
        return "mock_kokoro"

    @property
    def model_version(self) -> str:
        return "1.0.0-test"

    @property
    def sample_rate(self) -> int:
        return 24000

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def warmup(self) -> None:
        pass

    def synthesize(
        self,
        text: str,
        voice_id: str,
        language: str,
        speed: float = 1.0,
    ) -> SynthesisResult:
        self.validate_voice(voice_id)
        # Generate 0.5s of synthetic sine wave
        duration = 0.5
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz A tone

        return SynthesisResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            voice_id=voice_id,
            language=language,
            model_name=self.model_name,
            model_version=self.model_version,
        )

    def get_voices(self) -> List[VoiceInfo]:
        return [
            VoiceInfo(id="af_sarah", name="Sarah", language="English (US)", language_code="en-US", gender="female", is_default=True),
            VoiceInfo(id="am_adam", name="Adam", language="English (US)", language_code="en-US", gender="male"),
            VoiceInfo(id="bf_emma", name="Emma", language="English (GB)", language_code="en-GB", gender="female"),
            VoiceInfo(id="ef_dora", name="Dora", language="Spanish", language_code="es-ES", gender="female"),
        ]

    def get_supported_languages(self) -> List[str]:
        return ["en", "en-us", "en-gb", "es", "es-es"]


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    """Initializes in-memory SQLite schema for each test."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest.fixture
def mock_engine() -> MockTTSEngine:
    engine = MockTTSEngine()
    engine.load()
    return engine


@pytest.fixture
def test_tts_service(mock_engine) -> TTSService:
    audio_service = AudioService()
    storage_service = StorageService()
    cache_service = InMemoryCacheService()
    return TTSService(
        engine=mock_engine,
        audio_service=audio_service,
        storage_service=storage_service,
        cache_service=cache_service,
        concurrency_limit=2,
    )


@pytest_asyncio.fixture
async def client(mock_engine, test_tts_service) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP test client with dependency overrides."""
    # Override engine and service dependencies with mock
    app.dependency_overrides[get_engine_instance] = lambda: mock_engine
    app.dependency_overrides[get_tts_service] = lambda: test_tts_service
    app.dependency_overrides[get_voice_service] = lambda: VoiceService(engine=mock_engine)
    app.dependency_overrides[get_job_service] = lambda: JobService(tts_service_factory=lambda: test_tts_service)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers() -> dict:
    return {"X-API-Key": "test-key-123"}


@pytest.fixture
def admin_headers() -> dict:
    return {"X-API-Key": "test-key-123", "X-Admin-Key": "admin-secret-key"}
