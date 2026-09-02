"""
app/core/config.py

Centralized configuration management using Pydantic Settings.
All configurable values come from environment variables or a .env file.
No magic strings or configuration scattered throughout source code.
"""
from __future__ import annotations

import os
import secrets
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─── Enumerations ────────────────────────────────────────────────────────────

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class TTSDevice(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class TTSModel(str, Enum):
    KOKORO = "kokoro"


class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"


class StorageBackend(str, Enum):
    LOCAL = "local"
    S3 = "s3"


class CacheBackend(str, Enum):
    REDIS = "redis"
    MEMORY = "memory"
    NONE = "none"


class LogFormat(str, Enum):
    JSON = "json"
    TEXT = "text"


# ─── Settings ────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Pydantic automatically coerces types and validates values.
    Add new settings here — never hard-code them in application code.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars gracefully
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "Production TTS API"
    app_env: Environment = Environment.DEVELOPMENT
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=32)
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.TEXT

    # ── TTS Model ────────────────────────────────────────────────────────────
    tts_model: TTSModel = TTSModel.KOKORO
    tts_model_version: str = "0.9.4"
    tts_device: TTSDevice = TTSDevice.AUTO
    tts_model_dir: Path = Path("./models")
    max_text_length: int = Field(default=5000, ge=1, le=100_000)
    max_text_length_async: int = Field(default=50_000, ge=1, le=1_000_000)
    max_concurrent_synthesis: int = Field(default=2, ge=1, le=16)
    model_warmup_on_start: bool = True

    # ── Audio ─────────────────────────────────────────────────────────────────
    audio_sample_rate: int = Field(default=24_000, ge=8_000, le=96_000)
    audio_default_format: AudioFormat = AudioFormat.WAV
    audio_retention_hours: int = Field(default=24, ge=1, le=8760)
    audio_dir: Path = Path("./generated_audio")

    # ── Security ──────────────────────────────────────────────────────────────
    api_key_required: bool = True
    secret_key: str = Field(default_factory=lambda: secrets.token_hex(32))
    dev_api_keys: str = ""  # Comma-separated keys for simple dev mode
    cors_origins: str = "http://localhost:3000,http://localhost:8080"
    admin_api_key: str = Field(default_factory=lambda: secrets.token_hex(16))

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_enabled: bool = True
    rate_limit_requests: int = Field(default=100, ge=1)
    rate_limit_window: int = Field(default=60, ge=1)   # seconds
    rate_limit_storage: str = "memory"  # memory | redis

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./tts_api.db"
    database_pool_size: int = Field(default=5, ge=1)
    database_max_overflow: int = Field(default=10, ge=0)

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_password: str = ""
    redis_max_connections: int = Field(default=10, ge=1)

    # ── Caching ───────────────────────────────────────────────────────────────
    cache_enabled: bool = True
    cache_ttl: int = Field(default=3600, ge=0)  # 0 = no TTL
    cache_backend: CacheBackend = CacheBackend.REDIS

    # ── Storage ───────────────────────────────────────────────────────────────
    storage_backend: StorageBackend = StorageBackend.LOCAL
    s3_bucket: str = ""
    s3_endpoint: str = ""
    s3_region: str = "us-east-1"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_public_url: str = ""

    # ── Async Jobs ────────────────────────────────────────────────────────────
    job_queue_name: str = "tts-jobs"
    job_timeout: int = Field(default=300, ge=1)
    job_result_ttl: int = Field(default=86400, ge=0)
    job_failure_ttl: int = Field(default=86400, ge=0)

    # ── Observability ─────────────────────────────────────────────────────────
    metrics_enabled: bool = True
    metrics_prefix: str = "tts_api"

    # ─── Computed properties ──────────────────────────────────────────────────

    @property
    def is_development(self) -> bool:
        return self.app_env == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        return self.app_env == Environment.PRODUCTION

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS from comma-separated string to list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def dev_api_keys_list(self) -> list[str]:
        """Parse DEV_API_KEYS from comma-separated string to list."""
        return [k.strip() for k in self.dev_api_keys.split(",") if k.strip()]

    @property
    def resolved_tts_device(self) -> str:
        """
        Resolve 'auto' device to actual 'cuda' or 'cpu' based on availability.
        Import is local to avoid eager torch import at config load time.
        """
        if self.tts_device == TTSDevice.AUTO:
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.tts_device.value

    # ─── Validators ──────────────────────────────────────────────────────────

    @field_validator("tts_model_dir", "audio_dir", mode="before")
    @classmethod
    def make_path_absolute(cls, v: Any) -> Path:
        """Ensure all path settings are absolute Path objects."""
        path = Path(v)
        if not path.is_absolute():
            # Resolve relative to the project root (directory of this file's parent)
            project_root = Path(__file__).resolve().parent.parent.parent
            return project_root / path
        return path

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """
        Enforce stricter requirements in production.
        This prevents developers from accidentally deploying with insecure defaults.
        """
        if self.is_production:
            if not self.api_key_required:
                raise ValueError(
                    "API_KEY_REQUIRED must be true in production. "
                    "Set APP_ENV=development to disable auth for local testing."
                )
            if len(self.secret_key) < 32:
                raise ValueError(
                    "SECRET_KEY must be at least 32 characters in production."
                )
            if self.storage_backend == StorageBackend.LOCAL:
                import warnings
                warnings.warn(
                    "LOCAL storage in production is not recommended for distributed deployment. "
                    "Consider S3_BACKEND=s3.",
                    stacklevel=2,
                )
        return self

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.tts_model_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.
    Use this function everywhere instead of instantiating Settings directly.
    The lru_cache ensures the .env file is only read once.

    Example:
        from app.core.config import get_settings
        settings = get_settings()
    """
    return Settings()
