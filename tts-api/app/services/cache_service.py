"""
app/services/cache_service.py

Deterministic audio caching service.
Avoids running expensive ML inference on duplicate synthesis requests.

Cache Key Strategy:
    sha256(f"{model_name}:{model_version}:{language}:{voice}:{speed:.2f}:{fmt}:{normalized_text}")

When the model is updated or settings change, cache keys naturally invalidate,
guaranteeing old audio is never served for a newer model version.
"""
from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from typing import Optional

from app.core.config import CacheBackend, get_settings
from app.core.logging import get_logger
from app.core.metrics import CACHE_HITS_TOTAL, CACHE_MISSES_TOTAL

logger = get_logger(__name__)


def generate_cache_key(
    text: str,
    voice: str,
    language: str,
    speed: float,
    format_ext: str,
    model_name: str,
    model_version: str,
) -> str:
    """
    Generate a deterministic SHA-256 cache key.
    """
    normalized_text = text.strip()
    payload = f"{model_name}:{model_version}:{language.lower()}:{voice.lower()}:{speed:.2f}:{format_ext.lower()}:{normalized_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class BaseCacheService(ABC):
    """Abstract interface for audio caching."""

    @abstractmethod
    async def get(self, cache_key: str) -> Optional[bytes]:
        ...

    @abstractmethod
    async def set(self, cache_key: str, audio_bytes: bytes, ttl_seconds: int) -> None:
        ...

    @abstractmethod
    async def exists(self, cache_key: str) -> bool:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...


class InMemoryCacheService(BaseCacheService):
    """
    In-memory LRU cache suitable for single-node development or environments without Redis.
    """

    def __init__(self, max_items: int = 500) -> None:
        self._cache: dict[str, tuple[bytes, float]] = {}  # key -> (data, expires_at)
        self.max_items = max_items

    async def get(self, cache_key: str) -> Optional[bytes]:
        entry = self._cache.get(cache_key)
        if not entry:
            CACHE_MISSES_TOTAL.labels(backend="memory").inc()
            return None
        data, expires_at = entry
        if time.time() > expires_at:
            del self._cache[cache_key]
            CACHE_MISSES_TOTAL.labels(backend="memory").inc()
            return None
        CACHE_HITS_TOTAL.labels(backend="memory").inc()
        return data

    async def set(self, cache_key: str, audio_bytes: bytes, ttl_seconds: int) -> None:
        # Evict oldest if full
        if len(self._cache) >= self.max_items:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        expires_at = time.time() + ttl_seconds
        self._cache[cache_key] = (audio_bytes, expires_at)

    async def exists(self, cache_key: str) -> bool:
        return (await self.get(cache_key)) is not None

    async def close(self) -> None:
        self._cache.clear()


class RedisCacheService(BaseCacheService):
    """
    Distributed Redis cache with connection pooling and graceful error degradation.
    """

    def __init__(self, redis_url: str, password: Optional[str] = None) -> None:
        self.redis_url = redis_url
        self.password = password
        self._redis = None
        self._prefix = "tts:cache:"

    async def _get_client(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self.redis_url,
                password=self.password or None,
                decode_responses=False,
                socket_timeout=2.0,
            )
        return self._redis

    async def get(self, cache_key: str) -> Optional[bytes]:
        try:
            client = await self._get_client()
            data = await client.get(f"{self._prefix}{cache_key}")
            if data:
                CACHE_HITS_TOTAL.labels(backend="redis").inc()
                logger.debug("cache.hit", cache_key=cache_key[:12])
                return data
            CACHE_MISSES_TOTAL.labels(backend="redis").inc()
            return None
        except Exception as e:
            logger.warning("cache.redis.error", error=str(e))
            CACHE_MISSES_TOTAL.labels(backend="redis_error").inc()
            return None

    async def set(self, cache_key: str, audio_bytes: bytes, ttl_seconds: int) -> None:
        try:
            client = await self._get_client()
            await client.set(
                f"{self._prefix}{cache_key}",
                audio_bytes,
                ex=ttl_seconds if ttl_seconds > 0 else None,
            )
            logger.debug("cache.saved", cache_key=cache_key[:12], ttl=ttl_seconds)
        except Exception as e:
            logger.warning("cache.redis.set_error", error=str(e))

    async def exists(self, cache_key: str) -> bool:
        try:
            client = await self._get_client()
            return bool(await client.exists(f"{self._prefix}{cache_key}"))
        except Exception:
            return False

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
            self._redis = None


class NoOpCacheService(BaseCacheService):
    """Null object pattern when caching is disabled."""

    async def get(self, cache_key: str) -> Optional[bytes]:
        return None

    async def set(self, cache_key: str, audio_bytes: bytes, ttl_seconds: int) -> None:
        pass

    async def exists(self, cache_key: str) -> bool:
        return False

    async def close(self) -> None:
        pass


def create_cache_service() -> BaseCacheService:
    """
    Factory for instantiating the configured cache service.
    """
    settings = get_settings()
    if not settings.cache_enabled or settings.cache_backend == CacheBackend.NONE:
        return NoOpCacheService()
    if settings.cache_backend == CacheBackend.MEMORY:
        return InMemoryCacheService()
    if settings.cache_backend == CacheBackend.REDIS:
        return RedisCacheService(
            redis_url=settings.redis_url,
            password=settings.redis_password,
        )
    return InMemoryCacheService()
