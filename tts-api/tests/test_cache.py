"""
tests/test_cache.py

Unit tests for deterministic cache key generation and cache hit behavior.
"""
import pytest
from httpx import AsyncClient

from app.services.cache_service import InMemoryCacheService, generate_cache_key


def test_deterministic_cache_key():
    key1 = generate_cache_key("Hello world", "af_sarah", "en", 1.0, "wav", "kokoro", "0.9.4")
    key2 = generate_cache_key("Hello world", "af_sarah", "en", 1.0, "wav", "kokoro", "0.9.4")
    # Same parameters must yield exact same key
    assert key1 == key2

    # Different text must yield different key
    key3 = generate_cache_key("Different text", "af_sarah", "en", 1.0, "wav", "kokoro", "0.9.4")
    assert key1 != key3

    # Different model version must invalidate old cache key
    key4 = generate_cache_key("Hello world", "af_sarah", "en", 1.0, "wav", "kokoro", "0.9.5")
    assert key1 != key4


@pytest.mark.asyncio
async def test_cache_hit_flag(client: AsyncClient, auth_headers: dict):
    # First request -> Cache Miss
    resp1 = await client.post(
        "/api/v1/tts",
        headers=auth_headers,
        json={"text": "Cache me once", "voice": "af_sarah", "language": "en"},
    )
    assert resp1.status_code == 200
    assert resp1.headers.get("X-Cached") == "false"

    # Second identical request -> Cache Hit
    resp2 = await client.post(
        "/api/v1/tts",
        headers=auth_headers,
        json={"text": "Cache me once", "voice": "af_sarah", "language": "en"},
    )
    assert resp2.status_code == 200
    assert resp2.headers.get("X-Cached") == "true"
