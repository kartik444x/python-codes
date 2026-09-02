"""
tests/test_tts.py

Tests for synchronous TTS generation, validation rules, output formats, and headers.
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_tts_successful_wav_generation(client: AsyncClient, auth_headers: dict):
    response = await client.post(
        "/api/v1/tts",
        headers=auth_headers,
        json={
            "text": "Hello world, this is a test.",
            "voice": "af_sarah",
            "language": "en",
            "speed": 1.0,
            "format": "wav",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert "X-Audio-ID" in response.headers
    assert "X-Audio-Duration" in response.headers
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_tts_json_mode(client: AsyncClient, auth_headers: dict):
    response = await client.post(
        "/api/v1/tts",
        headers=auth_headers,
        json={
            "text": "Testing JSON response mode.",
            "voice": "af_sarah",
            "language": "en",
            "return_json": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "audio_id" in data
    assert data["audio_url"].startswith("/api/v1/audio/")
    assert data["duration_seconds"] > 0
    assert data["format"] == "wav"


@pytest.mark.asyncio
async def test_tts_empty_text_rejection(client: AsyncClient, auth_headers: dict):
    response = await client.post(
        "/api/v1/tts",
        headers=auth_headers,
        json={"text": "   ", "voice": "af_sarah"},
    )
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_tts_invalid_speed_rejection(client: AsyncClient, auth_headers: dict):
    response = await client.post(
        "/api/v1/tts",
        headers=auth_headers,
        json={"text": "Test speed", "speed": 5.0},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_tts_unsupported_voice_rejection(client: AsyncClient, auth_headers: dict):
    response = await client.post(
        "/api/v1/tts",
        headers=auth_headers,
        json={"text": "Test voice", "voice": "non_existent_voice_999"},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["error"]["code"] == "UNSUPPORTED_VOICE"
