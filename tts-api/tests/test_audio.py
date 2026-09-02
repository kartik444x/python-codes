"""
tests/test_audio.py

Tests for audio file retrieval, download headers, and 404 handling.
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_audio_retrieval_and_stream(client: AsyncClient, auth_headers: dict):
    # 1. Synthesize audio in JSON mode to obtain audio_id
    gen_resp = await client.post(
        "/api/v1/tts",
        headers=auth_headers,
        json={"text": "Hello audio retrieval", "return_json": True},
    )
    assert gen_resp.status_code == 200
    audio_id = gen_resp.json()["audio_id"]

    # 2. Retrieve audio binary via GET /api/v1/audio/{audio_id}
    audio_resp = await client.get(
        f"/api/v1/audio/{audio_id}?format=wav",
        headers=auth_headers,
    )
    assert audio_resp.status_code == 200
    assert audio_resp.headers["content-type"] == "audio/wav"
    assert "inline" in audio_resp.headers["content-disposition"]
    assert len(audio_resp.content) > 0


@pytest.mark.asyncio
async def test_audio_retrieval_download_attachment_header(client: AsyncClient, auth_headers: dict):
    gen_resp = await client.post(
        "/api/v1/tts",
        headers=auth_headers,
        json={"text": "Download test", "return_json": True},
    )
    audio_id = gen_resp.json()["audio_id"]

    audio_resp = await client.get(
        f"/api/v1/audio/{audio_id}?download=true",
        headers=auth_headers,
    )
    assert audio_resp.status_code == 200
    assert "attachment" in audio_resp.headers["content-disposition"]


@pytest.mark.asyncio
async def test_audio_not_found(client: AsyncClient, auth_headers: dict):
    response = await client.get(
        "/api/v1/audio/nonexistent_audio_id_999",
        headers=auth_headers,
    )
    assert response.status_code == 404
