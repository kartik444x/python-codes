"""
tests/test_voices.py

Tests for voice catalog query, filtering by language and gender, and voice registration.
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_all_voices(client: AsyncClient):
    response = await client.get("/api/v1/voices")
    assert response.status_code == 200
    data = response.json()
    assert "voices" in data
    assert data["total"] >= 4
    assert any(v["id"] == "af_sarah" for v in data["voices"])


@pytest.mark.asyncio
async def test_filter_voices_by_gender(client: AsyncClient):
    response = await client.get("/api/v1/voices?gender=female")
    assert response.status_code == 200
    data = response.json()
    assert all(v["gender"] == "female" for v in data["voices"])


@pytest.mark.asyncio
async def test_filter_voices_by_language(client: AsyncClient):
    response = await client.get("/api/v1/voices?language=es")
    assert response.status_code == 200
    data = response.json()
    assert all("es" in v["language_code"].lower() for v in data["voices"])


@pytest.mark.asyncio
async def test_get_single_voice_details(client: AsyncClient):
    response = await client.get("/api/v1/voices/af_sarah")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "af_sarah"
    assert data["is_default"] is True


@pytest.mark.asyncio
async def test_voice_registration_requires_consent(client: AsyncClient, auth_headers: dict):
    response = await client.post(
        "/api/v1/voices",
        headers=auth_headers,
        json={
            "voice_name": "Test Actor",
            "language_code": "en-US",
            "consent_confirmed": False,
        },
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "consent" in data["error"]["message"].lower()
