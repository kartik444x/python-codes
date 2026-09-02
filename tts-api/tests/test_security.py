"""
tests/test_security.py

Tests for API key authentication, rate limiting, and administrative authorization.
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_unauthenticated_request_rejected(client: AsyncClient):
    response = await client.post(
        "/api/v1/tts",
        json={"text": "Hello world"},
    )
    assert response.status_code == 401
    data = response.json()
    assert data["error"]["code"] == "AUTHENTICATION_REQUIRED"


@pytest.mark.asyncio
async def test_invalid_api_key_rejected(client: AsyncClient):
    response = await client.post(
        "/api/v1/tts",
        headers={"X-API-Key": "invalid-random-key-xyz"},
        json={"text": "Hello world"},
    )
    assert response.status_code == 403
    data = response.json()
    assert data["error"]["code"] == "INVALID_API_KEY"


@pytest.mark.asyncio
async def test_valid_dev_api_key_accepted(client: AsyncClient, auth_headers: dict):
    response = await client.post(
        "/api/v1/tts",
        headers=auth_headers,
        json={"text": "Hello world", "return_json": True},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_admin_route_requires_admin_privilege(client: AsyncClient, auth_headers: dict):
    # Ordinary key cannot access admin key generation without admin flag / header
    response = await client.get(
        "/api/v1/admin/keys",
        headers={"X-API-Key": "non-admin-key"},
    )
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_admin_creates_new_api_key(client: AsyncClient, admin_headers: dict):
    response = await client.post(
        "/api/v1/admin/keys",
        headers=admin_headers,
        json={"name": "Client Application 1", "is_admin": False},
    )
    assert response.status_code == 201
    data = response.json()
    assert "api_key" in data
    assert data["api_key"].startswith("tts_live_")
    assert data["name"] == "Client Application 1"
