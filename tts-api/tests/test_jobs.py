"""
tests/test_jobs.py

Tests for async job creation, polling status, and job cancellation.
"""
import asyncio
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_async_job_lifecycle(client: AsyncClient, auth_headers: dict):
    # 1. Submit async TTS request
    create_resp = await client.post(
        "/api/v1/tts/async",
        headers=auth_headers,
        json={
            "text": "This is a long batch speech synthesis task processed asynchronously.",
            "voice": "af_sarah",
            "language": "en",
        },
    )
    assert create_resp.status_code == 202
    job_data = create_resp.json()
    assert "job_id" in job_data
    assert job_data["status"] == "queued"
    job_id = job_data["job_id"]

    # 2. Wait a moment for worker execution
    await asyncio.sleep(0.5)

    # 3. Poll job status
    poll_resp = await client.get(
        f"/api/v1/jobs/{job_id}",
        headers=auth_headers,
    )
    assert poll_resp.status_code == 200
    poll_data = poll_resp.json()
    assert poll_data["job_id"] == job_id
    assert poll_data["status"] in ("queued", "processing", "completed")


@pytest.mark.asyncio
async def test_cancel_nonexistent_job(client: AsyncClient, auth_headers: dict):
    response = await client.delete(
        "/api/v1/jobs/job_nonexistent_123",
        headers=auth_headers,
    )
    assert response.status_code == 404
