"""
app/api/v1/routes/jobs.py

Job tracking endpoints for async TTS requests:
- GET /api/v1/jobs/{job_id}: Poll job status and retrieve generated audio URL upon completion.
- DELETE /api/v1/jobs/{job_id}: Cancel a queued or running job.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, status

from app.api.deps import get_job_service, verify_api_key
from app.models.db.tables.api_keys import ApiKey
from app.models.responses import JobStatusResponse
from app.services.job_service import JobService

router = APIRouter(prefix="/jobs", tags=["Async Jobs"])


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get Job Status",
    description="Poll the status of an asynchronous TTS synthesis job.",
)
async def get_job_status(
    job_id: str,
    api_key_record: Optional[ApiKey] = Depends(verify_api_key),
    job_service: JobService = Depends(get_job_service),
) -> JobStatusResponse:
    return await job_service.get_job_status(job_id)


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel Job",
    description="Cancel a queued or running async TTS job.",
)
async def cancel_job(
    job_id: str,
    api_key_record: Optional[ApiKey] = Depends(verify_api_key),
    job_service: JobService = Depends(get_job_service),
) -> dict:
    cancelled = await job_service.cancel_job(job_id)
    return {
        "job_id": job_id,
        "cancelled": cancelled,
        "status": "cancelled" if cancelled else "unable_to_cancel",
    }
