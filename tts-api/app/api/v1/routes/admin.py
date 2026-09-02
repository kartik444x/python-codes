"""
app/api/v1/routes/admin.py

Administrative endpoints for managing API keys, inspecting usage, and triggering storage cleanup.
Protected by X-Admin-Key or admin credentials.
"""
from __future__ import annotations

from typing import List, Optional

import ulid
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_storage_service, require_admin_key
from app.core.security import generate_api_key, hash_api_key
from app.models.db.session import get_db_session
from app.models.db.tables.api_keys import ApiKey
from app.models.db.tables.usage import UsageRecord
from app.models.responses import APIKeyCreateResponse, APIKeyInfoResponse
from app.services.storage_service import StorageService

router = APIRouter(prefix="/admin", tags=["Admin Operations"], dependencies=[Depends(require_admin_key)])


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=128, description="Identifier / client description")
    is_admin: bool = Field(False, description="Whether this key has admin privileges")
    rate_limit_override: Optional[int] = Field(None, description="Custom rate limit for this key (requests/min)")


@router.post(
    "/keys",
    response_model=APIKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create API Key",
    description="Generate a new API key. Plaintext key is shown once and never stored.",
)
async def create_new_api_key(
    payload: CreateKeyRequest,
    db: AsyncSession = Depends(get_db_session),
) -> APIKeyCreateResponse:
    raw_key = generate_api_key()
    key_id = str(ulid.new())
    key_prefix = raw_key[:16]
    hashed_key = hash_api_key(raw_key)

    api_key_record = ApiKey(
        id=key_id,
        name=payload.name,
        key_prefix=key_prefix,
        key_hash=hashed_key,
        is_active=True,
        is_admin=payload.is_admin,
        rate_limit_override=payload.rate_limit_override,
    )
    db.add(api_key_record)
    await db.commit()
    await db.refresh(api_key_record)

    return APIKeyCreateResponse(
        key_id=api_key_record.id,
        api_key=raw_key,
        name=api_key_record.name,
        created_at=api_key_record.created_at,
        message="Store this key securely. It will not be displayed again.",
    )


@router.get(
    "/keys",
    response_model=List[APIKeyInfoResponse],
    summary="List API Keys",
    description="List all active/revoked API keys without exposing their secrets.",
)
async def list_api_keys(
    db: AsyncSession = Depends(get_db_session),
) -> List[APIKeyInfoResponse]:
    result = await db.execute(select(ApiKey).order_by(ApiKey.created_at.desc()))
    records = result.scalars().all()
    return [
        APIKeyInfoResponse(
            key_id=r.id,
            name=r.name,
            is_active=r.is_active,
            created_at=r.created_at,
            last_used_at=r.last_used_at,
            request_count=r.request_count,
        )
        for r in records
    ]


@router.delete(
    "/keys/{key_id}",
    summary="Revoke API Key",
    description="Revoke an API key immediately.",
)
async def revoke_api_key(
    key_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await db.execute(select(ApiKey).where(ApiKey.id == key_id))
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail=f"API key '{key_id}' not found.")
    record.is_active = False
    await db.commit()
    return {"key_id": key_id, "status": "revoked"}


@router.post(
    "/cleanup",
    summary="Trigger Storage Retention Cleanup",
    description="Purge generated audio files older than the configured AUDIO_RETENTION_HOURS.",
)
async def trigger_cleanup(
    storage_service: StorageService = Depends(get_storage_service),
) -> dict:
    deleted_count = await storage_service.run_cleanup()
    return {"status": "cleanup_completed", "deleted_files": deleted_count}
