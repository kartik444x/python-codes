"""
app/services/storage_service.py

Audio storage abstraction supporting:
- LocalStorageBackend: Stores generated audio files on the local filesystem with automated lifecycle retention.
- S3StorageBackend: Stores generated audio files on Amazon S3, MinIO, or Cloudflare R2.
- StorageService: High-level orchestrator providing unified interface (save, get, delete, exists, cleanup).
"""
from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Optional, Tuple

import aiofiles
import aiofiles.os

from app.core.config import StorageBackend, get_settings
from app.core.exceptions import NotFoundError, StorageError
from app.core.logging import get_logger
from app.core.metrics import STORAGE_OPERATIONS_TOTAL

logger = get_logger(__name__)


class BaseStorageBackend(ABC):
    """Abstract interface for audio storage backends."""

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Name of storage backend."""
        ...

    @abstractmethod
    async def save(self, audio_id: str, data: bytes, format_ext: str = "wav") -> str:
        """
        Persist audio bytes. Returns the relative or absolute storage key/URI.
        """
        ...

    @abstractmethod
    async def get(self, audio_id: str, format_ext: str = "wav") -> bytes:
        """
        Retrieve audio bytes by audio_id.
        """
        ...

    @abstractmethod
    async def get_path_or_url(self, audio_id: str, format_ext: str = "wav") -> str:
        """
        Get local filepath or public URL for the audio asset.
        """
        ...

    @abstractmethod
    async def delete(self, audio_id: str, format_ext: str = "wav") -> bool:
        """
        Delete audio file. Returns True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def exists(self, audio_id: str, format_ext: str = "wav") -> bool:
        """
        Check if audio asset exists.
        """
        ...

    @abstractmethod
    async def cleanup_expired(self, max_age_hours: int) -> int:
        """
        Purge audio files older than max_age_hours. Returns count of deleted files.
        """
        ...


class LocalStorageBackend(BaseStorageBackend):
    """
    Local filesystem storage with non-blocking async file I/O and directory sandboxing.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def backend_type(self) -> str:
        return "local"

    def _resolve_path(self, audio_id: str, format_ext: str) -> Path:
        # Sanitize against path traversal attacks
        clean_id = "".join(c for c in audio_id if c.isalnum() or c in ("-", "_"))
        clean_ext = "".join(c for c in format_ext if c.isalnum()).lower()
        if not clean_id:
            raise StorageError("Invalid audio ID for file storage.")
        file_name = f"{clean_id}.{clean_ext}"
        target_path = (self.base_dir / file_name).resolve()
        # Ensure path is inside base_dir
        if not str(target_path).startswith(str(self.base_dir.resolve())):
            raise StorageError("Path traversal attempt detected.")
        return target_path

    async def save(self, audio_id: str, data: bytes, format_ext: str = "wav") -> str:
        target_path = self._resolve_path(audio_id, format_ext)
        try:
            async with aiofiles.open(target_path, "wb") as f:
                await f.write(data)
            STORAGE_OPERATIONS_TOTAL.labels(operation="save", backend="local", status="success").inc()
            return str(target_path)
        except Exception as e:
            STORAGE_OPERATIONS_TOTAL.labels(operation="save", backend="local", status="error").inc()
            logger.error("storage.local.save_error", audio_id=audio_id, error=str(e))
            raise StorageError(f"Failed to save audio file: {e}") from e

    async def get(self, audio_id: str, format_ext: str = "wav") -> bytes:
        target_path = self._resolve_path(audio_id, format_ext)
        if not await aiofiles.os.path.exists(target_path):
            STORAGE_OPERATIONS_TOTAL.labels(operation="get", backend="local", status="not_found").inc()
            raise NotFoundError("Audio", audio_id)
        try:
            async with aiofiles.open(target_path, "rb") as f:
                content = await f.read()
            STORAGE_OPERATIONS_TOTAL.labels(operation="get", backend="local", status="success").inc()
            return content
        except NotFoundError:
            raise
        except Exception as e:
            STORAGE_OPERATIONS_TOTAL.labels(operation="get", backend="local", status="error").inc()
            logger.error("storage.local.get_error", audio_id=audio_id, error=str(e))
            raise StorageError(f"Failed to read audio file: {e}") from e

    async def get_path_or_url(self, audio_id: str, format_ext: str = "wav") -> str:
        target_path = self._resolve_path(audio_id, format_ext)
        if not await aiofiles.os.path.exists(target_path):
            raise NotFoundError("Audio", audio_id)
        return str(target_path)

    async def delete(self, audio_id: str, format_ext: str = "wav") -> bool:
        target_path = self._resolve_path(audio_id, format_ext)
        if await aiofiles.os.path.exists(target_path):
            try:
                await aiofiles.os.remove(target_path)
                STORAGE_OPERATIONS_TOTAL.labels(operation="delete", backend="local", status="success").inc()
                return True
            except Exception as e:
                STORAGE_OPERATIONS_TOTAL.labels(operation="delete", backend="local", status="error").inc()
                logger.error("storage.local.delete_error", audio_id=audio_id, error=str(e))
                raise StorageError(f"Failed to delete audio file: {e}") from e
        return False

    async def exists(self, audio_id: str, format_ext: str = "wav") -> bool:
        target_path = self._resolve_path(audio_id, format_ext)
        return await aiofiles.os.path.exists(target_path)

    async def cleanup_expired(self, max_age_hours: int) -> int:
        cutoff = time.time() - (max_age_hours * 3600)
        deleted_count = 0
        try:
            for item in self.base_dir.iterdir():
                if item.is_file() and not item.name.startswith("."):
                    stat = item.stat()
                    if stat.st_mtime < cutoff:
                        item.unlink(missing_ok=True)
                        deleted_count += 1
            if deleted_count > 0:
                logger.info("storage.cleanup.completed", deleted_count=deleted_count, max_age_hours=max_age_hours)
            return deleted_count
        except Exception as e:
            logger.error("storage.cleanup.error", error=str(e))
            return 0


class S3StorageBackend(BaseStorageBackend):
    """
    S3-compatible storage backend (AWS S3, MinIO, Wasabi, Cloudflare R2).
    """

    def __init__(
        self,
        bucket: str,
        endpoint: str = "",
        region: str = "us-east-1",
        access_key: str = "",
        secret_key: str = "",
        public_url: str = "",
    ) -> None:
        self.bucket = bucket
        self.endpoint = endpoint
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key
        self.public_url = public_url

    @property
    def backend_type(self) -> str:
        return "s3"

    async def save(self, audio_id: str, data: bytes, format_ext: str = "wav") -> str:
        # boto3 async wrapper
        import boto3
        from botocore.client import Config
        s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint or None,
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
        )
        key = f"audio/{audio_id}.{format_ext}"
        content_type = "audio/wav" if format_ext == "wav" else "audio/mpeg"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
        )
        STORAGE_OPERATIONS_TOTAL.labels(operation="save", backend="s3", status="success").inc()
        return self.public_url.rstrip("/") + f"/{key}" if self.public_url else key

    async def get(self, audio_id: str, format_ext: str = "wav") -> bytes:
        import boto3
        from botocore.exceptions import ClientError
        s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint or None,
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
        key = f"audio/{audio_id}.{format_ext}"
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: s3.get_object(Bucket=self.bucket, Key=key)
            )
            data = await loop.run_in_executor(None, lambda: resp["Body"].read())
            STORAGE_OPERATIONS_TOTAL.labels(operation="get", backend="s3", status="success").inc()
            return data
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise NotFoundError("Audio", audio_id)
            raise StorageError(f"S3 get failed: {e}") from e

    async def get_path_or_url(self, audio_id: str, format_ext: str = "wav") -> str:
        key = f"audio/{audio_id}.{format_ext}"
        if self.public_url:
            return f"{self.public_url.rstrip('/')}/{key}"
        return f"s3://{self.bucket}/{key}"

    async def delete(self, audio_id: str, format_ext: str = "wav") -> bool:
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint or None,
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
        key = f"audio/{audio_id}.{format_ext}"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: s3.delete_object(Bucket=self.bucket, Key=key)
        )
        return True

    async def exists(self, audio_id: str, format_ext: str = "wav") -> bool:
        import boto3
        from botocore.exceptions import ClientError
        s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint or None,
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
        key = f"audio/{audio_id}.{format_ext}"
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, lambda: s3.head_object(Bucket=self.bucket, Key=key))
            return True
        except ClientError:
            return False

    async def cleanup_expired(self, max_age_hours: int) -> int:
        # In production S3, object lifecycle rules handle automated expiration
        return 0


class StorageService:
    """
    Unified storage service initialized from app settings.
    """

    def __init__(self) -> None:
        settings = get_settings()
        if settings.storage_backend == StorageBackend.S3 and settings.s3_bucket:
            self.backend: BaseStorageBackend = S3StorageBackend(
                bucket=settings.s3_bucket,
                endpoint=settings.s3_endpoint,
                region=settings.s3_region,
                access_key=settings.s3_access_key,
                secret_key=settings.s3_secret_key,
                public_url=settings.s3_public_url,
            )
        else:
            self.backend = LocalStorageBackend(base_dir=settings.audio_dir)

    async def save_audio(self, audio_id: str, data: bytes, format_ext: str = "wav") -> str:
        return await self.backend.save(audio_id, data, format_ext)

    async def get_audio(self, audio_id: str, format_ext: str = "wav") -> bytes:
        return await self.backend.get(audio_id, format_ext)

    async def get_audio_path_or_url(self, audio_id: str, format_ext: str = "wav") -> str:
        return await self.backend.get_path_or_url(audio_id, format_ext)

    async def delete_audio(self, audio_id: str, format_ext: str = "wav") -> bool:
        return await self.backend.delete(audio_id, format_ext)

    async def audio_exists(self, audio_id: str, format_ext: str = "wav") -> bool:
        return await self.backend.exists(audio_id, format_ext)

    async def run_cleanup(self) -> int:
        settings = get_settings()
        return await self.backend.cleanup_expired(settings.audio_retention_hours)
