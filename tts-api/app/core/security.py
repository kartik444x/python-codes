"""
app/core/security.py

API key generation, secure hashing, and validation utilities.
In production, API keys are hashed with bcrypt or SHA-256 before storing.
Never store or log raw API keys.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Optional

from passlib.context import CryptContext

from app.core.config import get_settings

# CryptContext for password / API key hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def generate_api_key(prefix: str = "tts_live_") -> str:
    """
    Generate a secure random API key.
    Format: tts_live_<32 random URL-safe hex characters>
    """
    random_bytes = secrets.token_hex(16)
    return f"{prefix}{random_bytes}"


def hash_api_key(api_key: str) -> str:
    """
    Compute a SHA-256 hash of an API key for fast, deterministic database lookup
    or bcrypt for salted storage.
    """
    settings = get_settings()
    # Use HMAC with application secret key for key fingerprinting
    return hmac.new(
        settings.secret_key.encode("utf-8"),
        api_key.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def verify_api_key_hash(plain_key: str, hashed_key: str) -> bool:
    """
    Constant-time comparison to prevent timing attacks.
    """
    calculated_hash = hash_api_key(plain_key)
    return hmac.compare_digest(calculated_hash, hashed_key)


def validate_dev_api_key(api_key: str) -> bool:
    """
    Check if the key is valid in local development/test mode.
    """
    settings = get_settings()
    if not settings.api_key_required:
        return True

    if not api_key:
        return False

    dev_keys = settings.dev_api_keys_list
    if dev_keys:
        for k in dev_keys:
            if hmac.compare_digest(k, api_key):
                return True

    # Check against admin api key
    if settings.admin_api_key and hmac.compare_digest(settings.admin_api_key, api_key):
        return True

    return False
