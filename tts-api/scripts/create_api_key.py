"""
scripts/create_api_key.py

CLI tool to generate and register API keys directly in the database.

Usage:
    python scripts/create_api_key.py --name "Mobile App Client"
    python scripts/create_api_key.py --name "Internal Admin" --admin
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ulid

from app.core.security import generate_api_key, hash_api_key
from app.models.db.session import close_db, get_session_maker, init_db
from app.models.db.tables.api_keys import ApiKey


async def create_key(name: str, is_admin: bool = False, rate_limit: int | None = None) -> str:
    await init_db()
    session_factory = get_session_maker()

    raw_key = generate_api_key()
    key_id = str(ulid.new())
    key_prefix = raw_key[:16]
    hashed = hash_api_key(raw_key)

    async with session_factory() as session:
        record = ApiKey(
            id=key_id,
            name=name,
            key_prefix=key_prefix,
            key_hash=hashed,
            is_active=True,
            is_admin=is_admin,
            rate_limit_override=rate_limit,
        )
        session.add(record)
        await session.commit()

    await close_db()
    return raw_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a secure API key for the TTS API")
    parser.add_argument("--name", required=True, help="Client or application name")
    parser.add_argument("--admin", action="store_true", help="Grant administrator privileges")
    parser.add_argument("--rate-limit", type=int, default=None, help="Custom rate limit (requests/min)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Creating New TTS API Key")
    print("=" * 60)

    raw_key = asyncio.run(create_key(name=args.name, is_admin=args.admin, rate_limit=args.rate_limit))

    print(f"\n  Client Name:  {args.name}")
    print(f"  Admin Access: {args.admin}")
    print(f"  API Key:      {raw_key}")
    print("\n  ⚠️ IMPORTANT: Copy this key now. It is hashed in the database and cannot be recovered.")
    print("=" * 60)


if __name__ == "__main__":
    main()
