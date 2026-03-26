from __future__ import annotations

import os
from urllib.parse import quote_plus

from dotenv import load_dotenv


def _build_pg_dsn_from_parts() -> str:
    """Build PostgreSQL DSN from PGVECTOR_* env parts.

    Supported vars:
    - PGVECTOR_HOST (required when PGVECTOR_DSN absent)
    - PGVECTOR_PORT (optional, default 5432)
    - PGVECTOR_DBNAME (required)
    - PGVECTOR_USER (optional)
    - PGVECTOR_PASSWORD (optional)
    - PGVECTOR_SSLMODE (optional)
    """
    host = os.getenv("PGVECTOR_HOST", "").strip()
    dbname = os.getenv("PGVECTOR_DBNAME", "").strip()
    port = os.getenv("PGVECTOR_PORT", "5432").strip() or "5432"
    user = os.getenv("PGVECTOR_USER", "").strip()
    password = os.getenv("PGVECTOR_PASSWORD", "").strip()
    sslmode = os.getenv("PGVECTOR_SSLMODE", "").strip()

    if not host or not dbname:
        return ""

    auth = ""
    if user and password:
        auth = f"{quote_plus(user)}:{quote_plus(password)}@"
    elif user:
        auth = f"{quote_plus(user)}@"

    dsn = f"postgresql://{auth}{host}:{port}/{dbname}"
    if sslmode:
        dsn = f"{dsn}?sslmode={quote_plus(sslmode)}"

    return dsn


def resolve_pg_dsn(explicit_dsn: str = "") -> str:
    """Resolve DSN from explicit arg, then PGVECTOR_DSN, then PGVECTOR_* parts."""
    load_dotenv()

    if explicit_dsn.strip():
        return explicit_dsn.strip()

    dsn = os.getenv("PGVECTOR_DSN", "").strip()
    if dsn:
        return dsn

    return _build_pg_dsn_from_parts()


def resolve_bookshelf_path(explicit_path: str = "") -> str:
    """Resolve bookshelf path from explicit arg or BOOKSHELF_PATH env."""
    load_dotenv()

    if explicit_path.strip():
        return explicit_path.strip()
    return os.getenv("BOOKSHELF_PATH", "").strip()
