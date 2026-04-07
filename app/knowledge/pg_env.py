from __future__ import annotations

from urllib.parse import quote_plus

from app.config import get_env_str


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
    host = get_env_str("PGVECTOR_HOST", "")
    dbname = get_env_str("PGVECTOR_DBNAME", "")
    port = get_env_str("PGVECTOR_PORT", "5432") or "5432"
    user = get_env_str("PGVECTOR_USER", "")
    password = get_env_str("PGVECTOR_PASSWORD", "")
    sslmode = get_env_str("PGVECTOR_SSLMODE", "")

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
    if explicit_dsn.strip():
        return explicit_dsn.strip()

    dsn = get_env_str("PGVECTOR_DSN", "")
    if dsn:
        return dsn

    return _build_pg_dsn_from_parts()


def resolve_bookshelf_path(explicit_path: str = "") -> str:
    """Resolve bookshelf path from explicit arg or BOOKSHELF_PATH env."""
    if explicit_path.strip():
        return explicit_path.strip()
    return get_env_str("BOOKSHELF_PATH", "")
