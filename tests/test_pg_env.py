from app.knowledge.pg_env import resolve_bookshelf_path, resolve_pg_dsn


def test_resolve_bookshelf_path_from_env(monkeypatch) -> None:
    monkeypatch.setenv("BOOKSHELF_PATH", "/tmp/bookshelf")
    resolved = resolve_bookshelf_path("")
    assert resolved == "/tmp/bookshelf"


def test_resolve_pg_dsn_from_split_env(monkeypatch) -> None:
    monkeypatch.delenv("PGVECTOR_DSN", raising=False)
    monkeypatch.setenv("PGVECTOR_HOST", "127.0.0.1")
    monkeypatch.setenv("PGVECTOR_PORT", "5433")
    monkeypatch.setenv("PGVECTOR_DBNAME", "agent_db")
    monkeypatch.setenv("PGVECTOR_USER", "tester")
    monkeypatch.setenv("PGVECTOR_PASSWORD", "secret")

    dsn = resolve_pg_dsn("")
    assert dsn == "postgresql://tester:secret@127.0.0.1:5433/agent_db"
