import sys
from types import ModuleType

from app.knowledge.pgvector_store import PgVectorKnowledgeStore


class _DummyVector:
    def __init__(self, values):
        self.values = list(values)


class _FakeCursor:
    def __init__(self):
        self.sql = ""
        self.params = []
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params):
        self.sql = sql
        self.params = list(params)
        if "DELETE FROM" in sql:
            self.rowcount = 3

    def fetchone(self):
        if "COUNT(*)" in self.sql:
            return [5]
        return None

    def fetchall(self):
        return []


class _FakeConn:
    def __init__(self):
        self._cursor = _FakeCursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self._cursor

    def commit(self):
        return None


def test_similarity_search_sql_params_order_with_filters(monkeypatch) -> None:
    fake_pgvector = ModuleType("pgvector")
    fake_pgvector.Vector = _DummyVector
    monkeypatch.setitem(sys.modules, "pgvector", fake_pgvector)

    store = PgVectorKnowledgeStore(dsn="postgresql://demo", embedding_dim=3)
    fake_conn = _FakeConn()
    monkeypatch.setattr(store, "_connect", lambda: fake_conn)

    store.similarity_search(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=4,
        document_id="doc-1",
        book_id="book-1",
        chapter="chapter-1",
    )

    params = fake_conn._cursor.params
    assert len(params) == 6

    # First placeholder in SELECT score must bind query vector.
    assert isinstance(params[0], _DummyVector)
    assert params[0].values == [0.1, 0.2, 0.3]

    # WHERE placeholders follow in order: document_id, book_id, chapter.
    assert params[1] == "doc-1"
    assert params[2] == "book-1"
    assert params[3] == "chapter-1"

    # ORDER BY vector then LIMIT.
    assert isinstance(params[4], _DummyVector)
    assert params[4].values == [0.1, 0.2, 0.3]
    assert params[5] == 4


def test_delete_and_count_by_document_id(monkeypatch) -> None:
    store = PgVectorKnowledgeStore(dsn="postgresql://demo", embedding_dim=3)
    fake_conn = _FakeConn()
    monkeypatch.setattr(store, "_connect", lambda: fake_conn)

    deleted = store.delete_by_document_id("doc-1")
    assert deleted == 3
    assert "DELETE FROM" in fake_conn._cursor.sql
    assert fake_conn._cursor.params == ["doc-1"]

    count = store.count_by_document_id("doc-1")
    assert count == 5
    assert "COUNT(*)" in fake_conn._cursor.sql
    assert fake_conn._cursor.params == ["doc-1"]
