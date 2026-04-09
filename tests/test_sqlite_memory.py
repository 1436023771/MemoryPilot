from pathlib import Path

import numpy as np
import pytest

from app.memory.embeddings import EmbeddingManager
from app.memory.read_only_memory import retrieve_memory_context
from app.memory.sqlite_memory import (
    load_embeddings_count,
    load_memory_chunks_from_sqlite,
    retrieve_memory_context_hybrid_from_sqlite,
    write_facts_to_sqlite,
)
from app.memory.write_memory import MemoryFact


class _FakeSentenceTransformer:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_sentence_embedding_dimension(self) -> int:
        return 8

    def encode(self, text: str, normalize_embeddings: bool = True):
        vec = np.zeros(8, dtype=np.float32)
        for ch in str(text or ""):
            vec[ord(ch) % 8] += 1.0
        norm = float(np.linalg.norm(vec))
        if normalize_embeddings and norm > 0:
            vec = vec / norm
        return vec


@pytest.fixture(autouse=True)
def _patch_embedding_model(monkeypatch):
    monkeypatch.setattr(EmbeddingManager, "_load_model", staticmethod(lambda _name: _FakeSentenceTransformer(_name)))


def _fact(key: str, value: str, text: str) -> MemoryFact:
    return MemoryFact(key=key, value=value, text=text)


def test_sqlite_write_and_load_chunks(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    written = write_facts_to_sqlite(
        db_path,
        [
            _fact("name", "小李", "用户姓名：小李。"),
            _fact("like", "Python", "用户喜欢：Python。"),
        ],
    )

    chunks = load_memory_chunks_from_sqlite(db_path)

    assert "用户姓名：小李。" in written
    assert "用户喜欢：Python。" in written
    assert len(chunks) == 2


def test_sqlite_singleton_update(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    write_facts_to_sqlite(db_path, [_fact("name", "小李", "用户姓名：小李。")])
    written = write_facts_to_sqlite(db_path, [_fact("name", "李华", "用户姓名：李华。")])
    chunks = load_memory_chunks_from_sqlite(db_path)

    texts = [chunk.text for chunk in chunks]
    assert written == ["用户姓名：李华。"]
    assert "用户姓名：小李。" not in texts
    assert "用户姓名：李华。" in texts


def test_sqlite_retrieval_context(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    write_facts_to_sqlite(
        db_path,
        [
            _fact("name", "李华", "用户姓名：李华。"),
            _fact("like", "简洁回答", "用户喜欢：简洁回答。"),
        ],
    )
    chunks = load_memory_chunks_from_sqlite(db_path)

    context = retrieve_memory_context("姓名", chunks, top_k=2)

    assert "李华" in context


def test_sqlite_embeddings_synced_with_facts(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    write_facts_to_sqlite(
        db_path,
        [
            _fact("name", "小李", "用户姓名：小李。"),
            _fact("like", "Python", "用户喜欢：Python。"),
        ],
    )

    assert load_embeddings_count(db_path) == 2


def test_sqlite_embeddings_rebuild_on_singleton_update(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    write_facts_to_sqlite(db_path, [_fact("name", "小李", "用户姓名：小李。")])
    assert load_embeddings_count(db_path) == 1

    write_facts_to_sqlite(db_path, [_fact("name", "李华", "用户姓名：李华。")])
    assert load_embeddings_count(db_path) == 1


def test_sqlite_hybrid_retrieval_returns_name(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    write_facts_to_sqlite(
        db_path,
        [
            _fact("name", "王小明", "用户姓名：王小明。"),
            _fact("like", "简洁回答", "用户喜欢：简洁回答。"),
            _fact("goal", "成为AI工程师", "用户目标：成为AI工程师。"),
        ],
    )

    context = retrieve_memory_context_hybrid_from_sqlite(db_path, "我叫什么名字", top_k=2)

    assert "王小明" in context


def test_sqlite_hybrid_retrieval_handles_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    context = retrieve_memory_context_hybrid_from_sqlite(db_path, "我叫什么名字", top_k=2)

    assert context == ""
