from pathlib import Path

from app.read_only_memory import retrieve_memory_context
from app.sqlite_memory import (
    load_embeddings_count,
    load_memory_chunks_from_sqlite,
    retrieve_memory_context_hybrid_from_sqlite,
    write_facts_to_sqlite,
)


def test_sqlite_write_and_load_chunks(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    written = write_facts_to_sqlite(
        db_path,
        ["用户姓名：小李；我是小李。", "用户喜欢：Python；我喜欢Python。"],
    )

    chunks = load_memory_chunks_from_sqlite(db_path)

    assert "用户姓名：小李；我是小李。" in written
    assert "用户喜欢：Python；我喜欢Python。" in written
    assert len(chunks) == 2


def test_sqlite_singleton_update(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    write_facts_to_sqlite(db_path, ["用户姓名：小李；我是小李。"])
    written = write_facts_to_sqlite(db_path, ["用户姓名：李华；我是李华。"])
    chunks = load_memory_chunks_from_sqlite(db_path)

    texts = [chunk.text for chunk in chunks]
    assert written == ["用户姓名：李华；我是李华。"]
    assert "用户姓名：小李；我是小李。" not in texts
    assert "用户姓名：李华；我是李华。" in texts


def test_sqlite_retrieval_context(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    write_facts_to_sqlite(
        db_path,
        [
            "用户姓名：李华；我是李华。",
            "用户喜欢：简洁回答；我喜欢简洁回答。",
        ],
    )
    chunks = load_memory_chunks_from_sqlite(db_path)

    context = retrieve_memory_context("我叫什么", chunks, top_k=2)

    assert "李华" in context


def test_sqlite_embeddings_synced_with_facts(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    write_facts_to_sqlite(
        db_path,
        [
            "用户姓名：小李；我是小李。",
            "用户喜欢：Python；我喜欢Python。",
        ],
    )

    assert load_embeddings_count(db_path) == 2


def test_sqlite_embeddings_rebuild_on_singleton_update(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    write_facts_to_sqlite(db_path, ["用户姓名：小李；我是小李。"])
    assert load_embeddings_count(db_path) == 1

    write_facts_to_sqlite(db_path, ["用户姓名：李华；我是李华。"])
    assert load_embeddings_count(db_path) == 1


def test_sqlite_hybrid_retrieval_returns_name(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    write_facts_to_sqlite(
        db_path,
        [
            "用户姓名：王小明；我是王小明。",
            "用户喜欢：简洁回答；我喜欢简洁回答。",
            "用户目标：成为AI工程师；我的目标是成为AI工程师。",
        ],
    )

    context = retrieve_memory_context_hybrid_from_sqlite(db_path, "我叫什么名字", top_k=2)

    assert "王小明" in context


def test_sqlite_hybrid_retrieval_handles_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    context = retrieve_memory_context_hybrid_from_sqlite(db_path, "我叫什么名字", top_k=2)

    assert context == ""
