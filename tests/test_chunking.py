from pathlib import Path

from app.knowledge import chunking
from app.knowledge.chunking import split_document_text, split_epub_text, split_text


def test_split_text_basic() -> None:
    text = "a" * 1000
    chunks = split_text(text, chunk_size=300, overlap=100)

    assert len(chunks) >= 4
    assert all(len(c) <= 300 for c in chunks)


def test_split_text_validates_overlap() -> None:
    try:
        split_text("hello", chunk_size=10, overlap=10)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_load_text_documents_supports_epub(monkeypatch, tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"dummy")

    monkeypatch.setattr(chunking, "_read_epub", lambda _p: "EPUB chapter content")

    docs = chunking.load_text_documents(tmp_path)

    assert len(docs) == 1
    assert docs[0].path.endswith("book.epub")
    assert "EPUB chapter content" in docs[0].text


def test_split_epub_text_prefers_paragraph_boundaries() -> None:
    p1 = "第一段内容" * 8
    p2 = "第二段内容" * 8
    p3 = "第三段内容" * 8
    text = f"{p1}\n\n{p2}\n\n{p3}"

    chunks = split_epub_text(text, chunk_size=80, overlap=20)

    assert p1 in chunks
    assert p2 in chunks
    assert p3 in chunks
    assert all(len(chunk) <= 80 for chunk in chunks)


def test_split_document_text_keeps_non_epub_strategy() -> None:
    text = "a" * 1000
    chunks = split_document_text(text, path="notes.md", chunk_size=300, overlap=100)

    assert len(chunks) >= 4
    assert all(len(c) <= 300 for c in chunks)
