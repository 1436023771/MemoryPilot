from pathlib import Path

from app.knowledge import chunking
from app.knowledge.chunking import split_text


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
