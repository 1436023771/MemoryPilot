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


def test_split_epub_text_respects_chapter_break_marker() -> None:
    chapter_a = "第1话\n\n" + ("A段落内容。" * 80)
    chapter_b = "第2话\n\n" + ("B段落内容。" * 80)
    text = chapter_a + "\n\n[[EPUB_CHAPTER_BREAK]]\n\n" + chapter_b

    chunks = split_epub_text(text, chunk_size=220, overlap=80)

    assert chunks
    # Ensure no chunk mixes chapter A and chapter B content together.
    assert all(not ("A段落内容" in c and "B段落内容" in c) for c in chunks)


def test_split_epub_text_handles_single_chapter_marker_only() -> None:
    text = "第1话\n\n正文段落一。\n\n正文段落二。"
    chunks = split_epub_text(text, chunk_size=60, overlap=20)

    assert len(chunks) >= 1
    assert all("[[EPUB_CHAPTER_BREAK]]" not in c for c in chunks)


def test_read_epub_falls_back_to_full_text_when_structured_text_is_toc_like(monkeypatch, tmp_path: Path) -> None:
    epub_path = tmp_path / "toc_like.epub"
    epub_path.write_bytes(b"dummy")

    class FakeItem:
        def __init__(self, html: str) -> None:
            self._html = html

        def get_content(self) -> bytes:
            return self._html.encode("utf-8")

    class FakeBook:
        def get_items_of_type(self, _item_type: object) -> list[FakeItem]:
            toc_lines = "".join(f"<h2>第{i}话</h2>" for i in range(1, 40))
            body = "<div>" + ("这是正文内容。" * 300) + "</div>"
            return [FakeItem(f"<html><body>{toc_lines}{body}</body></html>")]

    class FakeEpubModule:
        @staticmethod
        def read_epub(_path: str) -> FakeBook:
            return FakeBook()

    import sys
    import types

    fake_ebooklib = types.ModuleType("ebooklib")
    fake_ebooklib.ITEM_DOCUMENT = 999
    fake_ebooklib.epub = FakeEpubModule
    monkeypatch.setitem(sys.modules, "ebooklib", fake_ebooklib)

    extracted = chunking._read_epub(epub_path)

    assert "这是正文内容" in extracted
    assert len(extracted) > 1200
