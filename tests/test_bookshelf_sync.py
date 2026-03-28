from pathlib import Path

import app.knowledge.narrative_extraction as narrative_extraction
from app.cli.sync_bookshelf import _build_chunks
from app.cli.sync_bookshelf import _derive_bookshelf_fields


def test_derive_bookshelf_fields_series_book_structure(tmp_path: Path) -> None:
    root = tmp_path / "bookshelf"
    chapter_file = root / "History Series" / "Book One" / "01_intro.md"
    chapter_file.parent.mkdir(parents=True, exist_ok=True)
    chapter_file.write_text("content", encoding="utf-8")

    fields = _derive_bookshelf_fields(chapter_file, root)

    assert fields["series"] == "History Series"
    assert fields["book_name"] == "Book One"
    assert fields["chapter"] == "intro"
    assert fields["book_id"]


def test_derive_bookshelf_fields_series_file_structure(tmp_path: Path) -> None:
    root = tmp_path / "bookshelf"
    chapter_file = root / "Math Series" / "02_limits.md"
    chapter_file.parent.mkdir(parents=True, exist_ok=True)
    chapter_file.write_text("content", encoding="utf-8")

    fields = _derive_bookshelf_fields(chapter_file, root)

    assert fields["series"] == "Math Series"
    assert fields["book_name"] == "limits"
    assert fields["chapter"] == "limits"
    assert fields["section"] == ""


def test_build_chunks_contains_deterministic_narrative_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        narrative_extraction,
        "_analyze_content_with_cache",
        lambda _content: {
            "narrative_context": "present",
            "time_markers": ["第三天"],
            "character_mentions": ["李雷", "韩梅梅"],
            "relationship_edges": [],
        },
    )

    root = tmp_path / "bookshelf"
    ch1 = root / "Series A" / "Book One" / "01_intro.md"
    ch2 = root / "Series A" / "Book One" / "02_plot.md"
    ch1.parent.mkdir(parents=True, exist_ok=True)
    ch1.write_text("第三天清晨，李雷出现。", encoding="utf-8")
    ch2.write_text("后来韩梅梅也出现。", encoding="utf-8")

    chunks_a = _build_chunks(root, chunk_size=20, chunk_overlap=0)
    chunks_b = _build_chunks(root, chunk_size=20, chunk_overlap=0)

    assert len(chunks_a) == len(chunks_b)
    assert chunks_a

    for left, right in zip(chunks_a, chunks_b):
        assert left.document_id == right.document_id
        assert left.chunk_id == right.chunk_id
        assert left.chunk_order == right.chunk_order
        assert left.timeline_order == right.timeline_order
        assert left.scene_id == right.scene_id
        assert left.event_id == right.event_id

    first = chunks_a[0]
    assert first.chunk_order > 0
    assert first.timeline_order == first.chunk_order
    assert first.scene_id
    assert first.event_id
