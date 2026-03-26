from pathlib import Path

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
