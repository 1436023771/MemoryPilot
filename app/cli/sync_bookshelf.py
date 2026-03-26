from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from app.knowledge.chunking import load_text_documents, split_document_text
from app.knowledge.embeddings import embed_texts_sentence_transformers
from app.knowledge.pg_env import resolve_bookshelf_path, resolve_pg_dsn
from app.knowledge.pgvector_store import KnowledgeChunk, PgVectorKnowledgeStore


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", (text or "").strip().lower())
    slug = slug.strip("-")
    return slug or "book"


def _clean_label(text: str) -> str:
    cleaned = re.sub(r"[_\-]+", " ", (text or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip() or "Unknown"


def _strip_prefix_number(text: str) -> str:
    return re.sub(r"^[0-9]+[\._\-\s]*", "", (text or "").strip())


def _derive_bookshelf_fields(doc_path: Path, bookshelf_root: Path) -> dict[str, str]:
    """Derive series/book/chapter labels from bookshelf folder structure.

    Rule:
    - bookshelf_root/<series>/<book>/<chapter_file>
    - bookshelf_root/<series>/<chapter_file>  (book name defaults to file stem)
    """
    rel = doc_path.resolve().relative_to(bookshelf_root.resolve())
    parts = list(rel.parts)
    stem = _clean_label(_strip_prefix_number(doc_path.stem))

    if not parts:
        series = "default"
        book_name = stem
        section = ""
    elif len(parts) == 1:
        series = "default"
        book_name = stem
        section = ""
    elif len(parts) == 2:
        # series/<file>
        series = _clean_label(parts[0])
        book_name = stem
        section = ""
    else:
        # series/book/.../<file>
        series = _clean_label(parts[0])
        book_name = _clean_label(parts[1])
        nested = parts[2:-1]
        section = " / ".join(_clean_label(p) for p in nested) if nested else ""

    chapter = stem
    book_id = _slugify(f"{series}__{book_name}")

    return {
        "series": series,
        "book_name": book_name,
        "book_id": book_id,
        "chapter": chapter,
        "section": section,
        "relative_path": rel.as_posix(),
    }


def _load_sync_config(config_file: Path) -> dict:
    if not config_file.exists():
        return {}

    try:
        return json.loads(config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_sync_config(config_file: Path, config: dict) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync bookshelf folders to PostgreSQL + pgvector (series/book/chapter aware)",
    )
    parser.add_argument(
        "--bookshelf-path",
        default="",
        help="Bookshelf root path. Structure: <root>/<series>/<book>/<files>",
    )
    parser.add_argument("--pg-dsn", default="", help="PostgreSQL DSN (fallback to env PGVECTOR_DSN)")
    parser.add_argument("--table", default="", help="Target table name")
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Sentence-transformers model name",
    )
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap in characters")
    parser.add_argument("--config-file", default="memory/bookshelf_sync.json", help="Local sync config path")
    parser.add_argument("--reset", action="store_true", help="Truncate table before sync")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, do not write DB")
    parser.add_argument(
        "--no-save-config",
        action="store_true",
        help="Do not persist updated parameters to config file",
    )
    return parser.parse_args()


def _build_runtime_config(args: argparse.Namespace) -> dict:
    config_file = Path(args.config_file)
    saved = _load_sync_config(config_file)

    runtime = {
        "bookshelf_path": resolve_bookshelf_path(args.bookshelf_path.strip()) or saved.get("bookshelf_path", ""),
        "pg_dsn": resolve_pg_dsn(args.pg_dsn.strip()) or saved.get("pg_dsn", ""),
        "table": args.table.strip() or saved.get("table", "knowledge_chunks"),
        "embedding_model": args.embedding_model.strip()
        or saved.get("embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        "chunk_size": args.chunk_size if args.chunk_size is not None else int(saved.get("chunk_size", 800)),
        "chunk_overlap": args.chunk_overlap
        if args.chunk_overlap is not None
        else int(saved.get("chunk_overlap", 120)),
    }

    if not runtime["bookshelf_path"]:
        raise ValueError("bookshelf_path is required. Use --bookshelf-path or set BOOKSHELF_PATH in .env.")
    if not runtime["pg_dsn"] and not args.dry_run:
        raise ValueError(
            "pg_dsn is required. Use --pg-dsn or set PGVECTOR_DSN, or PGVECTOR_HOST/PGVECTOR_PORT/PGVECTOR_DBNAME in .env."
        )

    if not args.no_save_config:
        _save_sync_config(config_file, runtime)

    return runtime


def _build_chunks(bookshelf_root: Path, chunk_size: int, chunk_overlap: int) -> list[KnowledgeChunk]:
    docs = load_text_documents(bookshelf_root)

    chunks: list[KnowledgeChunk] = []
    for doc in docs:
        doc_path = Path(doc.path)
        fields = _derive_bookshelf_fields(doc_path, bookshelf_root)

        pieces = split_document_text(
            doc.text,
            path=doc.path,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
        for idx, piece in enumerate(pieces):
            chunks.append(
                KnowledgeChunk(
                    document_id=str(doc_path),
                    chunk_id=f"{idx:06d}",
                    content=piece,
                    metadata={
                        "source": str(doc_path),
                        "chunk_index": idx,
                        "series": fields["series"],
                        "book_name": fields["book_name"],
                        "book_id": fields["book_id"],
                        "chapter": fields["chapter"],
                        "section": fields["section"],
                        "relative_path": fields["relative_path"],
                    },
                    book_id=fields["book_id"],
                    chapter=fields["chapter"],
                    section=fields["section"],
                )
            )

    return chunks


def main() -> None:
    args = parse_args()
    runtime = _build_runtime_config(args)

    bookshelf_root = Path(runtime["bookshelf_path"]).expanduser().resolve()
    if not bookshelf_root.exists() or not bookshelf_root.is_dir():
        raise ValueError(f"Invalid bookshelf path: {bookshelf_root}")

    chunks = _build_chunks(
        bookshelf_root=bookshelf_root,
        chunk_size=int(runtime["chunk_size"]),
        chunk_overlap=int(runtime["chunk_overlap"]),
    )

    if not chunks:
        print("No supported files found in bookshelf. Nothing to sync.")
        return

    series_set = {
        str(chunk.metadata.get("series", "")).strip()
        for chunk in chunks
        if str(chunk.metadata.get("series", "")).strip()
    }
    book_set = {
        str(chunk.metadata.get("book_id", "")).strip()
        for chunk in chunks
        if str(chunk.metadata.get("book_id", "")).strip()
    }

    print(f"bookshelf_root: {bookshelf_root}")
    print(f"series_count: {len(series_set)}")
    print(f"book_count: {len(book_set)}")
    print(f"chunk_count: {len(chunks)}")

    if args.dry_run:
        print("dry_run=true, skip embedding and database write")
        return

    embeddings, dim = embed_texts_sentence_transformers(
        [chunk.content for chunk in chunks],
        model_name=str(runtime["embedding_model"]),
    )

    store = PgVectorKnowledgeStore(
        dsn=str(runtime["pg_dsn"]),
        table_name=str(runtime["table"]),
        embedding_dim=dim,
    )
    store.init_schema()

    if args.reset:
        store.clear_table()

    written = store.upsert_chunks(chunks, embeddings)

    print("sync_completed=true")
    print(f"written: {written}")
    print(f"embedding_dim: {dim}")


if __name__ == "__main__":
    main()
