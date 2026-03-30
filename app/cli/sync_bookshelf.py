from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

from app.cli.sync_bookshelf_config import (
    chapter_analysis_concurrency,
    incremental_enabled,
    auto_delete_removed,
    hash_check_enabled,
    show_incremental_stats,
    state_file_path,
)
from app.knowledge.chunking import TextDocument, load_text_documents, split_document_text
from app.knowledge.embeddings import embed_texts_sentence_transformers
from app.knowledge.narrative_extraction import build_narrative_fields_batch_async
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


def _content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _list_supported_files(root: Path) -> list[Path]:
    allowed_ext = {".txt", ".md", ".rst", ".py", ".pdf", ".epub"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed_ext])


def _load_sync_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {"documents": {}}
    try:
        parsed = json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"documents": {}}
    if not isinstance(parsed, dict):
        return {"documents": {}}
    docs = parsed.get("documents", {})
    if not isinstance(docs, dict):
        docs = {}
    return {"documents": docs}


def _save_sync_state(state_file: Path, documents: dict[str, dict]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"documents": documents}
    state_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_single_document(file_path: Path) -> TextDocument | None:
    docs = load_text_documents(file_path)
    if not docs:
        return None
    return docs[0]


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
    parser.add_argument("--state-file", default="", help="Incremental state file path")
    parser.add_argument("--reset", action="store_true", help="Truncate table before sync")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, do not write DB")
    parser.add_argument("--full-rebuild", action="store_true", help="Force rebuilding all documents")
    parser.add_argument(
        "--incremental",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable incremental sync mode",
    )
    parser.add_argument(
        "--auto-delete-removed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Automatically delete chunks for files removed from bookshelf",
    )
    parser.add_argument(
        "--hash-check",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use content hash to confirm changes when mtime/size changed",
    )
    parser.add_argument(
        "--show-incremental-stats",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print detailed incremental planning stats",
    )
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
        "state_file": args.state_file.strip() or str(saved.get("state_file", state_file_path())),
        "incremental": args.incremental if args.incremental is not None else bool(saved.get("incremental", incremental_enabled())),
        "auto_delete_removed": args.auto_delete_removed
        if args.auto_delete_removed is not None
        else bool(saved.get("auto_delete_removed", auto_delete_removed())),
        "hash_check": args.hash_check if args.hash_check is not None else bool(saved.get("hash_check", hash_check_enabled())),
        "show_incremental_stats": args.show_incremental_stats
        if args.show_incremental_stats is not None
        else bool(saved.get("show_incremental_stats", show_incremental_stats())),
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


def _build_chunks_for_documents(
    docs: list[TextDocument],
    bookshelf_root: Path,
    chunk_size: int,
    chunk_overlap: int,
    doc_content_hashes: dict[str, str] | None = None,
) -> list[KnowledgeChunk]:
    print(f"[INFO] 已规划处理 {len(docs)} 个文档")

    chapter_groups: list[dict] = []
    global_chunk_order = 0

    for doc_idx, doc in enumerate(docs, 1):
        doc_path = Path(doc.path)
        fields = _derive_bookshelf_fields(doc_path, bookshelf_root)

        pieces = split_document_text(
            doc.text,
            path=doc.path,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )

        if not pieces:
            continue

        print(f"[{doc_idx}/{len(docs)}] 发现章节: {fields['book_id']} / {fields['chapter']} ({len(pieces)} chunks)")

        chunk_indices = list(range(len(pieces)))
        chunk_orders = list(range(global_chunk_order + 1, global_chunk_order + len(pieces) + 1))
        global_chunk_order += len(pieces)

        chapter_groups.append(
            {
                "doc_path": str(doc_path),
                "fields": fields,
                "pieces": pieces,
                "chunk_indices": chunk_indices,
                "chunk_orders": chunk_orders,
                "content_hash": (doc_content_hashes or {}).get(str(doc_path), ""),
            }
        )

    if not chapter_groups:
        return []

    async def _analyze_chapter_groups() -> list[list]:
        semaphore = asyncio.Semaphore(chapter_analysis_concurrency())
        total = len(chapter_groups)

        async def _analyze_one(i: int, group: dict) -> tuple[int, list]:
            fields = group["fields"]
            async with semaphore:
                print(
                    f"[{i + 1}/{total}] 异步分析章节: "
                    f"{fields['book_id']} / {fields['chapter']} ({len(group['pieces'])} chunks)"
                )
                narratives = await build_narrative_fields_batch_async(
                    book_id=fields["book_id"],
                    chapter=fields["chapter"],
                    chunk_indices=group["chunk_indices"],
                    chunk_orders=group["chunk_orders"],
                    contents=group["pieces"],
                )
                print(f"  └─ LLM 分析完成: {fields['book_id']} / {fields['chapter']}")
                return i, narratives

        tasks = [_analyze_one(i, g) for i, g in enumerate(chapter_groups)]
        done = await asyncio.gather(*tasks)
        ordered: list[list] = [[] for _ in chapter_groups]
        for i, narratives in done:
            ordered[i] = narratives
        return ordered

    narratives_by_group = asyncio.run(_analyze_chapter_groups())

    chunks: list[KnowledgeChunk] = []
    for group, narratives in zip(chapter_groups, narratives_by_group):
        fields = group["fields"]
        for idx, piece, narrative in zip(group["chunk_indices"], group["pieces"], narratives):
            chunks.append(
                KnowledgeChunk(
                    document_id=group["doc_path"],
                    chunk_id=f"{idx:06d}",
                    content=piece,
                    chunk_order=narrative.chunk_order,
                    timeline_order=narrative.timeline_order,
                    scene_id=narrative.scene_id,
                    event_id=narrative.event_id,
                    narrative_context=narrative.narrative_context,
                    time_markers=narrative.time_markers,
                    character_mentions=narrative.character_mentions,
                    relationship_edges=narrative.relationship_edges,
                    metadata={
                        "source": group["doc_path"],
                        "chunk_index": idx,
                        "series": fields["series"],
                        "book_name": fields["book_name"],
                        "book_id": fields["book_id"],
                        "chapter": fields["chapter"],
                        "section": fields["section"],
                        "relative_path": fields["relative_path"],
                        "content_hash": group["content_hash"],
                        "chunk_order": narrative.chunk_order,
                        "timeline_order": narrative.timeline_order,
                        "scene_id": narrative.scene_id,
                        "event_id": narrative.event_id,
                        "narrative_context": narrative.narrative_context,
                        "time_markers": narrative.time_markers,
                        "character_mentions": narrative.character_mentions,
                        "relationship_edges": narrative.relationship_edges,
                    },
                    book_id=fields["book_id"],
                    chapter=fields["chapter"],
                    section=fields["section"],
                )
            )

    return chunks


def _build_chunks(bookshelf_root: Path, chunk_size: int, chunk_overlap: int) -> list[KnowledgeChunk]:
    docs = load_text_documents(bookshelf_root)
    return _build_chunks_for_documents(
        docs=docs,
        bookshelf_root=bookshelf_root,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _plan_incremental_documents(
    bookshelf_root: Path,
    previous_documents: dict[str, dict],
    incremental: bool,
    full_rebuild: bool,
    hash_check: bool,
) -> tuple[list[TextDocument], dict[str, dict], list[str], dict[str, int], dict[str, str]]:
    files = _list_supported_files(bookshelf_root)
    previous_keys = set(previous_documents.keys())

    docs_to_process: list[TextDocument] = []
    next_documents: dict[str, dict] = {}
    doc_content_hashes: dict[str, str] = {}
    stats = {
        "scanned": len(files),
        "new": 0,
        "changed": 0,
        "unchanged": 0,
    }

    for file_path in files:
        key = str(file_path.resolve())
        st = file_path.stat()
        mtime_ns = int(st.st_mtime_ns)
        size = int(st.st_size)
        prev = previous_documents.get(key, {})

        quick_changed = (not prev) or int(prev.get("mtime_ns", -1)) != mtime_ns or int(prev.get("size", -1)) != size
        should_process = bool(full_rebuild or (not incremental) or quick_changed)

        if should_process:
            loaded = _read_single_document(file_path)
            if loaded is None:
                continue

            doc_hash = _content_hash(loaded.text)
            prev_hash = str(prev.get("content_hash", "")).strip()
            hash_indicates_change = (not prev_hash) or (doc_hash != prev_hash)

            if incremental and (not full_rebuild) and hash_check and prev and quick_changed and (not hash_indicates_change):
                stats["unchanged"] += 1
                next_documents[key] = {
                    "mtime_ns": mtime_ns,
                    "size": size,
                    "content_hash": prev_hash,
                    "chunk_count": int(prev.get("chunk_count", 0) or 0),
                    "last_synced_at": int(prev.get("last_synced_at", 0) or 0),
                }
                continue

            docs_to_process.append(TextDocument(path=key, text=loaded.text))
            doc_content_hashes[key] = doc_hash
            if prev:
                stats["changed"] += 1
            else:
                stats["new"] += 1
        else:
            stats["unchanged"] += 1
            next_documents[key] = {
                "mtime_ns": mtime_ns,
                "size": size,
                "content_hash": str(prev.get("content_hash", "")).strip(),
                "chunk_count": int(prev.get("chunk_count", 0) or 0),
                "last_synced_at": int(prev.get("last_synced_at", 0) or 0),
            }

    removed_documents = sorted(previous_keys - set(next_documents.keys()) - set(doc_content_hashes.keys()))
    return docs_to_process, next_documents, removed_documents, stats, doc_content_hashes


def main() -> None:
    args = parse_args()
    runtime = _build_runtime_config(args)

    bookshelf_root = Path(runtime["bookshelf_path"]).expanduser().resolve()
    if not bookshelf_root.exists() or not bookshelf_root.is_dir():
        raise ValueError(f"Invalid bookshelf path: {bookshelf_root}")

    state_file = Path(str(runtime["state_file"])).expanduser()
    previous_state = _load_sync_state(state_file)
    previous_documents = previous_state.get("documents", {})
    if not isinstance(previous_documents, dict):
        previous_documents = {}

    docs_to_process, next_documents, removed_documents, planning_stats, doc_hashes = _plan_incremental_documents(
        bookshelf_root=bookshelf_root,
        previous_documents=previous_documents,
        incremental=bool(runtime["incremental"]),
        full_rebuild=bool(args.full_rebuild),
        hash_check=bool(runtime["hash_check"]),
    )

    chunks = _build_chunks_for_documents(
        docs=docs_to_process,
        bookshelf_root=bookshelf_root,
        chunk_size=int(runtime["chunk_size"]),
        chunk_overlap=int(runtime["chunk_overlap"]),
        doc_content_hashes=doc_hashes,
    )

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

    if bool(runtime["show_incremental_stats"]):
        print(
            "incremental_stats: "
            f"scanned={planning_stats['scanned']} "
            f"new={planning_stats['new']} "
            f"changed={planning_stats['changed']} "
            f"unchanged={planning_stats['unchanged']} "
            f"removed={len(removed_documents)}"
        )

    changed_document_ids = sorted({chunk.document_id for chunk in chunks})

    if args.dry_run:
        print("dry_run=true, skip embedding and database write")
        return

    dim = 384
    embeddings: list[list[float]] = []
    if chunks:
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

    deleted_for_changed = 0
    for document_id in changed_document_ids:
        deleted_for_changed += store.delete_by_document_id(document_id)

    deleted_removed = 0
    if bool(runtime["auto_delete_removed"]):
        for document_id in removed_documents:
            deleted_removed += store.delete_by_document_id(document_id)
    elif removed_documents:
        print(f"removed_documents_detected={len(removed_documents)} auto_delete_removed=false")

    written = 0
    if chunks:
        written = store.upsert_chunks(chunks, embeddings)

    doc_chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        doc_chunk_counts[chunk.document_id] = doc_chunk_counts.get(chunk.document_id, 0) + 1

    now_ts = int(time.time())
    for document_id, count in doc_chunk_counts.items():
        path = Path(document_id)
        if not path.exists():
            continue
        st = path.stat()
        next_documents[document_id] = {
            "mtime_ns": int(st.st_mtime_ns),
            "size": int(st.st_size),
            "content_hash": doc_hashes.get(document_id, ""),
            "chunk_count": int(count),
            "last_synced_at": now_ts,
        }

    _save_sync_state(state_file, next_documents)

    print("sync_completed=true")
    print(f"written: {written}")
    print(f"embedding_dim: {dim}")
    if bool(runtime["show_incremental_stats"]):
        print(f"deleted_for_changed: {deleted_for_changed}")
        print(f"deleted_removed: {deleted_removed}")


if __name__ == "__main__":
    main()
