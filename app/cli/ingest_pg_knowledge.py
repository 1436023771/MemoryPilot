from __future__ import annotations

import argparse
from pathlib import Path
import re

from app.knowledge.chunking import load_text_documents, split_document_text
from app.knowledge.embeddings import embed_texts_sentence_transformers
from app.knowledge.narrative_extraction import build_narrative_fields, build_narrative_fields_batch
from app.knowledge.pg_env import resolve_pg_dsn
from app.knowledge.pgvector_store import KnowledgeChunk, PgVectorKnowledgeStore


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", (text or "").strip().lower())
    slug = slug.strip("-")
    return slug or "book"


def _clean_title(text: str) -> str:
    cleaned = re.sub(r"[_\-]+", " ", (text or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip() or "Unknown"


def _infer_book_id(doc_path: Path, input_path: Path, explicit_book_id: str) -> str:
    if explicit_book_id.strip():
        return explicit_book_id.strip()

    root = input_path.expanduser().resolve()
    if root.is_file():
        return _slugify(root.stem)

    try:
        rel = doc_path.resolve().relative_to(root)
        if len(rel.parts) > 1:
            return _slugify(rel.parts[0])
    except Exception:  # noqa: BLE001
        pass

    return _slugify(doc_path.parent.name or doc_path.stem)


def _infer_chapter(doc_path: Path) -> str:
    stem = doc_path.stem
    stem = re.sub(r"^[0-9]+[\._\-\s]*", "", stem)
    return _clean_title(stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into PostgreSQL + pgvector knowledge table")
    parser.add_argument("--input-path", default="docs", help="File or directory to ingest")
    parser.add_argument("--pg-dsn", default="", help="PostgreSQL DSN")
    parser.add_argument("--table", default="knowledge_chunks", help="Target table name")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap in characters")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformers model name",
    )
    parser.add_argument("--book-id", default="", help="Optional fixed book_id for all ingested chunks")
    parser.add_argument("--book-title", default="", help="Optional fixed book title for all ingested chunks")
    parser.add_argument("--reset", action="store_true", help="Truncate table before ingest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pg_dsn = resolve_pg_dsn(args.pg_dsn)

    if not pg_dsn:
        raise ValueError(
            "--pg-dsn is required (or set PGVECTOR_DSN, or PGVECTOR_HOST/PGVECTOR_PORT/PGVECTOR_DBNAME in .env)"
        )

    input_path = Path(args.input_path)
    docs = load_text_documents(input_path)
    print(f"[INFO] 已加载 {len(docs)} 个文档")

    chunks: list[KnowledgeChunk] = []
    base_input_path = input_path.expanduser().resolve()
    global_chunk_order = 0
    
    for doc_idx, doc in enumerate(docs, 1):
        doc_path = Path(doc.path)
        inferred_book_id = _infer_book_id(doc_path=doc_path, input_path=base_input_path, explicit_book_id=args.book_id)
        chapter = _infer_chapter(doc_path)
        book_title = args.book_title.strip() or _clean_title(inferred_book_id)

        pieces = split_document_text(
            doc.text,
            path=doc.path,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
        )
        
        if not pieces:
            continue

        print(f"[{doc_idx}/{len(docs)}] 处理文档: {inferred_book_id} / {chapter} ({len(pieces)} chunks)")

        # 批量处理这个文档的所有 pieces
        chunk_indices = list(range(len(pieces)))
        chunk_orders = list(range(global_chunk_order + 1, global_chunk_order + len(pieces) + 1))
        global_chunk_order += len(pieces)

        narratives = build_narrative_fields_batch(
            book_id=inferred_book_id,
            chapter=chapter,
            chunk_indices=chunk_indices,
            chunk_orders=chunk_orders,
            contents=pieces,
        )
        print(f"  └─ LLM 分析完成")

        for idx, piece, narrative in zip(chunk_indices, pieces, narratives):
            chunk = KnowledgeChunk(
                document_id=doc.path,
                chunk_id=f"{idx:06d}",
                content=piece,
                book_id=inferred_book_id,
                chapter=chapter,
                section="",
                chunk_order=narrative.chunk_order,
                timeline_order=narrative.timeline_order,
                scene_id=narrative.scene_id,
                event_id=narrative.event_id,
                narrative_context=narrative.narrative_context,
                time_markers=narrative.time_markers,
                character_mentions=narrative.character_mentions,
                relationship_edges=narrative.relationship_edges,
                metadata={
                    "source": doc.path,
                    "chunk_index": idx,
                    "book_id": inferred_book_id,
                    "book_title": book_title,
                    "chapter": chapter,
                    "section": "",
                    "chunk_order": narrative.chunk_order,
                    "timeline_order": narrative.timeline_order,
                    "scene_id": narrative.scene_id,
                    "event_id": narrative.event_id,
                    "narrative_context": narrative.narrative_context,
                    "time_markers": narrative.time_markers,
                    "character_mentions": narrative.character_mentions,
                    "relationship_edges": narrative.relationship_edges,
                },
            )
            chunks.append(chunk)

    if not chunks:
        print("No chunks produced. Nothing to ingest.")
        return

    print(f"[INFO] 正在生成向量嵌入（{len(chunks)} chunks）...")
    embeddings, dim = embed_texts_sentence_transformers(
        [chunk.content for chunk in chunks],
        model_name=args.embedding_model,
    )
    print(f"[INFO] 向量生成完成，维度: {dim}")

    store = PgVectorKnowledgeStore(
        dsn=pg_dsn,
        table_name=args.table,
        embedding_dim=dim,
    )
    store.init_schema()

    if args.reset:
        print("[INFO] 清空表数据...")
        store.clear_table()

    print(f"[INFO] 正在写入数据库...")
    written = store.upsert_chunks(chunks, embeddings)

    print("[SUCCESS] Ingest completed")
    print(f"  documents: {len(docs)}")
    print(f"  chunks: {len(chunks)}")
    print(f"  written: {written}")
    print(f"  embedding_dim: {dim}")


if __name__ == "__main__":
    main()
