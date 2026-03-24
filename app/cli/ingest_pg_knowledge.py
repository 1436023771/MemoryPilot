from __future__ import annotations

import argparse
import os
from pathlib import Path

from app.knowledge.chunking import load_text_documents, split_text
from app.knowledge.embeddings import embed_texts_sentence_transformers
from app.knowledge.pgvector_store import KnowledgeChunk, PgVectorKnowledgeStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into PostgreSQL + pgvector knowledge table")
    parser.add_argument("--input-path", default="docs", help="File or directory to ingest")
    parser.add_argument("--pg-dsn", default=os.getenv("PGVECTOR_DSN", ""), help="PostgreSQL DSN")
    parser.add_argument("--table", default="knowledge_chunks", help="Target table name")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap in characters")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformers model name",
    )
    parser.add_argument("--reset", action="store_true", help="Truncate table before ingest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pg_dsn:
        raise ValueError("--pg-dsn is required (or set PGVECTOR_DSN)")

    input_path = Path(args.input_path)
    docs = load_text_documents(input_path)

    chunks: list[KnowledgeChunk] = []
    for doc in docs:
        pieces = split_text(doc.text, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
        for idx, piece in enumerate(pieces):
            chunk = KnowledgeChunk(
                document_id=doc.path,
                chunk_id=f"{idx:06d}",
                content=piece,
                metadata={"source": doc.path, "chunk_index": idx},
            )
            chunks.append(chunk)

    if not chunks:
        print("No chunks produced. Nothing to ingest.")
        return

    embeddings, dim = embed_texts_sentence_transformers(
        [chunk.content for chunk in chunks],
        model_name=args.embedding_model,
    )

    store = PgVectorKnowledgeStore(
        dsn=args.pg_dsn,
        table_name=args.table,
        embedding_dim=dim,
    )
    store.init_schema()

    if args.reset:
        store.clear_table()

    written = store.upsert_chunks(chunks, embeddings)

    print("Ingest completed")
    print(f"documents: {len(docs)}")
    print(f"chunks: {len(chunks)}")
    print(f"written: {written}")
    print(f"embedding_dim: {dim}")


if __name__ == "__main__":
    main()
