from __future__ import annotations

import argparse
import os

from app.knowledge.embeddings import embed_texts_sentence_transformers
from app.knowledge.pgvector_store import PgVectorKnowledgeStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query PostgreSQL + pgvector knowledge table")
    parser.add_argument("question", help="Question or query text")
    parser.add_argument("--pg-dsn", default=os.getenv("PGVECTOR_DSN", ""), help="PostgreSQL DSN")
    parser.add_argument("--table", default="knowledge_chunks", help="Target table name")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k chunks")
    parser.add_argument("--document-id", default="", help="Optional filter by document_id")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformers model name",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pg_dsn:
        raise ValueError("--pg-dsn is required (or set PGVECTOR_DSN)")

    query_vecs, dim = embed_texts_sentence_transformers([args.question], model_name=args.embedding_model)
    query_vec = query_vecs[0]

    store = PgVectorKnowledgeStore(
        dsn=args.pg_dsn,
        table_name=args.table,
        embedding_dim=dim,
    )

    hits = store.similarity_search(
        query_embedding=query_vec,
        top_k=args.top_k,
        document_id=args.document_id or None,
    )

    if not hits:
        print("No results found")
        return

    for idx, hit in enumerate(hits, start=1):
        print(f"[{idx}] score={hit['score']:.4f}")
        print(f"document: {hit['document_id']}")
        print(f"chunk: {hit['chunk_id']}")
        print(hit["content"][:300])
        print("-" * 40)


if __name__ == "__main__":
    main()
