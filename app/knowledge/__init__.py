"""Knowledge ingestion and retrieval utilities for PostgreSQL + pgvector."""

from app.knowledge.chunking import TextDocument, load_text_documents, split_text
from app.knowledge.embeddings import embed_texts_sentence_transformers
from app.knowledge.pgvector_store import KnowledgeChunk, PgVectorKnowledgeStore

__all__ = [
    "TextDocument",
    "KnowledgeChunk",
    "PgVectorKnowledgeStore",
    "load_text_documents",
    "split_text",
    "embed_texts_sentence_transformers",
]
