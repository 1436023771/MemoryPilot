"""PostgreSQL + pgvector knowledge retrieval tool and trace helpers."""

from __future__ import annotations

import os

from langchain.tools import tool

from app.knowledge.embeddings import embed_texts_sentence_transformers
from app.knowledge.pgvector_store import PgVectorKnowledgeStore

_knowledge_retrieval_log: list[dict] = []


def get_knowledge_retrieval_log() -> list[dict]:
    """Return a copy of current turn knowledge retrieval log."""
    return _knowledge_retrieval_log.copy()


def clear_knowledge_retrieval_log() -> None:
    """Clear knowledge retrieval log at the start of each turn."""
    global _knowledge_retrieval_log
    _knowledge_retrieval_log = []


def record_knowledge_retrieval(query: str, result: str) -> None:
    """Record retrieval query and formatted result for GUI sidebar display."""
    global _knowledge_retrieval_log
    _knowledge_retrieval_log.append({"query": query, "result": result})


def _format_hits(hits: list[dict], max_items: int = 5) -> str:
    lines: list[str] = []
    for idx, hit in enumerate(hits[:max_items], start=1):
        score = float(hit.get("score", 0.0))
        doc = str(hit.get("document_id", "")).strip()
        chunk = str(hit.get("chunk_id", "")).strip()
        content = str(hit.get("content", "")).strip().replace("\n", " ")
        preview = content[:260] + ("..." if len(content) > 260 else "")
        lines.append(f"[{idx}] score={score:.4f} doc={doc} chunk={chunk}\n{preview}")
    return "\n\n".join(lines)


@tool
def retrieve_pg_knowledge(query: str, top_k: int = 5) -> str:
    """Retrieve relevant chunks from PostgreSQL + pgvector knowledge base.

    Requires environment variable PGVECTOR_DSN.
    Optional env vars: PGVECTOR_TABLE, PGVECTOR_EMBEDDING_MODEL.
    """
    clean_query = (query or "").strip()
    if not clean_query:
        result = "知识检索失败: 空查询"
        record_knowledge_retrieval(query, result)
        return result

    dsn = os.getenv("PGVECTOR_DSN", "").strip()
    if not dsn:
        result = "知识检索失败: 缺少 PGVECTOR_DSN 环境变量"
        record_knowledge_retrieval(clean_query, result)
        return result

    table = os.getenv("PGVECTOR_TABLE", "knowledge_chunks").strip() or "knowledge_chunks"
    embedding_model = os.getenv(
        "PGVECTOR_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ).strip()

    try:
        query_vecs, dim = embed_texts_sentence_transformers([clean_query], model_name=embedding_model)
        if not query_vecs:
            result = "知识检索失败: 查询向量为空"
            record_knowledge_retrieval(clean_query, result)
            return result

        store = PgVectorKnowledgeStore(
            dsn=dsn,
            table_name=table,
            embedding_dim=dim,
        )
        hits = store.similarity_search(query_embedding=query_vecs[0], top_k=max(1, int(top_k)))

        if not hits:
            result = "知识库未命中相关内容"
            record_knowledge_retrieval(clean_query, result)
            return result

        body = _format_hits(hits, max_items=max(1, int(top_k)))
        result = f"知识库检索结果:\n{body}"
        record_knowledge_retrieval(clean_query, result)
        return result
    except Exception as exc:  # noqa: BLE001
        result = f"知识检索失败: {exc}"
        record_knowledge_retrieval(clean_query, result)
        return result


__all__ = [
    "retrieve_pg_knowledge",
    "get_knowledge_retrieval_log",
    "clear_knowledge_retrieval_log",
    "record_knowledge_retrieval",
]
