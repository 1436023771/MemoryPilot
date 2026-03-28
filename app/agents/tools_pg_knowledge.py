"""PostgreSQL + pgvector knowledge retrieval tool and trace helpers."""

from __future__ import annotations

import json
import os

from langchain.tools import tool
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.knowledge.embeddings import embed_texts_sentence_transformers
from app.knowledge.pg_env import resolve_pg_dsn
from app.knowledge.pgvector_store import PgVectorKnowledgeStore

_knowledge_retrieval_log: list[dict] = []

try:
    from langsmith import tracing_context
except Exception:  # noqa: BLE001
    tracing_context = None


def get_knowledge_retrieval_log() -> list[dict]:
    """Return a copy of current turn knowledge retrieval log."""
    return _knowledge_retrieval_log.copy()


def clear_knowledge_retrieval_log() -> None:
    """Clear knowledge retrieval log at the start of each turn."""
    global _knowledge_retrieval_log
    _knowledge_retrieval_log = []


def record_knowledge_retrieval(query: str, result: str, auxiliary: dict | None = None) -> None:
    """Record retrieval query and formatted result for GUI sidebar display."""
    global _knowledge_retrieval_log
    item = {"query": query, "result": result}
    if auxiliary:
        item["auxiliary"] = auxiliary
    _knowledge_retrieval_log.append(item)


def _rerank_hits_with_llm(query: str, hits: list[dict], top_k: int) -> tuple[list[dict], dict]:
    """Rerank retrieved chunks by relevance using a lightweight LLM judging step."""
    if not hits:
        return [], {}

    settings = get_settings()
    model = ChatOpenAI(
        model=settings.model_name,
        temperature=0.0,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )

    candidates: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        doc = str(hit.get("document_id", "")).strip()
        chunk = str(hit.get("chunk_id", "")).strip()
        book = str(hit.get("book_id", "")).strip()
        chapter = str(hit.get("chapter", "")).strip()
        content = str(hit.get("content", "")).strip().replace("\n", " ")
        preview = content[:420] + ("..." if len(content) > 420 else "")
        candidates.append(
            f"id={idx} book={book or '-'} chapter={chapter or '-'} doc={doc} chunk={chunk}\ntext={preview}"
        )

    prompt = (
        "你是检索重排序器。请根据用户查询评估候选片段相关性，并返回最相关的结果。\n"
        "要求：\n"
        "1) 只返回 JSON，不要额外文字。\n"
        "2) JSON 格式: [{\"id\": 1, \"score\": 0-100, \"reason\": \"...\"}]\n"
        "3) score 越高表示与查询越相关，优先语义匹配与时间线相关信息。\n"
        "4) 仅返回最相关的前 N 条，其中 N="
        f"{max(1, int(top_k))}。\n\n"
        f"用户查询:\n{query}\n\n"
        "候选片段:\n"
        + "\n\n".join(candidates)
    )

    if tracing_context is not None:
        with tracing_context(enabled=False):
            response = model.invoke(prompt)
    else:
        response = model.invoke(prompt)
    content = str(getattr(response, "content", response)).strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        fallback = hits[: max(1, int(top_k))]
        return fallback, {
            "rerank_prompt": prompt,
            "rerank_response": content,
            "rerank_used": False,
            "rerank_fallback": "invalid_json_parse",
        }

    if not isinstance(data, list):
        fallback = hits[: max(1, int(top_k))]
        return fallback, {
            "rerank_prompt": prompt,
            "rerank_response": content,
            "rerank_used": False,
            "rerank_fallback": "invalid_json_array",
        }

    by_id: dict[int, dict] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id"))
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if idx < 1 or idx > len(hits):
            continue

        selected = dict(hits[idx - 1])
        selected["rerank_score"] = score
        selected["rerank_reason"] = str(item.get("reason", "")).strip()
        by_id[idx] = selected

    if not by_id:
        fallback = hits[: max(1, int(top_k))]
        return fallback, {
            "rerank_prompt": prompt,
            "rerank_response": content,
            "rerank_used": False,
            "rerank_fallback": "empty_valid_selection",
        }

    ranked = sorted(by_id.values(), key=lambda h: float(h.get("rerank_score", 0.0)), reverse=True)
    return ranked[: max(1, int(top_k))], {
        "rerank_prompt": prompt,
        "rerank_response": content,
        "rerank_used": True,
        "rerank_fallback": "",
    }


def _format_hits(store: PgVectorKnowledgeStore, hits: list[dict], max_items: int = 5, context_window: int = 2) -> str:
    lines: list[str] = []
    for idx, hit in enumerate(hits[:max_items], start=1):
        score = float(hit.get("score", 0.0))
        book_id = str(hit.get("book_id", "")).strip()
        chapter = str(hit.get("chapter", "")).strip()
        section = str(hit.get("section", "")).strip()
        doc = str(hit.get("document_id", "")).strip()
        chunk = str(hit.get("chunk_id", "")).strip()
        context_info = store.get_chunk_with_context(
            document_id=doc,
            chunk_id=chunk,
            window=max(0, int(context_window)),
        )
        matched = str(context_info.get("matched_content", "")).strip().replace("\n", " ")
        matched_preview = matched[:260] + ("..." if len(matched) > 260 else "")
        context = str(context_info.get("context_content", "")).strip().replace("\n", " ")
        context_preview = context[:900] + ("..." if len(context) > 900 else "")
        context_ids = context_info.get("context_chunk_ids", [])
        context_id_text = ",".join(str(x) for x in context_ids) if context_ids else "-"
        cite = f"book={book_id or '-'} chapter={chapter or '-'} section={section or '-'}"
        rerank_score = hit.get("rerank_score", None)
        rerank_reason = str(hit.get("rerank_reason", "")).strip()
        rerank_line = ""
        if rerank_score is not None:
            rerank_line = f"rerank_score={float(rerank_score):.1f}"
            if rerank_reason:
                rerank_line += f" reason={rerank_reason[:120]}"
        lines.append(
            f"[{idx}] score={score:.4f} {cite} doc={doc} chunk={chunk}\n"
            f"{rerank_line}\n"
            f"matched:\n{matched_preview or '(empty)'}\n"
            f"context_chunks: {context_id_text}\n"
            f"context:\n{context_preview or '(empty)'}"
        )
    return "\n\n".join(lines)


@tool
def retrieve_pg_knowledge(
    query: str,
    top_k: int = 5,
    book_id: str = "",
    chapter: str = "",
    context_window: int = 2,
    rerank_candidates: int = 12,
) -> str:
    """Retrieve relevant chunks from PostgreSQL + pgvector knowledge base.

    Requires environment variable PGVECTOR_DSN.
    Optional env vars: PGVECTOR_TABLE, PGVECTOR_EMBEDDING_MODEL.

    Args:
        query: Search query text.
        top_k: Number of chunks to return.
        book_id: Optional exact filter of book_id.
        chapter: Optional exact filter of chapter.
        context_window: Neighbor chunk window for returning longer context around a hit.
        rerank_candidates: Candidate pool size for LLM relevance reranking before final output.
    """
    clean_query = (query or "").strip()
    if not clean_query:
        result = "知识检索失败: 空查询"
        record_knowledge_retrieval(query, result)
        return result

    dsn = resolve_pg_dsn("")
    if not dsn:
        result = "知识检索失败: 缺少数据库连接配置（PGVECTOR_DSN 或 PGVECTOR_HOST/PGVECTOR_PORT/PGVECTOR_DBNAME）"
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
        clean_book_id = (book_id or "").strip() or None
        clean_chapter = (chapter or "").strip() or None
        hits = store.similarity_search(
            query_embedding=query_vecs[0],
            top_k=max(max(1, int(top_k)), max(2, int(rerank_candidates))),
            book_id=clean_book_id,
            chapter=clean_chapter,
        )

        if not hits:
            result = "知识库未命中相关内容"
            record_knowledge_retrieval(clean_query, result)
            return result

        reranked_hits = hits[: max(1, int(top_k))]
        rerank_aux: dict = {}
        try:
            reranked_hits, rerank_aux = _rerank_hits_with_llm(
                query=clean_query,
                hits=hits,
                top_k=max(1, int(top_k)),
            )
        except Exception as exc:  # noqa: BLE001
            reranked_hits = hits[: max(1, int(top_k))]
            rerank_aux = {
                "rerank_prompt": "",
                "rerank_response": "",
                "rerank_used": False,
                "rerank_fallback": f"exception:{exc}",
            }

        body = _format_hits(
            store=store,
            hits=reranked_hits,
            max_items=max(1, int(top_k)),
            context_window=max(0, int(context_window)),
        )
        result = f"知识库检索结果:\n{body}"
        auxiliary = {
            "rerank_used": bool(rerank_aux.get("rerank_used", False)),
            "rerank_fallback": str(rerank_aux.get("rerank_fallback", "")).strip(),
            "rerank_prompt": str(rerank_aux.get("rerank_prompt", ""))[:1200],
            "rerank_response": str(rerank_aux.get("rerank_response", ""))[:1200],
        }
        record_knowledge_retrieval(clean_query, result, auxiliary=auxiliary)
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
