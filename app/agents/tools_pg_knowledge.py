"""PostgreSQL + pgvector knowledge retrieval tool and trace helpers."""

from __future__ import annotations

from collections import Counter
import json
import os
from typing import Any

from langchain.tools import tool
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.prompt_store import render_prompt
from app.agents.knowledge_config import (
    pgvector_table,
    pgvector_embedding_model,
    local_rerank_weights,
    blend_weights,
    top_k_default,
    context_window_default,
    rerank_candidates_default,
)
from app.knowledge.embeddings import embed_texts_sentence_transformers
from app.knowledge.pg_env import resolve_pg_dsn
from app.knowledge.pgvector_store import PgVectorKnowledgeStore

_knowledge_retrieval_log: list[dict] = []
_query_analysis_cache: dict[str, dict[str, Any]] = {}

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
    global _query_analysis_cache
    _knowledge_retrieval_log = []
    _query_analysis_cache = {}


def record_knowledge_retrieval(query: str, result: str, auxiliary: dict | None = None) -> None:
    """Record retrieval query and formatted result for GUI sidebar display."""
    global _knowledge_retrieval_log
    item = {"query": query, "result": result}
    if auxiliary:
        item["auxiliary"] = auxiliary
    _knowledge_retrieval_log.append(item)


def _normalize_score_0_1(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if upper <= lower:
        return 0.0
    clipped = max(lower, min(upper, value))
    return (clipped - lower) / (upper - lower)


def _normalize_query_cache_key(query: str, book_id: str | None) -> str:
    return f"book={str(book_id or '').strip().lower()}|query={str(query or '').strip().lower()}"


def _build_query_analysis_fallback(query: str, candidate_characters: list[str]) -> dict[str, Any]:
    """Build regex-free fallback analysis when LLM output is unavailable."""
    query_text = str(query or "")
    selected = [name for name in candidate_characters if name and name in query_text][:8]
    return {
        "characters": selected,
        "timeline_intent": "unknown",
        "relation_intent": False,
        "confidence": 0.0,
        "source": "fallback",
    }


def _analyze_query_with_llm(query: str, candidate_characters: list[str]) -> dict[str, Any]:
    settings = get_settings()
    model = ChatOpenAI(
        model=settings.model_name,
        temperature=0.0,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )

    candidate_text = ", ".join(candidate_characters[:80]) if candidate_characters else "(none)"
    prompt = render_prompt(
        "agents.pg_knowledge.query_analysis",
        query=query,
        candidate_text=candidate_text,
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

    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("query analysis must return JSON object")

    timeline_intent = str(parsed.get("timeline_intent", "unknown")).strip().lower()
    if timeline_intent not in {"none", "ordering", "evolution", "comparison", "unknown"}:
        timeline_intent = "unknown"

    raw_chars = parsed.get("characters", [])
    chars: list[str] = []
    if isinstance(raw_chars, list):
        candidate_set = {c.lower(): c for c in candidate_characters}
        for item in raw_chars:
            token = str(item).strip()
            if not token:
                continue
            mapped = candidate_set.get(token.lower(), token)
            if mapped not in chars:
                chars.append(mapped)
            if len(chars) >= 8:
                break

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "characters": chars,
        "timeline_intent": timeline_intent,
        "relation_intent": bool(parsed.get("relation_intent", False)),
        "confidence": confidence,
        "source": "llm",
    }


def _analyze_query_with_cache(query: str, book_id: str | None, candidate_characters: list[str]) -> dict[str, Any]:
    cache_key = _normalize_query_cache_key(query, book_id)
    cached = _query_analysis_cache.get(cache_key)
    if cached is not None:
        return dict(cached)

    try:
        analysis = _analyze_query_with_llm(query, candidate_characters)
    except Exception:  # noqa: BLE001
        analysis = _build_query_analysis_fallback(query, candidate_characters)

    _query_analysis_cache[cache_key] = dict(analysis)
    return analysis


def _to_str_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def _local_rerank_weights() -> tuple[float, float, float]:
    return local_rerank_weights()


def _blend_weights() -> tuple[float, float]:
    return blend_weights()


def _apply_role_timeline_rerank(hits: list[dict], analysis: dict[str, Any]) -> tuple[list[dict], dict]:
    """Apply deterministic rerank with character continuity and timeline coherence."""
    if not hits:
        return [], {
            "local_rerank_used": True,
            "query_characters": [],
            "timeline_intent": "unknown",
            "analysis_source": str(analysis.get("source", "unknown")),
        }

    query_chars = _to_str_list(analysis.get("characters", []))
    timeline_intent = str(analysis.get("timeline_intent", "unknown")).strip().lower()
    timeline_query = timeline_intent in {"ordering", "evolution", "comparison"}

    # If query does not name characters, infer anchors from frequent mentions in candidates.
    anchor_chars: list[str] = []
    if query_chars:
        anchor_chars = query_chars
    else:
        counter: Counter[str] = Counter()
        for hit in hits:
            for name in _to_str_list(hit.get("character_mentions", [])):
                counter[name] += 1
        anchor_chars = [name for name, _ in counter.most_common(3)]

    reranked: list[dict] = []
    w_semantic, w_character, w_timeline = _local_rerank_weights()
    for hit in hits:
        chars = _to_str_list(hit.get("character_mentions", []))
        char_set = set(chars)
        anchor_set = set(anchor_chars)

        # Character continuity: overlap with query/anchor characters.
        char_overlap = len(char_set & anchor_set)
        char_base = len(anchor_set) if anchor_set else 1
        character_score = char_overlap / char_base

        # Timeline coherence: prefer chunks with timeline metadata when timeline intent exists.
        timeline_order = int(hit.get("timeline_order", 0) or 0)
        time_markers = _to_str_list(hit.get("time_markers", []))
        if timeline_query:
            timeline_score = 0.0
            if timeline_order > 0:
                timeline_score += 0.7
            if time_markers:
                timeline_score += 0.3
        else:
            timeline_score = 0.5 if timeline_order > 0 else 0.2
        timeline_score = min(1.0, timeline_score)

        # Semantic base from vector similarity score field.
        semantic = _normalize_score_0_1(float(hit.get("score", 0.0)), lower=0.0, upper=1.0)

        local_score = w_semantic * semantic + w_character * character_score + w_timeline * timeline_score

        enriched = dict(hit)
        enriched["local_semantic_score"] = round(semantic * 100.0, 2)
        enriched["local_character_score"] = round(character_score * 100.0, 2)
        enriched["local_timeline_score"] = round(timeline_score * 100.0, 2)
        enriched["local_rerank_score"] = round(local_score * 100.0, 2)
        reranked.append(enriched)

    reranked.sort(key=lambda h: float(h.get("local_rerank_score", 0.0)), reverse=True)
    return reranked, {
        "local_rerank_used": True,
        "query_characters": anchor_chars,
        "timeline_intent": timeline_intent,
        "analysis_source": str(analysis.get("source", "unknown")),
        "analysis_confidence": float(analysis.get("confidence", 0.0) or 0.0),
    }


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
        timeline_order = int(hit.get("timeline_order", 0) or 0)
        scene_id = str(hit.get("scene_id", "")).strip()
        event_id = str(hit.get("event_id", "")).strip()
        character_mentions = ",".join(_to_str_list(hit.get("character_mentions", []))) or "-"
        time_markers = ",".join(_to_str_list(hit.get("time_markers", []))) or "-"
        content = str(hit.get("content", "")).strip().replace("\n", " ")
        preview = content[:420] + ("..." if len(content) > 420 else "")
        candidates.append(
            "id="
            f"{idx} book={book or '-'} chapter={chapter or '-'} timeline_order={timeline_order} "
            f"scene={scene_id or '-'} event={event_id or '-'} doc={doc} chunk={chunk}\n"
            f"characters={character_mentions}\n"
            f"time_markers={time_markers}\n"
            f"text={preview}"
        )

    prompt = render_prompt(
        "agents.pg_knowledge.rerank",
        top_k=max(1, int(top_k)),
        query=query,
        candidates="\n\n".join(candidates),
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
        timeline_order = int(hit.get("timeline_order", 0) or 0)
        scene_id = str(hit.get("scene_id", "")).strip()
        event_id = str(hit.get("event_id", "")).strip()
        character_mentions = ",".join(_to_str_list(hit.get("character_mentions", []))) or "-"
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
        local_sem = float(hit.get("local_semantic_score", 0.0) or 0.0)
        local_char = float(hit.get("local_character_score", 0.0) or 0.0)
        local_time = float(hit.get("local_timeline_score", 0.0) or 0.0)
        local_total = float(hit.get("local_rerank_score", 0.0) or 0.0)
        rerank_line = ""
        if rerank_score is not None:
            rerank_line = f"rerank_score={float(rerank_score):.1f}"
            if rerank_reason:
                rerank_line += f" reason={rerank_reason[:120]}"
        local_line = (
            f"local_rerank={local_total:.1f} semantic={local_sem:.1f} "
            f"character={local_char:.1f} timeline={local_time:.1f}"
        )
        lines.append(
            f"[{idx}] score={score:.4f} {cite} doc={doc} chunk={chunk} "
            f"timeline_order={timeline_order} scene={scene_id or '-'} event={event_id or '-'}\n"
            f"{rerank_line}\n"
            f"{local_line}\n"
            f"characters: {character_mentions}\n"
            f"matched:\n{matched_preview or '(empty)'}\n"
            f"context_chunks: {context_id_text}\n"
            f"context:\n{context_preview or '(empty)'}"
        )
    return "\n\n".join(lines)


@tool
def retrieve_pg_knowledge(
    query: str,
    top_k: int = 0,
    book_id: str = "",
    chapter: str = "",
    context_window: int = -1,
    rerank_candidates: int = 0,
) -> str:
    """Retrieve relevant chunks from PostgreSQL + pgvector knowledge base.

    Requires environment variable PGVECTOR_DSN.
    Optional env vars: PGVECTOR_TABLE, PGVECTOR_EMBEDDING_MODEL.

    Args:
        query: Search query text.
        top_k: Number of chunks to return. <=0 时读取环境变量 KNOWLEDGE_TOP_K_DEFAULT。
        book_id: Optional exact filter of book_id.
        chapter: Optional exact filter of chapter.
        context_window: Neighbor chunk window for returning longer context around a hit.
            <0 时读取环境变量 KNOWLEDGE_CONTEXT_WINDOW_DEFAULT。
        rerank_candidates: Candidate pool size for LLM relevance reranking before final output.
            <=0 时读取环境变量 KNOWLEDGE_RERANK_CANDIDATES_DEFAULT。
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

    table = pgvector_table()
    embedding_model = pgvector_embedding_model()

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
        character_candidates = store.get_character_candidates(book_id=clean_book_id, limit=200)
        query_analysis = _analyze_query_with_cache(
            query=clean_query,
            book_id=clean_book_id,
            candidate_characters=character_candidates,
        )
        effective_top_k = int(top_k)
        if effective_top_k <= 0:
            effective_top_k = top_k_default()

        effective_context_window = int(context_window)
        if effective_context_window < 0:
            effective_context_window = context_window_default()

        effective_rerank_candidates = int(rerank_candidates)
        if effective_rerank_candidates <= 0:
            effective_rerank_candidates = rerank_candidates_default()

        hits = store.similarity_search(
            query_embedding=query_vecs[0],
            top_k=max(effective_top_k, effective_rerank_candidates),
            book_id=clean_book_id,
            chapter=clean_chapter,
        )

        if not hits:
            result = "知识库未命中相关内容"
            record_knowledge_retrieval(clean_query, result)
            return result

        # Step 1: deterministic local rerank (character continuity + timeline coherence).
        locally_reranked, local_aux = _apply_role_timeline_rerank(hits, query_analysis)

        reranked_hits = locally_reranked[:effective_top_k]
        rerank_aux: dict = {}
        try:
            llm_reranked_hits, rerank_aux = _rerank_hits_with_llm(
                query=clean_query,
                hits=locally_reranked,
                top_k=effective_top_k,
            )
            # Blend LLM and local signals to avoid losing deterministic continuity constraints.
            w_llm, w_local = _blend_weights()
            blended: list[dict] = []
            for hit in llm_reranked_hits:
                llm_score = float(hit.get("rerank_score", 0.0) or 0.0)
                local_score = float(hit.get("local_rerank_score", 0.0) or 0.0)
                hit2 = dict(hit)
                hit2["blended_rerank_score"] = round(w_llm * llm_score + w_local * local_score, 2)
                blended.append(hit2)
            blended.sort(key=lambda h: float(h.get("blended_rerank_score", 0.0)), reverse=True)
            reranked_hits = blended[:effective_top_k]
        except Exception as exc:  # noqa: BLE001
            reranked_hits = locally_reranked[:effective_top_k]
            rerank_aux = {
                "rerank_prompt": "",
                "rerank_response": "",
                "rerank_used": False,
                "rerank_fallback": f"exception:{exc}",
            }

        body = _format_hits(
            store=store,
            hits=reranked_hits,
            max_items=effective_top_k,
            context_window=max(0, effective_context_window),
        )
        result = f"知识库检索结果:\n{body}"
        auxiliary = {
            "rerank_used": bool(rerank_aux.get("rerank_used", False)),
            "rerank_fallback": str(rerank_aux.get("rerank_fallback", "")).strip(),
            "rerank_prompt": str(rerank_aux.get("rerank_prompt", ""))[:1200],
            "rerank_response": str(rerank_aux.get("rerank_response", ""))[:1200],
            "local_rerank_used": bool(local_aux.get("local_rerank_used", False)),
            "query_characters": local_aux.get("query_characters", []),
            "timeline_intent": str(local_aux.get("timeline_intent", "unknown")),
            "query_analysis_source": str(local_aux.get("analysis_source", "unknown")),
            "query_analysis_confidence": float(local_aux.get("analysis_confidence", 0.0) or 0.0),
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
