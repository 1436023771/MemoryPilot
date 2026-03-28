from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import get_settings

try:
    from langsmith import tracing_context
except Exception:  # noqa: BLE001
    tracing_context = None


_NARRATIVE_CACHE: dict[str, dict[str, Any]] = {}


def _slugify(text: str) -> str:
    raw = (text or "").strip().lower()
    out: list[str] = []
    prev_dash = False
    for ch in raw:
        is_ascii_alnum = ("0" <= ch <= "9") or ("a" <= ch <= "z")
        is_cjk = "\u4e00" <= ch <= "\u9fff"
        if is_ascii_alnum or is_cjk:
            out.append(ch)
            prev_dash = False
            continue
        if not prev_dash:
            out.append("-")
            prev_dash = True
    slug = "".join(out).strip("-")
    return slug or "unknown"


def _dedupe_keep_order(items: list[str], max_items: int = 24) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= max_items:
            break
    return out


def _normalize_relationship_edges(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    edges: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        src = str(item.get("source", "")).strip()
        dst = str(item.get("target", "")).strip()
        rel = str(item.get("relation", "")).strip()
        if not src or not dst:
            continue
        edges.append({"source": src, "target": dst, "relation": rel})
        if len(edges) >= 24:
            break
    return edges


def _build_empty_analysis() -> dict[str, Any]:
    return {
        "narrative_context": "present",
        "time_markers": [],
        "character_mentions": [],
        "relationship_edges": [],
    }


def _normalize_analysis_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _build_empty_analysis()

    narrative_context = str(payload.get("narrative_context", "present")).strip().lower()
    if narrative_context not in {"present", "flashback", "dream", "reported"}:
        narrative_context = "present"

    time_markers_raw = payload.get("time_markers", [])
    character_mentions_raw = payload.get("character_mentions", [])

    if isinstance(time_markers_raw, list):
        time_markers = _dedupe_keep_order([str(x).strip() for x in time_markers_raw if str(x).strip()], max_items=16)
    else:
        time_markers = []

    if isinstance(character_mentions_raw, list):
        character_mentions = _dedupe_keep_order([str(x).strip() for x in character_mentions_raw if str(x).strip()], max_items=20)
    else:
        character_mentions = []

    relationship_edges = _normalize_relationship_edges(payload.get("relationship_edges", []))

    return {
        "narrative_context": narrative_context,
        "time_markers": time_markers,
        "character_mentions": character_mentions,
        "relationship_edges": relationship_edges,
    }


def _call_llm_for_analysis(content: str) -> dict[str, Any]:
    settings = get_settings()
    model = ChatOpenAI(
        model=settings.model_name,
        temperature=0.0,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )

    prompt = (
        "你是小说叙事结构抽取器。请基于文本提取结构化信息，并仅输出 JSON。\n"
        "字段要求：\n"
        "- narrative_context: one of [present, flashback, dream, reported]\n"
        "- time_markers: 时间表达数组（最多16项）\n"
        "- character_mentions: 角色名数组（最多20项）\n"
        "- relationship_edges: 关系边数组（最多24项），元素格式"
        " {\"source\":\"角色A\",\"target\":\"角色B\",\"relation\":\"关系\"}\n"
        "不要输出任何额外文字。\n\n"
        "文本内容:\n"
        f"{content}\n"
    )

    if tracing_context is not None:
        with tracing_context(enabled=False):
            response = model.invoke(prompt)
    else:
        response = model.invoke(prompt)

    content_text = str(getattr(response, "content", response)).strip()
    if content_text.startswith("```json"):
        content_text = content_text[7:]
    if content_text.startswith("```"):
        content_text = content_text[3:]
    if content_text.endswith("```"):
        content_text = content_text[:-3]
    content_text = content_text.strip()

    payload = json.loads(content_text)
    return _normalize_analysis_payload(payload)


def _call_llm_for_batch_analysis(contents: list[str]) -> list[dict[str, Any]]:
    """批量分析多个文本内容，单次 LLM 调用。节省系统指令 Token。"""
    if not contents:
        return []

    settings = get_settings()
    model = ChatOpenAI(
        model=settings.model_name,
        temperature=0.0,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )

    # 构建批量提示词
    content_blocks = []
    for i, content in enumerate(contents, 1):
        content_blocks.append(f"--- 文本段落 {i} ---\n{content}\n")

    prompt = (
        "你是小说叙事结构抽取器。请分析以下文本段落，对每段提取结构化信息。\n"
        "返回一个 JSON 数组，每项按段落顺序对应。\n\n"
        "字段要求（对每段）：\n"
        "- narrative_context: one of [present, flashback, dream, reported]\n"
        "- time_markers: 时间表达数组（最多16项）\n"
        "- character_mentions: 角色名数组（最多20项）\n"
        "- relationship_edges: 关系边数组（最多24项），元素格式"
        " {\"source\":\"角色A\",\"target\":\"角色B\",\"relation\":\"关系\"}\n\n"
        "文本内容:\n"
        + "".join(content_blocks)
        + "\n只输出 JSON 数组，不要任何其他文字。"
    )

    if tracing_context is not None:
        with tracing_context(enabled=False):
            response = model.invoke(prompt)
    else:
        response = model.invoke(prompt)

    content_text = str(getattr(response, "content", response)).strip()
    if content_text.startswith("```json"):
        content_text = content_text[7:]
    if content_text.startswith("```"):
        content_text = content_text[3:]
    if content_text.endswith("```"):
        content_text = content_text[:-3]
    content_text = content_text.strip()

    try:
        payloads = json.loads(content_text)
        if not isinstance(payloads, list):
            payloads = [payloads]
    except (json.JSONDecodeError, ValueError):
        # 如果 JSON 解析失败，为所有内容返回空分析
        payloads = [_build_empty_analysis() for _ in contents]
        return payloads

    # 确保返回正确数量的结果（不足时补充空分析）
    while len(payloads) < len(contents):
        payloads.append(_build_empty_analysis())

    return [_normalize_analysis_payload(p) for p in payloads[:len(contents)]]


async def _call_llm_for_batch_analysis_async(contents: list[str]) -> list[dict[str, Any]]:
    """异步批量分析多个文本内容，单次 LLM 调用。"""
    if not contents:
        return []

    settings = get_settings()
    model = ChatOpenAI(
        model=settings.model_name,
        temperature=0.0,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )

    content_blocks = []
    for i, content in enumerate(contents, 1):
        content_blocks.append(f"--- 文本段落 {i} ---\n{content}\n")

    prompt = (
        "你是小说叙事结构抽取器。请分析以下文本段落，对每段提取结构化信息。\n"
        "返回一个 JSON 数组，每项按段落顺序对应。\n\n"
        "字段要求（对每段）：\n"
        "- narrative_context: one of [present, flashback, dream, reported]\n"
        "- time_markers: 时间表达数组（最多16项）\n"
        "- character_mentions: 角色名数组（最多20项）\n"
        "- relationship_edges: 关系边数组（最多24项），元素格式"
        " {\"source\":\"角色A\",\"target\":\"角色B\",\"relation\":\"关系\"}\n\n"
        "文本内容:\n"
        + "".join(content_blocks)
        + "\n只输出 JSON 数组，不要任何其他文字。"
    )

    if tracing_context is not None:
        with tracing_context(enabled=False):
            response = await model.ainvoke(prompt)
    else:
        response = await model.ainvoke(prompt)

    content_text = str(getattr(response, "content", response)).strip()
    if content_text.startswith("```json"):
        content_text = content_text[7:]
    if content_text.startswith("```"):
        content_text = content_text[3:]
    if content_text.endswith("```"):
        content_text = content_text[:-3]
    content_text = content_text.strip()

    try:
        payloads = json.loads(content_text)
        if not isinstance(payloads, list):
            payloads = [payloads]
    except (json.JSONDecodeError, ValueError):
        payloads = [_build_empty_analysis() for _ in contents]
        return payloads

    while len(payloads) < len(contents):
        payloads.append(_build_empty_analysis())

    return [_normalize_analysis_payload(p) for p in payloads[:len(contents)]]


def _analyze_content_with_cache(content: str) -> dict[str, Any]:
    normalized = (content or "").strip()
    if not normalized:
        return _build_empty_analysis()

    key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    cached = _NARRATIVE_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    try:
        analyzed = _call_llm_for_analysis(normalized)
    except Exception:  # noqa: BLE001
        analyzed = _build_empty_analysis()

    _NARRATIVE_CACHE[key] = dict(analyzed)
    return analyzed


def analyze_contents_batch_with_cache(contents: list[str]) -> list[dict[str, Any]]:
    """批量分析多个内容，利用缓存避免重复 LLM 调用。
    
    先检查哪些内容已缓存，只对未缓存的内容进行批量 LLM 调用。
    返回与 contents 等长的结果列表。
    """
    if not contents:
        return []

    # 规范化所有内容
    normalized_contents = [(content or "").strip() for content in contents]

    # 为每个内容计算缓存 key
    keys = [hashlib.sha256(nc.encode("utf-8")).hexdigest() for nc in normalized_contents]

    # 检查缓存
    results = []
    to_analyze_indices = []
    to_analyze_contents = []

    for i, (nc, key) in enumerate(zip(normalized_contents, keys)):
        if not nc:
            results.append(_build_empty_analysis())
        elif key in _NARRATIVE_CACHE:
            results.append(dict(_NARRATIVE_CACHE[key]))
        else:
            results.append(None)  # 占位符
            to_analyze_indices.append(i)
            to_analyze_contents.append(nc)

    # 如果有未缓存的内容，批量调用 LLM
    if to_analyze_contents:
        try:
            batch_results = _call_llm_for_batch_analysis(to_analyze_contents)
        except Exception:  # noqa: BLE001
            batch_results = [_build_empty_analysis() for _ in to_analyze_contents]

        # 将批量结果填回原位置，并存入缓存
        for idx, batch_idx in enumerate(to_analyze_indices):
            analyzed = batch_results[idx] if idx < len(batch_results) else _build_empty_analysis()
            results[batch_idx] = analyzed
            # 缓存这个结果
            key = keys[batch_idx]
            _NARRATIVE_CACHE[key] = dict(analyzed)

    return results


async def analyze_contents_batch_with_cache_async(contents: list[str]) -> list[dict[str, Any]]:
    """异步批量分析多个内容，利用缓存避免重复 LLM 调用。"""
    if not contents:
        return []

    normalized_contents = [(content or "").strip() for content in contents]
    keys = [hashlib.sha256(nc.encode("utf-8")).hexdigest() for nc in normalized_contents]

    results: list[dict[str, Any] | None] = []
    to_analyze_indices: list[int] = []
    to_analyze_contents: list[str] = []

    for i, (nc, key) in enumerate(zip(normalized_contents, keys)):
        if not nc:
            results.append(_build_empty_analysis())
        elif key in _NARRATIVE_CACHE:
            results.append(dict(_NARRATIVE_CACHE[key]))
        else:
            results.append(None)
            to_analyze_indices.append(i)
            to_analyze_contents.append(nc)

    if to_analyze_contents:
        try:
            batch_results = await _call_llm_for_batch_analysis_async(to_analyze_contents)
        except Exception:  # noqa: BLE001
            batch_results = [_build_empty_analysis() for _ in to_analyze_contents]

        for idx, batch_idx in enumerate(to_analyze_indices):
            analyzed = batch_results[idx] if idx < len(batch_results) else _build_empty_analysis()
            results[batch_idx] = analyzed
            _NARRATIVE_CACHE[keys[batch_idx]] = dict(analyzed)

    return [r if isinstance(r, dict) else _build_empty_analysis() for r in results]


def extract_time_markers(content: str) -> list[str]:
    analyzed = _analyze_content_with_cache(content)
    return _dedupe_keep_order(_dedupe_keep_order([str(x) for x in analyzed.get("time_markers", [])], max_items=16), max_items=12)


def infer_narrative_context(content: str) -> str:
    analyzed = _analyze_content_with_cache(content)
    return str(analyzed.get("narrative_context", "present"))


def extract_character_mentions(content: str) -> list[str]:
    analyzed = _analyze_content_with_cache(content)
    mentions = [str(x) for x in analyzed.get("character_mentions", [])]
    return _dedupe_keep_order(mentions, max_items=16)


@dataclass(frozen=True)
class NarrativeFields:
    chunk_order: int
    timeline_order: int
    scene_id: str
    event_id: str
    narrative_context: str
    time_markers: list[str]
    character_mentions: list[str]
    relationship_edges: list[dict[str, str]]


def build_narrative_fields(
    *,
    book_id: str,
    chapter: str,
    chunk_index: int,
    chunk_order: int,
    content: str,
) -> NarrativeFields:
    safe_book = _slugify(book_id)
    safe_chapter = _slugify(chapter)
    scene_no = (max(0, int(chunk_index)) // 4) + 1
    event_no = (max(0, int(chunk_index)) // 2) + 1
    analyzed = _analyze_content_with_cache(content)
    markers = _dedupe_keep_order([str(x) for x in analyzed.get("time_markers", [])], max_items=16)
    mentions = _dedupe_keep_order([str(x) for x in analyzed.get("character_mentions", [])], max_items=20)
    edges = _normalize_relationship_edges(analyzed.get("relationship_edges", []))
    narrative_context = str(analyzed.get("narrative_context", "present"))

    return NarrativeFields(
        chunk_order=int(chunk_order),
        timeline_order=int(chunk_order),
        scene_id=f"{safe_book}:{safe_chapter}:scene-{scene_no:04d}",
        event_id=f"{safe_book}:{safe_chapter}:event-{event_no:04d}",
        narrative_context=narrative_context,
        time_markers=markers,
        character_mentions=mentions,
        relationship_edges=edges,
    )


def build_narrative_fields_batch(
    *,
    book_id: str,
    chapter: str,
    chunk_indices: list[int],
    chunk_orders: list[int],
    contents: list[str],
) -> list[NarrativeFields]:
    """批量构建 NarrativeFields，使用单次批量 LLM 调用。
    
    参数：
    - book_id: 书籍 ID
    - chapter: 章节名
    - chunk_indices: 对应每个内容的 chunk_index
    - chunk_orders: 对应每个内容的全局 chunk_order
    - contents: 要分析的文本内容列表
    
    返回与 contents 等长的 NarrativeFields 列表。
    """
    if not contents:
        return []

    safe_book = _slugify(book_id)
    safe_chapter = _slugify(chapter)

    # 批量分析所有内容（利用缓存避免重复）
    analyzed_list = analyze_contents_batch_with_cache(contents)

    results = []
    for i, (chunk_idx, chunk_order, analyzed) in enumerate(
        zip(chunk_indices, chunk_orders, analyzed_list)
    ):
        scene_no = (max(0, int(chunk_idx)) // 4) + 1
        event_no = (max(0, int(chunk_idx)) // 2) + 1
        markers = _dedupe_keep_order([str(x) for x in analyzed.get("time_markers", [])], max_items=16)
        mentions = _dedupe_keep_order([str(x) for x in analyzed.get("character_mentions", [])], max_items=20)
        edges = _normalize_relationship_edges(analyzed.get("relationship_edges", []))
        narrative_context = str(analyzed.get("narrative_context", "present"))

        results.append(
            NarrativeFields(
                chunk_order=int(chunk_order),
                timeline_order=int(chunk_order),
                scene_id=f"{safe_book}:{safe_chapter}:scene-{scene_no:04d}",
                event_id=f"{safe_book}:{safe_chapter}:event-{event_no:04d}",
                narrative_context=narrative_context,
                time_markers=markers,
                character_mentions=mentions,
                relationship_edges=edges,
            )
        )

    return results


async def build_narrative_fields_batch_async(
    *,
    book_id: str,
    chapter: str,
    chunk_indices: list[int],
    chunk_orders: list[int],
    contents: list[str],
) -> list[NarrativeFields]:
    """异步批量构建 NarrativeFields。"""
    if not contents:
        return []

    safe_book = _slugify(book_id)
    safe_chapter = _slugify(chapter)
    analyzed_list = await analyze_contents_batch_with_cache_async(contents)

    results = []
    for chunk_idx, chunk_order, analyzed in zip(chunk_indices, chunk_orders, analyzed_list):
        scene_no = (max(0, int(chunk_idx)) // 4) + 1
        event_no = (max(0, int(chunk_idx)) // 2) + 1
        markers = _dedupe_keep_order([str(x) for x in analyzed.get("time_markers", [])], max_items=16)
        mentions = _dedupe_keep_order([str(x) for x in analyzed.get("character_mentions", [])], max_items=20)
        edges = _normalize_relationship_edges(analyzed.get("relationship_edges", []))
        narrative_context = str(analyzed.get("narrative_context", "present"))

        results.append(
            NarrativeFields(
                chunk_order=int(chunk_order),
                timeline_order=int(chunk_order),
                scene_id=f"{safe_book}:{safe_chapter}:scene-{scene_no:04d}",
                event_id=f"{safe_book}:{safe_chapter}:event-{event_no:04d}",
                narrative_context=narrative_context,
                time_markers=markers,
                character_mentions=mentions,
                relationship_edges=edges,
            )
        )

    return results
