"""
Web search tool and search trace helpers.
"""

from __future__ import annotations

import re

from langchain.tools import tool

# Save all searches in current turn for GUI display.
_search_log: list[dict] = []


def get_search_log() -> list[dict]:
    """Return a copy of current turn search log."""
    return _search_log.copy()


def clear_search_log() -> None:
    """Clear search log at the start of each turn."""
    global _search_log
    _search_log = []


def record_search(query: str, results: str) -> None:
    """Record search query and result text."""
    global _search_log
    _search_log.append({"query": query, "results": results})


def _normalize_query(query: str) -> str:
    """Normalize query by compacting spaces and clipping length."""
    compact = re.sub(r"\s+", " ", query or "").strip()
    return compact[:180]


def _build_query_variants(query: str) -> list[str]:
    """Build fallback query variants to improve hit rate."""
    base = _normalize_query(query)
    if not base:
        return []

    variants: list[str] = [base]

    shorter = base
    for token in ("最新", "最近", "今年", "近期", "行程", "访问", "访问地点", "去过哪些地方"):
        shorter = shorter.replace(token, " ")
    shorter = re.sub(r"\b(schedule|itinerary|recent|latest)\b", " ", shorter, flags=re.IGNORECASE)
    shorter = re.sub(r"\s+", " ", shorter).strip()
    if shorter and shorter != base:
        variants.append(shorter)

    if any(k in base for k in ("美国总统", "拜登", "白宫")):
        variants.extend(
            [
                "U.S. President schedule White House",
                "President Biden schedule White House",
                "White House presidential schedule",
                "site:whitehouse.gov presidential schedule",
                "site:whitehouse.gov remarks president",
            ]
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped[:5]


def _format_items(items: list[dict], max_items: int = 6) -> str:
    """Format DDG items as readable lines."""
    lines: list[str] = []
    for item in items[:max_items]:
        title = str(item.get("title", "")).strip()
        body = str(item.get("body", "")).strip()
        link = str(item.get("href", "")).strip() or str(item.get("url", "")).strip()
        if not (title or body or link):
            continue
        lines.append(f"- {title}\n  {body}\n  ({link})")
    return "\n".join(lines)


def _is_presidential_query(query: str) -> bool:
    q = (query or "").lower()
    keys = ("美国总统", "拜登", "白宫", "president", "biden", "white house")
    return any(k in q for k in keys)


def _filter_relevant_items(query: str, items: list[dict]) -> list[dict]:
    """Apply light relevance filter for presidential schedule queries."""
    if not _is_presidential_query(query):
        return items

    must_have = ("president", "biden", "white house", "potus", "美国总统", "拜登", "白宫", "schedule", "trip", "visit")
    filtered: list[dict] = []
    for item in items:
        hay = (
            f"{item.get('title', '')} {item.get('body', '')} {item.get('href', '')} {item.get('url', '')}"
        ).lower()
        if any(k in hay for k in must_have):
            filtered.append(item)
    return filtered or items


@tool
def web_search(query: str) -> str:
    """Search the web for real-time information using DuckDuckGo.

    Args:
        query: Search query.
    """
    clean_query = _normalize_query(query)
    if not clean_query:
        result_text = "搜索出错: 空查询"
        record_search(query, result_text)
        return result_text

    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        attempts: list[str] = []
        collected: list[dict] = []
        seen_links: set[str] = set()

        def add_results(candidates):
            for item in candidates or []:
                link = str(item.get("href", "")).strip() or str(item.get("url", "")).strip()
                key = link or f"{item.get('title', '')}|{item.get('body', '')}"
                if key in seen_links:
                    continue
                seen_links.add(key)
                collected.append(item)

        variants = _build_query_variants(clean_query)
        with DDGS() as ddgs:
            for q in variants:
                attempts.append(q)
                text_hits = list(ddgs.text(q, max_results=5, region="wt-wt", safesearch="moderate"))
                add_results(text_hits)
                if len(collected) >= 4:
                    break

            if len(collected) < 2:
                for q in variants[:3]:
                    text_hits_fallback = list(ddgs.text(q, max_results=8))
                    add_results(text_hits_fallback)
                    if len(collected) >= 3:
                        break

            if len(collected) < 3:
                for q in variants[:3]:
                    news_hits = list(ddgs.news(q, max_results=5))
                    add_results(news_hits)
                    if len(collected) >= 4:
                        break

        filtered = _filter_relevant_items(clean_query, collected)
        body = _format_items(filtered, max_items=6)
        if not body:
            body = (
                f"未找到 '{clean_query}' 的搜索结果。\n"
                "建议改短关键词、改英文关键词、或加上权威站点词（如 whitehouse.gov）。"
            )

        result_text = "尝试查询:\n" + "\n".join(f"- {item}" for item in attempts) + "\n\n搜索结果:\n" + body
        record_search(clean_query, result_text)
        return result_text
    except ImportError:
        error_msg = "搜索工具不可用: 请安装 ddgs（或 duckduckgo-search）"
        record_search(query, error_msg)
        return error_msg
    except Exception as e:  # noqa: BLE001
        error_msg = f"搜索出错: {str(e)}"
        record_search(query, error_msg)
        return error_msg


__all__ = ["web_search", "get_search_log", "clear_search_log", "record_search"]
