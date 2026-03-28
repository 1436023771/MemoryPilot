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


def _extract_tokens(text: str) -> set[str]:
    """Extract comparable tokens from mixed Chinese/English text."""
    lowered = (text or "").lower()
    words = re.findall(r"[a-z0-9]+", lowered)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text or "")
    return {t for t in words + cjk_chars if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _remove_recency_modifiers(text: str) -> str:
    """Remove common recency modifiers without relying on topic-specific words."""
    out = text
    for token in ("最新", "最近", "今年", "近期", "当前", "目前"):
        out = out.replace(token, " ")
    out = re.sub(r"\b(recent|latest|current|today|this year|new)\b", " ", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def _strip_punctuation_noise(text: str) -> str:
    out = re.sub(r"[!?！？,.，。;；:：()\[\]{}<>\"'`~@#$%^&*_+=|\\/-]+", " ", text or "")
    return re.sub(r"\s+", " ", out).strip()


def _shorten_by_clause(text: str) -> str:
    parts = [p.strip() for p in re.split(r"[。！？!?；;，,：:\n]", text or "") if p.strip()]
    if not parts:
        return ""
    return max(parts, key=lambda p: len(_extract_tokens(p)))


def _truncate_for_search(text: str, max_chars: int = 36) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip()


def _noise_penalty(text: str) -> float:
    raw = text or ""
    if not raw:
        return 1.0

    punct_count = len(re.findall(r"[^\w\s\u4e00-\u9fff]", raw))
    punct_ratio = punct_count / max(len(raw), 1)
    digit_count = len(re.findall(r"\d", raw))
    digit_ratio = digit_count / max(len(raw), 1)
    url_like = bool(re.search(r"https?://|www\.|\.[a-z]{2,}", raw.lower()))

    penalty = min(1.0, punct_ratio * 1.2 + digit_ratio * 0.6 + (0.2 if url_like else 0.0))
    return penalty


def _brevity_score(text: str, target_min: int = 8, target_max: int = 36) -> float:
    length = len(re.sub(r"\s+", "", text or ""))
    if length == 0:
        return 0.0
    if target_min <= length <= target_max:
        return 1.0
    if length < target_min:
        return max(0.2, length / target_min)
    overflow = length - target_max
    return max(0.0, 1.0 - overflow / target_max)


def _score_candidate(base: str, candidate: str, already_selected: list[str]) -> float:
    base_tokens = _extract_tokens(base)
    cand_tokens = _extract_tokens(candidate)
    preserve = _jaccard(base_tokens, cand_tokens)
    brevity = _brevity_score(candidate)
    noise = _noise_penalty(candidate)

    diversity = 1.0
    if already_selected:
        similarities = [_jaccard(cand_tokens, _extract_tokens(item)) for item in already_selected]
        diversity = 1.0 - max(similarities)

    # Weighted rule score: preserve intent + concise phrasing + non-duplicated variants.
    return 0.45 * preserve + 0.25 * brevity + 0.15 * diversity - 0.15 * noise


def _build_query_variants(query: str) -> list[str]:
    """Build generic fallback query variants via rule-based scoring."""
    base = _normalize_query(query)
    if not base:
        return []

    raw_candidates = [
        base,
        _remove_recency_modifiers(base),
        _strip_punctuation_noise(base),
        _shorten_by_clause(base),
        _truncate_for_search(base, max_chars=36),
        _truncate_for_search(_remove_recency_modifiers(base), max_chars=36),
    ]

    deduped_candidates: list[str] = []
    seen: set[str] = set()
    for item in raw_candidates:
        normalized = _normalize_query(item)
        key = normalized.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped_candidates.append(normalized)

    selected: list[str] = [base]
    remaining = [item for item in deduped_candidates if item != base]
    while remaining and len(selected) < 5:
        scored = [(_score_candidate(base, item, selected), item) for item in remaining]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_item = scored[0]
        if best_score < 0.35:
            break
        selected.append(best_item)
        remaining = [item for item in remaining if item != best_item]

    return selected[:5]


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
        from ddgs import DDGS

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

        body = _format_items(collected, max_items=6)
        if not body:
            body = (
                f"未找到 '{clean_query}' 的搜索结果。\n"
                "建议改短关键词、改英文关键词、或加上权威站点词（如官网域名）。"
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
