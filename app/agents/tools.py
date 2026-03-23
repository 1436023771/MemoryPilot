"""
LangChain Tools for Agent: web search via DuckDuckGo.
"""

import re

from langchain.tools import tool

# 全局搜索记录：存储本轮 Agent 执行中的所有搜索信息
_search_log: list[dict] = []


def get_search_log() -> list[dict]:
    """获取本轮搜索记录"""
    return _search_log.copy()


def clear_search_log() -> None:
    """清空搜索记录（每轮开始时调用）"""
    global _search_log
    _search_log = []


def record_search(query: str, results: str) -> None:
    """记录搜索查询和结果"""
    global _search_log
    _search_log.append({"query": query, "results": results})


def _normalize_query(query: str) -> str:
    """清理查询字符串，减少无效空白和过长输入。"""
    compact = re.sub(r"\s+", " ", query or "").strip()
    return compact[:180]


def _build_query_variants(query: str) -> list[str]:
    """生成降级查询变体，提升 DDG 命中率。"""
    base = _normalize_query(query)
    if not base:
        return []

    variants: list[str] = [base]

    # 去掉常见噪声词，构造更短关键词查询。
    shorter = base
    for token in ("最新", "最近", "今年", "近期", "行程", "访问", "访问地点", "去过哪些地方"):
        shorter = shorter.replace(token, " ")
    shorter = re.sub(r"\b(schedule|itinerary|recent|latest)\b", " ", shorter, flags=re.IGNORECASE)
    shorter = re.sub(r"\s+", " ", shorter).strip()
    if shorter and shorter != base:
        variants.append(shorter)

    # 中文涉及美国总统场景时，补充英文关键词变体。
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

    # 去重并保持顺序。
    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped[:5]


def _format_items(items: list[dict], max_items: int = 6) -> str:
    """将搜索结果格式化为可读文本。"""
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
    """对总统行程类问题做轻量相关性过滤，减少明显跑题结果。"""
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
    """搜索网络获取实时信息。使用 DuckDuckGo 搜索引擎。
    
    Args:
        query: 搜索关键词或问题
    """
    clean_query = _normalize_query(query)
    if not clean_query:
        result_text = "搜索出错: 空查询"
        record_search(query, result_text)
        return result_text

    try:
        try:
            from ddgs import DDGS  # 新包名，优先使用
        except ImportError:
            from duckduckgo_search import DDGS  # 兼容旧包

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

            # 一次检索不足时，尝试更宽松参数再检索一次。
            if len(collected) < 2:
                for q in variants[:3]:
                    text_hits_fallback = list(ddgs.text(q, max_results=8))
                    add_results(text_hits_fallback)
                    if len(collected) >= 3:
                        break

            # 文本检索不足时，补充新闻检索。
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

        # 记录搜索信息供 GUI 显示
        record_search(clean_query, result_text)

        return result_text
    except ImportError:
        error_msg = "搜索工具不可用: 请安装 ddgs（或 duckduckgo-search）"
        record_search(query, error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"搜索出错: {str(e)}"
        record_search(query, error_msg)
        return error_msg


__all__ = ["web_search", "get_search_log", "clear_search_log", "record_search"]
