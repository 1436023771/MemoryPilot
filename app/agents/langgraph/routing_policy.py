from __future__ import annotations


def _detect_route(question: str) -> str:
    q = (question or "").lower()
    knowledge_keys = [
        "书",
        "章节",
        "chapter",
        "剧情",
        "角色",
        "设定",
        "文档",
        "项目",
        "book",
        "timeline",
        "time line",
    ]
    if any(k in q for k in knowledge_keys):
        return "knowledge"
    return "direct"


__all__ = ["_detect_route"]
