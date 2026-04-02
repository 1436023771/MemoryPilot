"""配置层：LangGraph 流程环境变量读取逻辑。"""

import os

from app.config import get_env_int


DEFAULT_MAX_HISTORY_TOKENS = 8000


def langchain_project() -> str:
    return os.getenv("LANGCHAIN_PROJECT", "agent-langgraph").strip() or "agent-langgraph"


def max_history_tokens() -> int:
    raw = os.getenv("MAX_HISTORY_TOKENS", str(DEFAULT_MAX_HISTORY_TOKENS)).strip()
    try:
        value = int(raw)
        if value < 0:
            return DEFAULT_MAX_HISTORY_TOKENS
        return value
    except (ValueError, TypeError):
        return DEFAULT_MAX_HISTORY_TOKENS


def top_k_default() -> int:
    return get_env_int("KNOWLEDGE_TOP_K_DEFAULT", default=5, min_value=1)


def context_window_default() -> int:
    return get_env_int("KNOWLEDGE_CONTEXT_WINDOW_DEFAULT", default=3, min_value=0)


def rerank_candidates_default() -> int:
    return get_env_int("KNOWLEDGE_RERANK_CANDIDATES_DEFAULT", default=14, min_value=2)
