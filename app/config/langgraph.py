"""配置层：LangGraph 流程环境变量读取逻辑。"""

from app.config import get_env_str
from app.config.knowledge import (
    context_window_default as knowledge_context_window_default,
    rerank_candidates_default as knowledge_rerank_candidates_default,
    top_k_default as knowledge_top_k_default,
)


DEFAULT_MAX_HISTORY_TOKENS = 8000


def langchain_project() -> str:
    return get_env_str("LANGCHAIN_PROJECT", "agent-langgraph") or "agent-langgraph"


def max_history_tokens() -> int:
    raw = get_env_str("MAX_HISTORY_TOKENS", str(DEFAULT_MAX_HISTORY_TOKENS))
    try:
        value = int(raw)
        if value < 0:
            return DEFAULT_MAX_HISTORY_TOKENS
        return value
    except (ValueError, TypeError):
        return DEFAULT_MAX_HISTORY_TOKENS


def top_k_default() -> int:
    return knowledge_top_k_default()


def context_window_default() -> int:
    return knowledge_context_window_default()


def rerank_candidates_default() -> int:
    return knowledge_rerank_candidates_default()
