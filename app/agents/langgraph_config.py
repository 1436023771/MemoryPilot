"""配置层：LangGraph流程的所有环境变量读取逻辑集中于此模块。

职责：
- 读取和验证LangGraph流程相关的环境变量
- 提供合理的默认值
- 不包含业务逻辑，仅负责参数获取
"""

import os
from app.core.config import get_env_int

# LangGraph默认的最大历史上下文令牌数
DEFAULT_MAX_HISTORY_TOKENS = 8000


def langchain_project() -> str:
    """读取LangChain项目名称。"""
    return os.getenv("LANGCHAIN_PROJECT", "agent-langgraph").strip() or "agent-langgraph"


def max_history_tokens() -> int:
    """读取对话历史的最大令牌数限制。"""
    raw = os.getenv("MAX_HISTORY_TOKENS", str(DEFAULT_MAX_HISTORY_TOKENS)).strip()
    try:
        value = int(raw)
        if value < 0:
            return DEFAULT_MAX_HISTORY_TOKENS
        return value
    except (ValueError, TypeError):
        return DEFAULT_MAX_HISTORY_TOKENS


def top_k_default() -> int:
    """读取知识检索的默认返回结果数。"""
    return get_env_int("KNOWLEDGE_TOP_K_DEFAULT", default=5, min_value=1)


def context_window_default() -> int:
    """读取知识检索的默认上下文窗口大小（chunks数）。"""
    return get_env_int("KNOWLEDGE_CONTEXT_WINDOW_DEFAULT", default=3, min_value=0)


def rerank_candidates_default() -> int:
    """读取知识检索的默认重排候选数。"""
    return get_env_int("KNOWLEDGE_RERANK_CANDIDATES_DEFAULT", default=14, min_value=2)
