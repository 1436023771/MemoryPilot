"""配置层：知识检索和重排相关的所有环境变量读取逻辑集中于此模块。

职责：
- 读取和验证知识检索相关的环境变量
- 提供合理的默认值
- 不包含业务逻辑，仅负责参数获取
"""

import os
from app.core.config import get_env_float, get_env_int


def pgvector_table() -> str:
    """读取pgvector表名。"""
    return os.getenv("PGVECTOR_TABLE", "knowledge_chunks").strip() or "knowledge_chunks"


def pgvector_embedding_model() -> str:
    """读取embedding模型名称。"""
    return (
        os.getenv(
            "PGVECTOR_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ).strip()
        or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def local_rerank_weights() -> tuple[float, float, float]:
    """读取本地重排的权重：语义、角色、时间线。
    
    返回值为归一化后的权重三元组，确保和为1。
    """
    semantic = get_env_float("KNOWLEDGE_LOCAL_RERANK_WEIGHT_SEMANTIC", 0.5, min_value=0.0)
    character = get_env_float("KNOWLEDGE_LOCAL_RERANK_WEIGHT_CHARACTER", 0.35, min_value=0.0)
    timeline = get_env_float("KNOWLEDGE_LOCAL_RERANK_WEIGHT_TIMELINE", 0.15, min_value=0.0)
    weight_sum = semantic + character + timeline
    if weight_sum <= 0.0:
        return 0.5, 0.35, 0.15
    return semantic / weight_sum, character / weight_sum, timeline / weight_sum


def blend_weights() -> tuple[float, float]:
    """读取结果混合权重：LLM、本地重排。
    
    返回值为归一化后的权重二元组，确保和为1。
    """
    llm = get_env_float("KNOWLEDGE_BLEND_WEIGHT_LLM", 0.6, min_value=0.0)
    local = get_env_float("KNOWLEDGE_BLEND_WEIGHT_LOCAL", 0.4, min_value=0.0)
    weight_sum = llm + local
    if weight_sum <= 0.0:
        return 0.6, 0.4
    return llm / weight_sum, local / weight_sum


def top_k_default() -> int:
    """读取知识检索的默认返回结果数。"""
    return get_env_int("KNOWLEDGE_TOP_K_DEFAULT", default=5, min_value=1)


def context_window_default() -> int:
    """读取知识检索的默认上下文窗口大小（chunks数）。"""
    return get_env_int("KNOWLEDGE_CONTEXT_WINDOW_DEFAULT", default=2, min_value=0)


def rerank_candidates_default() -> int:
    """读取知识检索的默认重排候选数。"""
    return get_env_int("KNOWLEDGE_RERANK_CANDIDATES_DEFAULT", default=12, min_value=2)
