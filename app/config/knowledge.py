"""配置层：知识检索和重排相关环境变量读取逻辑。"""

import os

from app.config import get_env_float, get_env_int


def pgvector_table() -> str:
    return os.getenv("PGVECTOR_TABLE", "knowledge_chunks").strip() or "knowledge_chunks"


def pgvector_embedding_model() -> str:
    return (
        os.getenv(
            "PGVECTOR_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ).strip()
        or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def local_rerank_weights() -> tuple[float, float, float]:
    semantic = get_env_float("KNOWLEDGE_LOCAL_RERANK_WEIGHT_SEMANTIC", 0.5, min_value=0.0)
    character = get_env_float("KNOWLEDGE_LOCAL_RERANK_WEIGHT_CHARACTER", 0.35, min_value=0.0)
    timeline = get_env_float("KNOWLEDGE_LOCAL_RERANK_WEIGHT_TIMELINE", 0.15, min_value=0.0)
    weight_sum = semantic + character + timeline
    if weight_sum <= 0.0:
        return 0.5, 0.35, 0.15
    return semantic / weight_sum, character / weight_sum, timeline / weight_sum


def blend_weights() -> tuple[float, float]:
    llm = get_env_float("KNOWLEDGE_BLEND_WEIGHT_LLM", 0.6, min_value=0.0)
    local = get_env_float("KNOWLEDGE_BLEND_WEIGHT_LOCAL", 0.4, min_value=0.0)
    weight_sum = llm + local
    if weight_sum <= 0.0:
        return 0.6, 0.4
    return llm / weight_sum, local / weight_sum


def top_k_default() -> int:
    return get_env_int("KNOWLEDGE_TOP_K_DEFAULT", default=5, min_value=1)


def context_window_default() -> int:
    return get_env_int("KNOWLEDGE_CONTEXT_WINDOW_DEFAULT", default=2, min_value=0)


def rerank_candidates_default() -> int:
    return get_env_int("KNOWLEDGE_RERANK_CANDIDATES_DEFAULT", default=12, min_value=2)
