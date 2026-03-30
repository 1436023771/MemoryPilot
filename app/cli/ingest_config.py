"""配置层：ingest_pg_knowledge 的所有环境变量读取逻辑集中于此模块。

职责：
- 读取和验证环境变量
- 提供合理的默认值
- 不包含业务逻辑，仅负责参数获取
"""

from app.core.config import get_env_int


def chapter_analysis_concurrency() -> int:
    """读取章节异步分析的并发数。"""
    return get_env_int("KNOWLEDGE_CHAPTER_ANALYSIS_CONCURRENCY", default=4, min_value=1)
