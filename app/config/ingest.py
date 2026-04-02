"""配置层：ingest_pg_knowledge 的环境变量读取逻辑。"""

from app.config import get_env_int


def chapter_analysis_concurrency() -> int:
    return get_env_int("KNOWLEDGE_CHAPTER_ANALYSIS_CONCURRENCY", default=4, min_value=1)
