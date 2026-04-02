"""配置层：sync_bookshelf 的环境变量读取逻辑。"""

import os

from app.config import get_env_bool, get_env_int


def chapter_analysis_concurrency() -> int:
    return get_env_int("KNOWLEDGE_CHAPTER_ANALYSIS_CONCURRENCY", default=4, min_value=1)


def incremental_enabled() -> bool:
    return get_env_bool("KNOWLEDGE_SYNC_INCREMENTAL", default=True)


def auto_delete_removed() -> bool:
    return get_env_bool("KNOWLEDGE_SYNC_AUTO_DELETE_REMOVED", default=True)


def hash_check_enabled() -> bool:
    return get_env_bool("KNOWLEDGE_SYNC_HASH_CHECK", default=True)


def show_incremental_stats() -> bool:
    return get_env_bool("KNOWLEDGE_SYNC_SHOW_INCREMENTAL_STATS", default=True)


def state_file_path() -> str:
    return (
        os.getenv("KNOWLEDGE_SYNC_STATE_FILE", "memory/bookshelf_sync_state.json").strip()
        or "memory/bookshelf_sync_state.json"
    )
