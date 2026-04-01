"""配置层：sync_bookshelf 的所有环境变量读取逻辑集中于此模块。

职责：
- 读取和验证环境变量
- 提供合理的默认值
- 不包含业务逻辑，仅负责参数获取
"""

import os
from app.core.config import get_env_bool, get_env_int


def chapter_analysis_concurrency() -> int:
    """读取章节异步分析的并发数。"""
    return get_env_int("KNOWLEDGE_CHAPTER_ANALYSIS_CONCURRENCY", default=4, min_value=1)


def incremental_enabled() -> bool:
    """读取是否启用增量同步模式。"""
    return get_env_bool("KNOWLEDGE_SYNC_INCREMENTAL", default=True)


def auto_delete_removed() -> bool:
    """读取是否自动删除已移除文件的chunks。"""
    return get_env_bool("KNOWLEDGE_SYNC_AUTO_DELETE_REMOVED", default=True)


def hash_check_enabled() -> bool:
    """读取是否使用内容哈希验证文件变更。"""
    return get_env_bool("KNOWLEDGE_SYNC_HASH_CHECK", default=True)


def show_incremental_stats() -> bool:
    """读取是否显示增量同步的详细统计信息。"""
    return get_env_bool("KNOWLEDGE_SYNC_SHOW_INCREMENTAL_STATS", default=True)


def state_file_path() -> str:
    """读取增量同步状态文件的路径。"""
    return (
        os.getenv("KNOWLEDGE_SYNC_STATE_FILE", "memory/bookshelf_sync_state.json").strip()
        or "memory/bookshelf_sync_state.json"
    )
