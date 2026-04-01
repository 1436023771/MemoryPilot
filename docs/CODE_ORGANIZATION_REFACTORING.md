## 代码组织重构：环境变量获取与业务逻辑分离

### 重构目标
将环境变量读取配置逻辑与业务实现逻辑分离到不同的文件中，提高代码可维护性和清晰度。

### 创建的新配置模块

#### 1. `app/cli/sync_bookshelf_config.py`
**职责**：集中管理 `sync_bookshelf.py` 的环境变量读取

**导出函数**：
- `chapter_analysis_concurrency()` - 章节异步分析的并发数
- `incremental_enabled()` - 是否启用增量同步模式
- `auto_delete_removed()` - 是否自动删除已移除文件的chunks
- `hash_check_enabled()` - 是否使用内容哈希验证文件变更
- `show_incremental_stats()` - 是否显示增量统计信息
- `state_file_path()` - 增量同步状态文件的路径

#### 2. `app/agents/knowledge_config.py`
**职责**：集中管理知识检索和重排相关的环境变量

**导出函数**：
- `pgvector_table()` - pgvector 表名
- `pgvector_embedding_model()` - embedding 模型名称
- `local_rerank_weights()` - 本地重排权重（语义、角色、时间线），返回归一化的权重三元组
- `blend_weights()` - 结果混合权重（LLM、本地重排），返回归一化的权重二元组
- `top_k_default()` - 默认返回结果数
- `context_window_default()` - 默认上下文窗口大小
- `rerank_candidates_default()` - 默认重排候选数

#### 3. `app/agents/langgraph_config.py`
**职责**：集中管理 LangGraph 流程的环境变量

**导出函数**：
- `langchain_project()` - LangChain 项目名称
- `max_history_tokens()` - 对话历史最大令牌数限制
- `top_k_default()` - 知识检索默认返回数
- `context_window_default()` - 默认上下文窗口大小
- `rerank_candidates_default()` - 默认重排候选数

#### 4. `app/cli/ingest_config.py`
**职责**：集中管理 `ingest_pg_knowledge.py` 的环境变量

**导出函数**：
- `chapter_analysis_concurrency()` - 章节异步分析的并发数

### 修改的现有文件

#### `app/cli/sync_bookshelf.py`
- **删除**：`_chapter_analysis_concurrency()`、`_default_incremental_enabled()`、`_default_auto_delete_removed()`、`_default_hash_check()`、`_default_show_incremental_stats()`、`_default_state_file_path()` 函数
- **新增导入**：从 `app.cli.sync_bookshelf_config` 导入所有配置函数
- **更新**：所有环境变量读取调用改为使用导入的配置函数

#### `app/agents/tools_pg_knowledge.py`
- **删除**：`get_env_float`、`get_env_int` 从导入列表中移除
- **新增导入**：从 `app.agents.knowledge_config` 导入所有配置函数
- **简化**：`_local_rerank_weights()` 和 `_blend_weights()` 现在直接代理到配置模块的函数
- **更新**：`os.getenv("PGVECTOR_TABLE"...)` 改为 `pgvector_table()`；`os.getenv("PGVECTOR_EMBEDDING_MODEL"...)` 改为 `pgvector_embedding_model()`
- **更新**：所有 `get_env_int("KNOWLEDGE_...")` 调用改为调用相应的配置函数

#### `app/agents/langgraph_flow.py`
- **删除**：`get_env_int` 从导入列表中移除
- **新增导入**：从 `app.agents.langgraph_config` 导入所有配置函数
- **简化**：`_graph_config()` 现在调用 `langchain_project()` 而不是 `os.getenv()`
- **简化**：`_history_token_limit()` 现在直接调用 `max_history_tokens()`
- **更新**：所有 `get_env_int("KNOWLEDGE_...")` 调用改为调用相应的配置函数

#### `app/cli/ingest_pg_knowledge.py`
- **删除**：`get_env_int` 从导入列表中移除和 `_chapter_analysis_concurrency()` 函数
- **新增导入**：从 `app.cli.ingest_config` 导入 `chapter_analysis_concurrency`
- **更新**：所有 `_chapter_analysis_concurrency()` 调用改为 `chapter_analysis_concurrency()`

### 设计原则

1. **单一职责**：每个配置模块只负责其对应功能模块的环境变量读取
2. **无业务逻辑**：配置模块不包含任何业务实现逻辑，仅负责参数获取和验证
3. **统一入口**：所有环境变量访问必须通过配置模块，避免散落在业务代码中
4. **明确的映射**：配置模块文件位置与使用它的业务模块位置相近或配对
5. **向下兼容**：所有现有功能保持不变，仅代码结构改善

### 带来的好处

✅ **提高可维护性**
- 环境变量定义和默认值集中管理
- 修改配置无需查找散落的代码位置

✅ **便于测试**
- 可以轻松 mock 配置模块进行单元测试
- 配置模块本身更易于测试

✅ **提高代码清晰度**
- 业务模块代码精简，专注核心逻辑
- 一目了然地看出模块依赖哪些配置

✅ **易于扩展**
- 添加新的环境变量只需在对应配置模块中添加
- 新增模块可以轻松遵循既有的组织模式

### 测试验证

✅ **全部 81 个测试通过**
- 包括所有原有的测试套件
- 无任何功能回归
- 确保重构的正确性

### 示例：配置函数使用模式

**之前**：环境变量读取散布在业务代码中
```python
# 在 sync_bookshelf.py 中
def _chapter_analysis_concurrency() -> int:
    return get_env_int("KNOWLEDGE_CHAPTER_ANALYSIS_CONCURRENCY", default=4, min_value=1)

def _default_incremental_enabled() -> bool:
    return get_env_bool("KNOWLEDGE_SYNC_INCREMENTAL", default=True)

# 在业务代码中使用
semaphore = asyncio.Semaphore(_chapter_analysis_concurrency())
runtime['incremental'] = _default_incremental_enabled()
```

**之后**：环境变量读取集中在配置模块
```python
# 在 sync_bookshelf_config.py 中
def chapter_analysis_concurrency() -> int:
    return get_env_int("KNOWLEDGE_CHAPTER_ANALYSIS_CONCURRENCY", default=4, min_value=1)

def incremental_enabled() -> bool:
    return get_env_bool("KNOWLEDGE_SYNC_INCREMENTAL", default=True)

# 在 sync_bookshelf.py 中
from app.cli.sync_bookshelf_config import chapter_analysis_concurrency, incremental_enabled

semaphore = asyncio.Semaphore(chapter_analysis_concurrency())
runtime['incremental'] = incremental_enabled()
```
