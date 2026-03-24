# Agent 时间感知与网络搜索改进

## 已完成改进

### 1. Agent Prompt 注入当前时间

- system prompt 注入当前本地时间。
- 用户输入包装里包含 Current local time。
- 对 this year、recently 这类相对时间表达有更稳定的解释。

对应实现：
- app/agents/chains.py
    - _format_agent_user_input
    - build_qa_chain

### 2. web_search 稳定性增强

主要策略：
- 查询标准化（去噪、限长）
- 查询变体生成（中英混合回退）
- 多轮回退检索（text -> fallback text -> news）
- 主题相关性过滤（总统/白宫等场景）

对应实现：
- app/agents/tools.py
    - _normalize_query
    - _build_query_variants
    - _filter_relevant_items
    - web_search

### 3. GUI 搜索可观测性

- 每轮调用前 clear_search_log。
- 回答完成后读取 get_search_log。
- 侧边栏展示查询词和结果摘要。

对应实现：
- app/ui/gui_chat.py

## 使用示例

### CLI

```bash
python -m app.cli.main "美国总统最近的行程是什么？"
```

### GUI

```bash
python -m app.ui.gui_chat
```

说明：当前代码已统一为 agent-only，CLI 不再提供 cot/brief 相关参数。

## 验证建议

1. 用时效性问题触发搜索（如近期新闻、公开日程）。
2. 在 GUI 侧边栏确认 Web Search 段落是否出现。
3. 对比不同问法，验证回退查询是否提升命中率。
