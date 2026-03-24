## Web Search Display in GUI

### 已实现功能

右侧侧边栏会显示以下信息：
1. Query（用户本轮输入）
2. Retrieved Context（长期记忆检索结果）
3. Extracted Facts（本轮提取的候选事实）
4. Written Facts（实际写入长期记忆的事实）
5. Web Search（Agent 调用 web_search 的查询和结果摘要）

### 实现位置

1. 搜索记录模块
- 文件：app/agents/tools.py
- 关键函数：get_search_log, clear_search_log, record_search

2. GUI 集成
- 文件：app/ui/gui_chat.py
- 关键流程：
  - 每轮开始先 clear_search_log
  - Agent 执行后 get_search_log
  - 传给 _append_to_sidebar 展示

### 使用方式

1. 启动 GUI

```bash
python -m app.ui.gui_chat
```

2. 正常发起问题，若 Agent 触发了 web_search，侧边栏会出现 Web Search 段落。

3. CLI 无需额外模式参数，默认就是 agent 链路：

```bash
python -m app.cli.main "美国总统最近的行程是什么？"
```

### 数据流

```text
User Query
  -> clear_search_log()
  -> Agent Execution
  -> web_search(query)
  -> record_search(query, results)
  -> get_search_log()
  -> _append_to_sidebar(searches=...)
```

### 说明

- 当前项目已清理 cot/brief 模式，核心仅保留 agent 路径。
- 文档中的命令与代码路径均已按现状更新。
