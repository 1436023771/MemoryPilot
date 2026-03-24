# 搜索显示与 Agent 执行流程

## 目标

记录并展示每轮 Agent 的网络搜索行为，方便调试与可观测性分析。

## 关键模块

### 1. 搜索记录

文件：app/agents/tools.py

核心函数：
- clear_search_log：每轮开始清空历史记录
- record_search：web_search 执行后记录查询和结果
- get_search_log：供 GUI 拉取本轮搜索轨迹

### 2. GUI 侧边栏展示

文件：app/ui/gui_chat.py

核心流程：
1. _process_turn 开始时调用 clear_search_log
2. 调用 chain.invoke 执行 agent
3. 回答后调用 get_search_log
4. _append_to_sidebar 展示 Query / Context / Facts / Web Search

## 当前执行链路

```text
User Input
  -> _build_retrieved_context
  -> Agent chain.invoke
  -> web_search (optional)
  -> record_search
  -> _write_long_term_memory
  -> _append_to_sidebar
```

## 说明

- 项目当前为 agent-only 架构，不再包含 cot/brief 分支。
- GUI 中已无模式切换控件，所有对话统一走 agent 链路。
