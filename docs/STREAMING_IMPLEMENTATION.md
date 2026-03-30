# GUI 流式传输改写说明

## 概述

对GUI应用进行了完整的流式传输改造，使用户能在更短的时间内看到中间处理进度和逐步的答案，而不是等待整个流程完成后才显示结果。

## 主要改动

### 1. **消息结构定义** (`app/agents/stream_messages.py`)
- 新增 `MessageType` 枚举定义消息类型：
  - `PROGRESS`: 处理进度通知（如"正在检索..."）
  - `THINKING`: LLM思考过程
  - `TOOL_START`: 工具开始执行
  - `TOOL_RESULT`: 工具打印结果
  - `FINAL_ANSWER`: 最终答案片段
  - `ERROR`: 错误消息

- 新增 `StreamMessage` 数据类封装流式消息，提供快捷方法创建各类消息

### 2. **LangGraph流式改造** (`app/agents/langgraph_flow.py`)
- 在 `QAState` 中添加 `stream_messages` 字段用于存储中间消息
- 改造各节点以生成进度消息：
  - `_plan_node`: 发出路由决策消息
  - `_knowledge_node`: 发出检索进度（"正在检索知识库..."、 "检索完成"）
  - `_build_prompt_node`: 发出提示词准备进度
  - `_assistant_node`: 发出思考进度
  - `_finalize_node`: 发出最终答案消息

- 新增 `StreamingLanggraphChain` 包装类：
  - 暴露 `invoke()` 方法（同步调用，向后兼容）
  - 暴露 `stream()` 方法（流式调用，返回 `StreamMessage` 迭代器）
  - `stream()` 方法通过 `graph.stream()` 获取中间状态

### 3. **后端适配层** (`app/agents/chains.py`)
- `build_qa_chain()` 统一返回 LangGraph 流式链对象（LangGraph-only）

### 4. **GUI改造** (`app/ui/gui_chat.py`)
- 添加 `message_queue`（线程安全队列）接收后台线程的流式消息
- 改造 `_process_turn()` 方法：
  - 使用 `chain.stream()` 而不是 `chain.invoke()`
  - 后台线程迭代消息并put到队列
  - 处理完成后put None 信号

- 新增 `_poll_message_queue()` 方法：
  - 主线程定期轮询队列（每100ms）
  - 非阻塞读取，避免UI冻结
  - 处理各类消息类型，实时更新UI

- 新增 `_handle_stream_message()` 方法：
  - 按消息类型分发处理
  - PROGRESS: 显示"[进度]信息"
  - TOOL_START/TOOL_RESULT: 显示工具调用历史
  - FINAL_ANSWER: 追加到答案区
  - ERROR: 显示错误信息

- 改造 `_on_send()` 方法：
  - 启动后台线程处理，同时启动轮询函数
  - 不再阻塞UI

### 5. **测试** (`tests/test_streaming.py`)
- 13个单元测试覆盖：
  - `StreamMessage` 创建、转换、快捷方法
  - `StreamingLanggraphChain` 的 invoke/stream 接口
  - LangGraph-only chain 的 stream/invoke 接口
  - 消息类型定义和值

## 使用效果

### 改造前
```
用户输入 → 等待5-10秒（整个流程同步执行）→ 显示完整答案
```

### 改造后
```
用户输入 → 立即启动后台线程和轮询 
  ↓
1-2秒内显示"[进度] 路由: 知识库"
  ↓
2-3秒显示"[进度] 正在检索知识库..."
  ↓
3-4秒显示检索结果摘要
  ↓
4-5秒显示"[进度] 正在思考..."
  ↓
5-6秒开始逐步显示答案
  ↓
最后显示完整答案和侧边栏统计
```

## 向后兼容性

- 所有链仍然支持 `invoke()` 方法（同步调用），保持向后兼容
- 现有代码无需改动，可直接使用新的流式功能
- LangGraph-only 模式支持流式

## 后续改进方向

1. **Token级流式输出**：当前 LLM 答案是整体返回。可升级为使用 LLM 的 `.stream()` API 获得逐token输出

2. **工具输出流式**：可在工具执行进度中发出更细粒度的消息（如"搜索中... 已找到5条结果"）

3. **取消机制**：可添加"停止查询"按钮，通过 Event 或超时中断 stream()

4. **UI优化**：
   - 进度条展示（而非文本）
   - 分类显示进度（知识库/网搜/LLM思考）
   - 答案流式显示优化（分段而非逐行）

## 测试方式

### 运行所有流式测试
```bash
pytest tests/test_streaming.py -v
```

### 运行GUI回归测试
```bash
pytest tests/test_langgraph_gui_regression.py -v
```

### 手动测试GUI
```bash
python -m app.ui.gui_chat
```

输入一个查询（如"《恋人不行》中关于记忆的设定是什么？"），观察：
- 进度消息逐步显示
- 答案区逐步填充
- 侧边栏更新检索结果和执行日志

## 关键配置

- 消息队列大小：`maxsize=10`（防止积压）
- 轮询间隔：`100ms`（平衡响应性和CPU使用）
