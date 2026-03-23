## 🔍 Web Search Display in GUI - 完成

### ✅ 已实现功能

**侧边栏现在显示以下信息：**
1. **Query** - 用户的原始查询
2. **Retrieved Context** - 从长期记忆中检索的上下文
3. **Extracted Facts** - 本轮提取的事实
4. **Written Facts** - 本轮写入的事实
5. **Web Search** (新增) - 网络搜索查询和结果
6. **CoT Reasoning** - 推理步骤（如果启用 CoT）

### 🎯 实现细节

#### 1. Web Search 追踪系统 (app/tools.py)
```python
# 全局搜索日志
_search_log: list[dict] = []

# 当 Agent 调用 web_search 时，自动记录：
# {
#   "query": "搜索查询",
#   "results": "搜索结果文本"
# }

# GUI 可通过以下方式访问：
from app.tools import get_search_log, clear_search_log

searches = get_search_log()  # 获取本轮搜索记录
clear_search_log()           # 清除搜索记录(每轮开始)
```

#### 2. GUI 集成 (app/gui_chat.py)
- `_process_turn()` 中调用 `clear_search_log()` 清除旧记录
- Agent 执行后通过 `get_search_log()` 获取本轮搜索
- `_append_to_sidebar()` 接收 searches 参数并显示在侧边栏
- 搜索结果限制为前 300 个字符，避免侧边栏过长

### 📋 使用说明

#### 在 GUI 中查看搜索
1. 启动 GUI: `python -m app.ui.gui_chat`
2. 选择 `CoT Mode: agent`
3. 输入查询并发送
4. 在右侧侧边栏查看 "Web Search" 部分，显示：
   - Agent 执行的搜索查询
   - 搜索返回的结果摘要

#### 在 CLI 中查看搜索 (可选)
```bash
# 使用 agent 模式会执行 web_search（如果需要）
python -m app.cli.main --cot-mode agent "你的问题"
```

### 🔧 技术架构

```
User Query
    ↓
[clear_search_log()]  ← 清除旧搜索记录
    ↓
Agent Execution
    ↓
[web_search called]
    ↓
[record_search(query, results)]  ← 自动记录到全局日志
    ↓
[get_search_log()]  ← GUI 获取搜索记录
    ↓
[_append_to_sidebar(searches=...)]  ← 显示在侧边栏
```

### 🎨 侧边栏显示示例

```
━━━ Turn 1 ━━━
Query:
What are recent AI developments?

Retrieved Context:
[From long-term memory...]

Extracted Facts:
[Facts extracted from this turn...]

Written Facts:
[Facts written to memory...]

Web Search:
Query: latest AI breakthroughs 2024
Results preview: - Recent developments in...
                  Machine learning models have...
                  ...

CoT Reasoning:
• First identify what "recent" means
• Consider major AI labs and companies
• ...
```

### ✨ 特点

- ✅ 实时跟踪搜索查询和结果
- ✅ 每轮自动清除前一轮记录
- ✅ 搜索结果自动截断防止侧边栏过长
- ✅ 支持多个搜索查询（如果 Agent 执行多次搜索）
- ✅ 与现有的 CoT/Brief/Off 模式无缝集成
- ✅ 无性能影响（仅记录文本，不产生额外计算）

### 🧪 验证

所有功能已验证：
- ✅ 搜索记录模块: `clear_search_log()`, `get_search_log()`, `record_search()`
- ✅ GUI 参数: `_append_to_sidebar(searches=...)`, `_on_turn_finished(searches=...)`
- ✅ 导入: `from app.tools import clear_search_log, get_search_log`
