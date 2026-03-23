# 搜索信息显示 - 代码变更总结

## 📝 修改的文件

### 1. app/tools.py - 新增搜索追踪系统

```python
# 全局搜索记录（每轮保存 Agent 执行中的所有搜索）
_search_log: list[dict] = []

def get_search_log() -> list[dict]:
    """获取本轮搜索记录"""
    return _search_log.copy()

def clear_search_log() -> None:
    """清空搜索记录（每轮开始时调用）"""
    global _search_log
    _search_log = []

def record_search(query: str, results: str) -> None:
    """当 web_search 被调用时自动记录"""
    global _search_log
    _search_log.append({"query": query, "results": results})

@tool
def web_search(query: str) -> str:
    # ... 搜索逻辑 ...
    result_text = ...
    
    # 自动记录搜索（对 Agent 透明）
    record_search(query, result_text)
    return result_text
```

### 2. app/gui_chat.py - GUI 集成

#### 变更 1: 导入搜索函数
```python
from app.tools import clear_search_log, get_search_log
```

#### 变更 2: 修改 _process_turn() 
```python
def _process_turn(self, user_input: str) -> None:
    try:
        self.turn_count += 1
        clear_search_log()  # 清除上轮搜索记录 ← NEW
        
        retrieved_context = self._build_retrieved_context(user_input)
        response = self.chain.invoke(...)
        
        answer, cot_steps = self._parse_cot_response(str(response))
        searches = get_search_log()  # 获取本轮搜索 ← NEW
        
        extracted_facts, written_facts = self._write_long_term_memory(user_input)
        
        self.root.after(
            0,
            self._on_turn_finished,
            answer,
            extracted_facts,
            written_facts,
            None,
            user_input,
            retrieved_context,
            cot_steps,
            searches,  # 传递搜索信息 ← NEW
        )
```

#### 变更 3: 修改 _on_turn_finished()
```python
def _on_turn_finished(
    self,
    response: str,
    extracted_facts: list[str],
    written_facts: list[str],
    error: str | None,
    user_input: str = "",
    retrieved_context: str = "",
    cot_steps: list[str] | None = None,
    searches: list[dict] | None = None,  # ← NEW
) -> None:
    if cot_steps is None:
        cot_steps = []
    if searches is None:
        searches = []
    
    # 记录到侧边栏（包含搜索信息）
    self._append_to_sidebar(
        self.turn_count,
        user_input,
        retrieved_context,
        extracted_facts,
        written_facts,
        cot_steps=cot_steps,
        searches=searches,  # 传到侧边栏 ← NEW
    )
    # ... 其余逻辑 ...
```

#### 变更 4: 修改 _append_to_sidebar()
```python
def _append_to_sidebar(
    self,
    turn_num: int,
    question: str,
    context: str,
    extracted_facts: list[str],
    written_facts: list[str],
    cot_steps: list[str] | None = None,
    searches: list[dict] | None = None,  # ← NEW
) -> None:
    # ... 原有代码显示 Query, Context, Facts 等 ...
    
    # 网络搜索结果（新增部分）← NEW START
    if searches:
        self.sidebar_box.insert(tk.END, "Web Search:\n", "context_header")
        for search in searches:
            query = search.get("query", "")
            results = search.get("results", "")
            self.sidebar_box.insert(tk.END, f"Query: {query}\n", "user_query")
            # 限制搜索结果显示长度
            result_preview = results[:300] + "..." if len(results) > 300 else results
            self.sidebar_box.insert(tk.END, f"{result_preview}\n\n", "context")
    # ← NEW END
    
    # ... CoT 步骤显示 ...
```

## 🔄 执行流程

### 无搜索的查询流程
```
用户输入
  ↓
clear_search_log()    # _search_log = []
  ↓
Agent 执行（不需要搜索）
  ↓
get_search_log()      # 返回 []
  ↓
侧边栏显示（无搜索部分）
```

### 有搜索的查询流程
```
用户输入
  ↓
clear_search_log()    # _search_log = []
  ↓
Agent 执行
  ├─ Agent 决定需要搜索
  ├─ 调用 web_search("query")
  │  ├─ DuckDuckGo 搜索
  │  ├─ record_search("query", results)  ← 自动记录
  │  └─ 返回结果
  ├─ Agent 处理搜索结果
  └─ Agent 生成最终答案
  ↓
get_search_log()      # 返回 [{"query": "...", "results": "..."}, ...]
  ↓
侧边栏显示包含 Web Search 部分
```

## 📊 数据流

```
web_search() 被调用
    ↓
DuckDuckGo API 返回结果
    ↓
record_search(query, result_text)
    ↓
_search_log.append({"query": ..., "results": ...})
    ↓
Agent 继续执行
    ↓
_process_turn() 完成
    ↓
searches = get_search_log()  ← 检索所有记录
    ↓
_on_turn_finished(..., searches=searches)
    ↓
_append_to_sidebar(..., searches=searches)
    ↓
GUI 侧边栏显示搜索信息
```

## ✅ 验证清单

- [x] 搜索记录模块实现
- [x] web_search 自动调用 record_search
- [x] GUI 导入搜索函数
- [x] _process_turn 清除和获取搜索日志
- [x] _on_turn_finished 接收搜索参数
- [x] _append_to_sidebar 显示搜索信息
- [x] 导入语句检查通过
- [x] 函数签名验证通过
- [x] 搜索记录 API 验证通过
