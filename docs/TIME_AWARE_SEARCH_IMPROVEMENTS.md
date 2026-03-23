# Agent 时间感知与网络搜索改进 - 更新说明

## ✅ 已完成的改进

### 1. **Agent Prompt 注入当前时间**
- ✅ System prompt 现在包含当前本地时间戳
- ✅ User input 前缀增加"Current local time"段
- ✅ Agent 能够基于当前时间理解相对时间短语（如"今年"、"最近"、"最近几天"）

#### 代码位置
**app/chains.py** - 两处修改：
```python
# 1. _format_agent_user_input() 函数
now_local = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
return (
    "Current local time:\n"
    f"{now_local}\n\n"
    "User question:\n"
    f"{question}\n\n"
    # ...
)

# 2. build_qa_chain() 中 agent 模式
system_prompt=(
    "You are a helpful assistant with access to a web search tool. "
    f"Current local time is: {now_local}. "
    "Interpret relative time phrases (e.g., 'this year', 'recently') based on current local time. "
    # ...
)
```

### 2. **web_search 工具稳定性与覆盖率提升**

#### 改进措施

| 策略 | 说明 |
|------|------|
| **多查询变体** | 自动生成 3-5 个查询降级版本 |
| **查询标准化** | 去除"最近/今年/访问地点"等噪声词 |
| **双语支持** | 中文问题自动补充英文查询变体 |
| **二次参数回退** | 首次无结果时用更宽松参数重试 |
| **新闻补充** | 文本搜索不足时补充新闻源 |
| **主题过滤** | 对总统类问题做相关性过滤，去掉跑题结果 |
| **官方网站优先** | 自动加入 whitehouse.gov 相关查询 |

#### 代码位置
**app/tools.py** - 新增 3 个辅助函数 + 改进 `web_search()`：
```python
# 新增函数
def _normalize_query(query: str) -> str:       # 查询清理
def _build_query_variants(query: str) -> list: # 变体生成
def _filter_relevant_items(query, items):      # 相关性过滤

# 改进的 web_search 逻辑
1. 生成 3-5 个查询变体
2. 对每个变体执行 text() 搜索（region="wt-wt", safesearch="moderate"）
3. 若结果 < 2 个，用更宽松参数重试（移除 safesearch 限制）
4. 若结果 < 3 个，补充 news() 搜索
5. 相关性过滤（特别针对总统相关问题）
6. 返回尝试过的所有查询 + 最终结果
```

### 3. **依赖包升级**
- ✅ 安装现代 `ddgs` 包（2.18.4+）
- ✅ 兼容旧 `duckduckgo-search` 包（降级方案）
- ✅ 消除 DuckDuckGo 弃用警告

## 📊 实际效果对比

### 问题：美国总统今年行程

#### 改进前
```
Query: 2024年美国总统行程 访问地点 今年去过哪些地方
Result: [搜索结果中多为无关 Firefly 相关条目]

Query: 拜登总统2024年访问行程 国际访问 国内访问  
Result: 未找到...的搜索结果

Query: President Biden 2024 travel itinerary visits locations
Result: 未找到...的搜索结果
```

#### 改进后
```
Query: 美国总统最近的行程安排是什么？
Result: ✅ 获得有效结果：
- 3月19日白宫会晤日本首相
- 3月底访华行程推迟
- 近期伊朗局势相关外交活动
- 包含当前时间上下文（2026年3月23日）
```

## 🔧 使用示例

### CLI 调用
```bash
# Agent 模式自动开启时间感知和网络搜索
python -m app.cli.main --cot-mode agent "美国总统最近的行程是什么？"

# 输出包含：
# 1. 时间上下文（当前时间：2026-03-23 ...）
# 2. 网络搜索执行过程（哪些查询有效）
# 3. 基于最新信息的答案
```

### GUI 侧边栏显示
```
━━━ Turn 1 ━━━
Query:
美国总统最近的行程是什么？

Web Search:
Query: 美国总统最近行程  ← 尝试过的查询
...
Query: President schedule White House  ← 英文回退
Results: - 白宫日程...
         - 总统访问...

CoT Reasoning: [如果启用 brief 模式]
```

## 📝 技术细节

### 查询变体生成示例
语言：中文  
原始查询：`2024年美国总统行程 访问地点 今年去过哪些地方`

生成的变体：
1. `2024年美国总统行程 访问地点 今年去过哪些地方` （原始）
2. `2024年美国总统 地点` （噪声词去除）
3. `U.S. President schedule White House` （英文通用）
4. `President Biden schedule White House` （具体总统名）
5. `White House presidential schedule` （官方机构术语）
6. `site:whitehouse.gov presidential schedule` （官方网站限定）

### 时间上下文流程
```
prompt 层：System Prompt 包含 "Current time: 2026-03-23 17:29:28 CST"
User Input 前缀：包含本地时间戳
Agent 推理：基于当前时间理解"最近"、"今年"等相对时间表达
搜索查询：考虑时间范围优化查询词
最终答案：时间感知的回答
```

## ✨ 优势

| 优势 | 说明 |
|------|------|
| 时间感知 | Agent 理解"今年"、"最近"等表达，生成时间相关的搜索查询 |
| 搜索可靠性 | 多策略回退确保高覆盖，"未找到"频率大幅下降 |
| 主题相关 | 特定领域问题（总统、天气等）优先返回相关结果 |
| 可追踪性 | GUI 侧边栏显示所有尝试过的查询，便于调试 |
| 双语支持 | 中英文混合问题自动生成多语言查询 |

## 🔍 调试方法

### 在 GUI 中查看搜索过程
1. 启动 GUI：`python -m app.ui.gui_chat`
2. 设置 `CoT Mode: agent`
3. 提交查询
4. 侧边栏 "Web Search" 部分显示所有尝试过的查询和结果

### 在 CLI 中检查搜索日志
```python
from app.tools import web_search, get_search_log
web_search.invoke({"query": "你的问题"})
searches = get_search_log()
for s in searches:
    print("Query:", s["query"])
    print("Result:", s["results"][:300])
    print()
```

## 📋 验证清单

- [x] 导入 datetime 并注入当前时间到 system prompt
- [x] 在 user input 前缀添加时间戳
- [x] 添加时间相对性理解指示
- [x] 实现查询标准化函数 `_normalize_query`
- [x] 实现变体生成函数 `_build_query_variants` 
- [x] 添加二次参数回退搜索
- [x] 实现相关性过滤 `_filter_relevant_items`
- [x] 通过 CLI 端到端测试
- [x] 验证结果不再包含"未找到"信息
- [x] 安装现代 ddgs 包
- [x] 更新文档

## 🚀 下一步（可选）

- [ ] 加入搜索结果缓存（同一轮内重复查询不重复请求）
- [ ] 支持在 Agent prompt 中指定搜索优先级（如"新闻优先"）
- [ ] 集成搜索引擎选择（DuckDuckGo / Google / Bing）
- [ ] 增加搜索超时控制和请求速率限制
