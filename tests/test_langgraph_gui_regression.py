from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.langgraph_flow import StreamingLanggraphChain, _finalize_node
from app.agents.langgraph_flow import _compress_history_by_token_budget
from app.agents.langgraph_flow import _compress_text_to_token_budget
from app.agents.langgraph_flow import _estimate_message_tokens
from app.agents.langgraph_flow import _estimate_text_tokens


def test_finalize_returns_answer_and_stream_messages() -> None:
    """测试_finalize_node返回answer和stream_messages。"""
    pre_tool = AIMessage(
        content="我先查一下最新信息。",
        tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "call_1"}],
    )
    tool_result = ToolMessage(content="搜索结果: A, B, C", tool_call_id="call_1")

    result = _finalize_node({"messages": [pre_tool, tool_result]})
    answer = str(result.get("answer", ""))

    # 检查answer内容
    assert "我先查一下最新信息" in answer
    
    # 检查stream_messages是否存在
    assert "stream_messages" in result
    assert len(result["stream_messages"]) > 0
    
    # 检查最后一个消息是FINAL_ANSWER类型
    last_msg = result["stream_messages"][-1]
    assert last_msg.message_type.value == "final_answer"


def test_finalize_handles_empty_messages() -> None:
    """测试_finalize_node处理空消息列表。"""
    result = _finalize_node({"messages": []})
    
    assert result.get("answer") == ""
    assert "stream_messages" in result
    # 应该有一个ERROR消息
    assert any(msg.message_type.value == "error" for msg in result["stream_messages"])


def test_stream_mode_persists_session_history() -> None:
    """测试langgraph stream模式会写入短期会话历史。"""

    class _FakeGraph:
        def stream(self, *_args, **_kwargs):
            yield (
                "updates",
                {
                    "finalize": {
                        "answer": "这是最终答案",
                        "stream_messages": [],
                        "messages": [AIMessage(content="这是最终答案")],
                    }
                },
            )

    store: dict[str, InMemoryChatMessageHistory] = {}

    def _get_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chain = StreamingLanggraphChain(_FakeGraph(), _FakeGraph(), None, _get_history, lambda _n: {})

    _ = list(
        chain.stream(
            {"question": "请记住这句话", "retrieved_context": ""},
            session_id="test-session",
            config={"configurable": {"session_id": "test-session"}},
        )
    )

    history = _get_history("test-session")
    assert len(history.messages) == 2
    assert isinstance(history.messages[0], HumanMessage)
    assert isinstance(history.messages[1], AIMessage)
    assert "请记住这句话" in str(history.messages[0].content)
    assert "这是最终答案" in str(history.messages[1].content)


def test_compress_history_by_token_budget_keeps_all_turns() -> None:
    """测试历史压缩保留所有轮次消息结构。"""
    history = [
        HumanMessage(content="这是第一轮问题，比较长比较长比较长比较长"),
        AIMessage(content="这是第一轮回答，比较长比较长比较长比较长"),
        HumanMessage(content="第二轮问题"),
        AIMessage(content="第二轮回答"),
    ]

    compressed = _compress_history_by_token_budget(history, max_tokens=12)

    assert len(compressed) == len(history)
    assert all(isinstance(m, (HumanMessage, AIMessage)) for m in compressed)
    before_total = sum(_estimate_message_tokens(m) for m in history)
    after_total = sum(_estimate_message_tokens(m) for m in compressed)
    assert after_total <= 12
    assert after_total <= before_total


def test_compress_history_by_token_budget_prioritizes_recent_content() -> None:
    """测试预算紧张时最近消息保留信息更多。"""
    history = [
        HumanMessage(content="第一轮问题：" + "A" * 120),
        AIMessage(content="第一轮回答：" + "B" * 120),
        HumanMessage(content="第二轮问题：" + "C" * 120),
        AIMessage(content="第二轮回答：" + "D" * 120),
    ]

    compressed = _compress_history_by_token_budget(history, max_tokens=30)

    first_len = len(str(compressed[0].content))
    last_len = len(str(compressed[-1].content))
    assert last_len >= first_len


def test_compress_history_by_token_budget_no_change_when_budget_large() -> None:
    """测试预算充足时不压缩。"""
    history = [
        HumanMessage(content="hello"),
        AIMessage(content="world"),
    ]

    compressed = _compress_history_by_token_budget(history, max_tokens=1000)
    assert len(compressed) == 2
    assert str(compressed[0].content) == "hello"
    assert str(compressed[1].content) == "world"


def test_compress_text_preserves_key_tokens_when_possible() -> None:
    """测试本地抽取压缩尽量保留日期/数字/否定等关键token。"""
    text = (
        "会议时间是2026-03-28，预算上限12000元，"
        "注意不要改成15000元。"
        "这里是一些可压缩的背景说明，用于制造冗余文本。" * 4
    )

    compressed = _compress_text_to_token_budget(text, max_tokens=42)

    assert _estimate_text_tokens(compressed) <= 42
    assert "2026-03-28" in compressed
    assert "12000" in compressed
    assert ("不要" in compressed) or ("不" in compressed)
