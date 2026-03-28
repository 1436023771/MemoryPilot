from langchain_core.messages import AIMessage, ToolMessage

from app.agents.langgraph_flow import _finalize_node


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
