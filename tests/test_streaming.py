"""测试流式消息和流式chain功能。"""

import pytest
from app.agents.stream_messages import StreamMessage, MessageType


class TestStreamMessage:
    """测试StreamMessage类。"""
    
    def test_stream_message_creation(self):
        """测试基本的StreamMessage创建。"""
        msg = StreamMessage(MessageType.PROGRESS, "正在处理...")
        assert msg.message_type == MessageType.PROGRESS
        assert msg.content == "正在处理..."
        assert msg.metadata == {}
    
    def test_stream_message_with_metadata(self):
        """测试带元数据的StreamMessage。"""
        msg = StreamMessage(MessageType.TOOL_RESULT, "result", {"tool_name": "search"})
        assert msg.content == "result"
        assert msg.metadata["tool_name"] == "search"
    
    def test_stream_message_progress_shortcut(self):
        """测试PROGRESS消息的快捷方法。"""
        msg = StreamMessage.progress("进度提示")
        assert msg.message_type == MessageType.PROGRESS
        assert msg.content == "进度提示"
    
    def test_stream_message_error_shortcut(self):
        """测试ERROR消息的快捷方法。"""
        msg = StreamMessage.error("出错了")
        assert msg.message_type == MessageType.ERROR
        assert msg.content == "出错了"
    
    def test_stream_message_final_answer_shortcut(self):
        """测试FINAL_ANSWER消息的快捷方法。"""
        msg = StreamMessage.final_answer("答案是...")
        assert msg.message_type == MessageType.FINAL_ANSWER
        assert msg.content == "答案是..."
    
    def test_stream_message_tool_shortcut(self):
        """测试TOOL_RESULT消息的快捷方法。"""
        msg = StreamMessage.tool_result("web_search", "search results")
        assert msg.message_type == MessageType.TOOL_RESULT
        assert msg.content == "search results"
        assert msg.metadata["tool_name"] == "web_search"
    
    def test_stream_message_to_dict(self):
        """测试StreamMessage转换为字典。"""
        msg = StreamMessage.progress("进度", level=1)
        d = msg.to_dict()
        assert d["message_type"] == "progress"
        assert d["content"] == "进度"
        assert d["metadata"]["level"] == 1
    
    def test_stream_message_type_string_conversion(self):
        """测试字符串类型的MessageType转换。"""
        msg = StreamMessage("progress", "test")
        assert msg.message_type == MessageType.PROGRESS


class TestStreamingChain:
    """测试流式chain的功能（需要实际的chain实例）。"""
    
    def test_streaming_chain_has_stream_method(self):
        """测试配置为langgraph时chain有stream()方法。"""
        from app.agents.chains import build_qa_chain
        from app.core.config import get_settings
        
        # 构建langgraph chain
        settings = get_settings()
        chain = build_qa_chain(settings)
        
        # 检查是否有stream方法
        assert hasattr(chain, 'stream'), "chain should have stream() method"
        assert callable(getattr(chain, 'stream')), "stream should be callable"
    
    def test_streaming_chain_has_invoke_method(self):
        """测试chain有invoke()方法（向后兼容）。"""
        from app.agents.chains import build_qa_chain
        from app.core.config import get_settings
        
        settings = get_settings()
        chain = build_qa_chain(settings)
        
        # 检查是否有invoke方法
        assert hasattr(chain, 'invoke'), "chain should have invoke() method"
        assert callable(getattr(chain, 'invoke')), "invoke should be callable"


class TestLanggraphOnlyChain:
    """测试LangGraph-only链行为。"""

    def test_langgraph_only_chain_has_stream_method(self):
        """LangGraph-only模式下chain应提供stream方法。"""
        from app.agents.chains import build_qa_chain
        from app.core.config import get_settings

        settings = get_settings()
        chain = build_qa_chain(settings)

        assert hasattr(chain, 'stream'), "chain should have stream() method"
        assert callable(getattr(chain, 'stream')), "stream should be callable"


@pytest.mark.asyncio
class TestStreamMessageTypes:
    """测试各种消息类型的有效性。"""
    
    def test_all_message_types_defined(self):
        """测试所有MessageType都有定义。"""
        expected_types = ["PROGRESS", "THINKING", "TOOL_START", "TOOL_RESULT", "FINAL_ANSWER", "ERROR"]
        for msg_type in expected_types:
            assert hasattr(MessageType, msg_type), f"MessageType.{msg_type} should be defined"
    
    def test_message_type_values(self):
        """测试MessageType值的正确性。"""
        assert MessageType.PROGRESS.value == "progress"
        assert MessageType.THINKING.value == "thinking"
        assert MessageType.TOOL_START.value == "tool_start"
        assert MessageType.TOOL_RESULT.value == "tool_result"
        assert MessageType.FINAL_ANSWER.value == "final_answer"
        assert MessageType.ERROR.value == "error"
