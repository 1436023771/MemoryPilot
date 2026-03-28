"""流式消息定义：用于在GUI和Agent之间传递中间状态。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    """流式消息类型枚举。"""
    
    PROGRESS = "progress"  # 处理进度通知，如"正在检索..."
    THINKING = "thinking"  # LLM思考过程（可选）
    TOOL_START = "tool_start"  # 工具开始执行
    TOOL_RESULT = "tool_result"  # 工具执行结果
    FINAL_ANSWER = "final_answer"  # 最终答案流式输出（token级）
    ERROR = "error"  # 错误消息


@dataclass
class StreamMessage:
    """流式消息对象，用于传递Agent执行的中间状态。"""
    
    message_type: MessageType  # 消息类型
    content: str  # 消息内容
    metadata: dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def __post_init__(self):
        """确保message_type是MessageType实例。"""
        if isinstance(self.message_type, str):
            self.message_type = MessageType(self.message_type)
    
    @classmethod
    def progress(cls, content: str, **metadata) -> "StreamMessage":
        """快捷方法：创建进度消息。"""
        return cls(MessageType.PROGRESS, content, metadata)
    
    @classmethod
    def thinking(cls, content: str, **metadata) -> "StreamMessage":
        """快捷方法：创建思考消息。"""
        return cls(MessageType.THINKING, content, metadata)
    
    @classmethod
    def tool_start(cls, tool_name: str, **metadata) -> "StreamMessage":
        """快捷方法：创建工具开始消息。"""
        return cls(MessageType.TOOL_START, tool_name, metadata)
    
    @classmethod
    def tool_result(cls, tool_name: str, result: str, **metadata) -> "StreamMessage":
        """快捷方法：创建工具结果消息。"""
        return cls(MessageType.TOOL_RESULT, result, {"tool_name": tool_name, **metadata})
    
    @classmethod
    def final_answer(cls, content: str, **metadata) -> "StreamMessage":
        """快捷方法：创建最终答案消息（可以是单token或片段）。"""
        return cls(MessageType.FINAL_ANSWER, content, metadata)
    
    @classmethod
    def error(cls, content: str, **metadata) -> "StreamMessage":
        """快捷方法：创建错误消息。"""
        return cls(MessageType.ERROR, content, metadata)
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典，方便序列化和调试。"""
        return {
            "message_type": self.message_type.value,
            "content": self.content,
            "metadata": self.metadata,
        }
