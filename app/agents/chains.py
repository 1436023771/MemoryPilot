from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from app.core.config import Settings


# 进程内会话仓库：key 是 session_id，value 是该会话的消息历史。
_SESSION_STORE: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """根据 session_id 获取（或创建）会话历史。"""
    # 首次访问某个 session_id 时，创建新的内存历史对象。
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]


def build_qa_chain(settings: Settings):
    """构建问答链（LangGraph-only）。"""
    from app.agents.langgraph_flow import build_langgraph_chain

    return build_langgraph_chain(settings=settings, get_session_history=get_session_history)
