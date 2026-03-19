from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.prompts import DEFAULT_QA_PROMPT


# 进程内会话仓库：key 是 session_id，value 是该会话的消息历史。
_SESSION_STORE: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """根据 session_id 获取（或创建）会话历史。"""
    # 首次访问某个 session_id 时，创建新的内存历史对象。
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]


def build_qa_chain(settings: Settings):
    """构建带短期记忆的问答链：Prompt -> 模型 -> 输出解析。"""
    # 初始化聊天模型，兼容 OpenAI 与 DeepSeek（OpenAI 兼容接口）。
    model = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )

    # 基础链只关注单轮输入输出。
    base_chain = DEFAULT_QA_PROMPT | model | StrOutputParser()

    # 用 RunnableWithMessageHistory 包装后，链会自动读写会话历史。
    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
