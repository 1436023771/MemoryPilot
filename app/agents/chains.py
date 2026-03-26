from datetime import datetime

from langchain.agents import create_agent
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.agents.tools import retrieve_pg_knowledge, run_python_code, web_search


# 进程内会话仓库：key 是 session_id，value 是该会话的消息历史。
_SESSION_STORE: dict[str, InMemoryChatMessageHistory] = {}


def _format_agent_user_input(question: str, retrieved_context: str) -> str:
    """将问题和检索上下文组装成给 Agent 的用户输入。"""
    context = retrieved_context.strip() if retrieved_context else ""
    if not context:
        context = "(none)"

    now_local = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    return (
        "Current local time:\n"
        f"{now_local}\n\n"
        "You can use tool: web_search(query) for real-time and factual lookup.\n"
        "You can use tool: retrieve_pg_knowledge(query, top_k, book_id, chapter, context_window, rerank_candidates) for internal document knowledge retrieval.\n"
        "You can use tool: run_python_code(code) for precise calculation and logic verification.\n"
        "If the question asks current events, schedules, statistics, or facts you are not fully sure about, call web_search first.\n"
        "If the question is about project docs/domain knowledge, call retrieve_pg_knowledge first.\n"
        "If the question requires exact arithmetic, simulation, or multi-step logic validation, call run_python_code.\n"
        "Do NOT say you cannot access information before trying web_search at least once (unless user explicitly forbids search).\n\n"
        "User question:\n"
        f"{question}\n\n"
        "Retrieved memory context (use only if relevant):\n"
        f"{context}\n\n"
        "Please answer clearly in 3 bullet points or fewer."
    )


def _to_agent_messages(data: dict) -> list[BaseMessage]:
    """将 RunnableWithMessageHistory 注入的数据转换为 Agent 所需消息列表。"""
    messages: list[BaseMessage] = []
    history = data.get("history", [])

    if isinstance(history, list):
        for item in history:
            if isinstance(item, BaseMessage):
                messages.append(item)

    messages.append(
        HumanMessage(
            content=_format_agent_user_input(
                question=str(data.get("question", "")),
                retrieved_context=str(data.get("retrieved_context", "")),
            )
        )
    )
    return messages


def _extract_agent_answer(result: dict) -> str:
    """从 create_agent 的输出中提取最后一条助手消息。"""
    if not isinstance(result, dict):
        return str(result)

    messages = result.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return str(result)

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content", ""))

    return str(messages[-1])



def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """根据 session_id 获取（或创建）会话历史。"""
    # 首次访问某个 session_id 时，创建新的内存历史对象。
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]


def build_qa_chain(settings: Settings):
    """构建仅包含 Agent 模式的问答链。"""
    # 初始化聊天模型，兼容 OpenAI 与 DeepSeek（OpenAI 兼容接口）。
    model = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )

    # LangChain v1 内置 Agent（create_agent）模式，配备网络搜索能力。
    now_local = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    built_in_agent = create_agent(
        model=model,
        tools=[web_search, retrieve_pg_knowledge, run_python_code],
        system_prompt=(
            "You are a helpful assistant with access to tools. "
            f"Current local time is: {now_local}. "
            "Interpret relative time phrases (e.g., 'this year', 'recently') based on current local time. "
            "Available tool: web_search(query) for real-time information retrieval. "
            "Available tool: retrieve_pg_knowledge(query, top_k, book_id, chapter, context_window, rerank_candidates) for internal knowledge base retrieval from PostgreSQL+pgvector. "
            "Available tool: run_python_code(code) for exact calculations and logic checks. "
            "Tool usage policy: for current events, schedules, timelines, public-office activities, and any uncertain factual query, call web_search before answering. "
            "Tool usage policy: for questions about project documents, design details, or stored knowledge, call retrieve_pg_knowledge before answering. "
            "If the user mentions a specific book or chapter, pass precise book_id/chapter filters to retrieve_pg_knowledge to reduce noise. "
            "Tool usage policy: for math, algorithmic reasoning, or any answer requiring precise computation, call run_python_code before finalizing. "
            "Do not respond with 'I don't know' or 'my knowledge is limited' until you have attempted web_search at least once (unless user asks for no browsing). "
            "When needed, run multiple searches with refined keywords and summarize the best-supported findings. "
            "Then provide a concise final answer. "
            "Always use retrieved memory when available and relevant."
        ),
    )

    base_chain = (
        RunnableLambda(lambda x: {"messages": _to_agent_messages(x)})
        | built_in_agent
        | RunnableLambda(_extract_agent_answer)
    )

    # 用 RunnableWithMessageHistory 包装后，链会自动读写会话历史。
    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
