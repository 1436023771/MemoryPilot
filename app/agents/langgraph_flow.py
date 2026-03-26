from __future__ import annotations

from datetime import datetime
from typing import Callable
from typing import Annotated
from typing import TypedDict

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

from app.agents.tools import retrieve_pg_knowledge, run_python_code, web_search
from app.core.config import Settings


class QAState(TypedDict, total=False):
    question: str
    retrieved_context: str
    history: list[BaseMessage]
    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    knowledge_context: str
    final_prompt: str
    answer: str


def _detect_route(question: str) -> str:
    q = (question or "").lower()
    knowledge_keys = [
        "书",
        "章节",
        "chapter",
        "剧情",
        "角色",
        "设定",
        "文档",
        "项目",
        "book",
        "timeline",
        "time line",
    ]
    if any(k in q for k in knowledge_keys):
        return "knowledge"
    return "direct"


def _plan_node(state: QAState) -> QAState:
    question = str(state.get("question", "")).strip()
    route = _detect_route(question)
    return {"route": route}


def _knowledge_node(state: QAState) -> QAState:
    question = str(state.get("question", "")).strip()
    if not question:
        return {"knowledge_context": ""}

    try:
        # 先做召回+重排，再把结果回传给回答节点。
        retrieved = retrieve_pg_knowledge.invoke(
            {
                "query": question,
                "top_k": 5,
                "context_window": 3,
                "rerank_candidates": 14,
            }
        )
        return {"knowledge_context": str(retrieved)}
    except Exception as exc:  # noqa: BLE001
        return {"knowledge_context": f"知识检索失败: {exc}"}


def _build_prompt_node(state: QAState) -> QAState:
    question = str(state.get("question", "")).strip()
    memory_ctx = str(state.get("retrieved_context", "")).strip() or "(none)"
    knowledge_ctx = str(state.get("knowledge_context", "")).strip() or "(none)"
    now_local = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    prompt = (
        "You are a concise and reliable reading companion assistant.\n"
        f"Current local time: {now_local}\n\n"
        "Task policy:\n"
        "1) Prefer internal knowledge context when answering book/project questions.\n"
        "2) If timeline is involved, list events in chronological order and mention uncertainty explicitly.\n"
        "3) If evidence conflicts, present both versions and cite the chunk references from retrieval context.\n"
        "4) Keep final answer within 3 bullet points.\n\n"
        "User question:\n"
        f"{question}\n\n"
        "Retrieved memory context:\n"
        f"{memory_ctx}\n\n"
        "Retrieved knowledge context:\n"
        f"{knowledge_ctx}\n"
    )

    history = state.get("history", [])
    messages: list[BaseMessage] = []
    if isinstance(history, list):
        messages.extend(m for m in history if isinstance(m, BaseMessage))
    messages.append(HumanMessage(content=prompt))

    return {"final_prompt": prompt, "messages": messages}


def _assistant_node_factory(model_with_tools):
    def _assistant_node(state: QAState) -> QAState:
        messages = state.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        resp = model_with_tools.invoke(messages)
        return {"messages": [resp]}

    return _assistant_node


def _should_call_tools(state: QAState) -> str:
    messages = state.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return "finalize"

    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return "tools"
    return "finalize"


def _finalize_node(state: QAState) -> QAState:
    messages = state.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return {"answer": ""}

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return {"answer": str(msg.content)}

    return {"answer": str(messages[-1])}


def _route_from_plan(state: QAState) -> str:
    return str(state.get("route", "direct"))


def _extract_answer(result: dict) -> str:
    if not isinstance(result, dict):
        return str(result)
    return str(result.get("answer", ""))


def build_langgraph_chain(
    settings: Settings,
    get_session_history: Callable[[str], BaseChatMessageHistory],
):
    """Build a LangGraph-based QA chain with explicit routing and retrieval nodes."""
    model = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )

    model_with_tools = model.bind_tools([web_search, retrieve_pg_knowledge, run_python_code])

    graph = StateGraph(QAState)
    graph.add_node("plan", _plan_node)
    graph.add_node("retrieve_knowledge", _knowledge_node)
    graph.add_node("build_prompt", _build_prompt_node)
    graph.add_node("assistant", _assistant_node_factory(model_with_tools))
    graph.add_node("tools", ToolNode([web_search, retrieve_pg_knowledge, run_python_code]))
    graph.add_node("finalize", _finalize_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        _route_from_plan,
        {
            "knowledge": "retrieve_knowledge",
            "direct": "build_prompt",
        },
    )
    graph.add_edge("retrieve_knowledge", "build_prompt")
    graph.add_edge("build_prompt", "assistant")
    graph.add_conditional_edges(
        "assistant",
        _should_call_tools,
        {
            "tools": "tools",
            "finalize": "finalize",
        },
    )
    graph.add_edge("tools", "assistant")
    graph.add_edge("finalize", END)

    app = graph.compile()

    base_chain = RunnableLambda(
        lambda x: {
            "question": str(x.get("question", "")),
            "retrieved_context": str(x.get("retrieved_context", "")),
            "history": x.get("history", []),
        }
    ) | app | RunnableLambda(_extract_answer)

    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
