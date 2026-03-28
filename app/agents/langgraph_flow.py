from __future__ import annotations

from datetime import datetime
import os
from typing import Callable
from typing import Annotated
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

from app.agents.tools import retrieve_pg_knowledge, run_python_code, web_search
from app.agents.stream_messages import StreamMessage, MessageType
from app.core.config import Settings


def _add_stream_messages(left: list[StreamMessage], right: list[StreamMessage]) -> list[StreamMessage]:
    """合并流消息列表（用于Annotated字段）。"""
    return left + right


class QAState(TypedDict, total=False):
    question: str
    retrieved_context: str
    history: list[BaseMessage]
    messages: Annotated[list[BaseMessage], add_messages]
    stream_messages: Annotated[list[StreamMessage], _add_stream_messages]  # 新增：流式消息
    route: str
    knowledge_context: str
    final_prompt: str
    answer: str


def _node_config(name: str, extra_tags: list[str] | None = None) -> dict:
    tags = ["langgraph", "qa-flow", name]
    if extra_tags:
        tags.extend(extra_tags)
    return {
        "run_name": f"lg_{name}",
        "tags": tags,
        "metadata": {
            "orchestrator": "langgraph",
            "flow": "qa",
            "node": name,
        },
    }


def _graph_config() -> dict:
    project = os.getenv("LANGCHAIN_PROJECT", "agent-langgraph")
    return {
        "run_name": "langgraph_qa_chain",
        "tags": ["langgraph", "qa-flow"],
        "metadata": {
            "orchestrator": "langgraph",
            "flow": "qa",
            "project": project,
        },
    }


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
    # 仅在需要知识库路径时给出执行信息，减少噪音提示。
    messages = [StreamMessage.progress("将先检索知识库证据")]
    if route != "knowledge":
        messages = []
    return {"route": route, "stream_messages": messages}


def _knowledge_node(state: QAState) -> QAState:
    question = str(state.get("question", "")).strip()
    if not question:
        return {"knowledge_context": "", "stream_messages": []}

    messages = [StreamMessage.progress("正在检索知识库...")]
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
        messages.append(StreamMessage.progress("知识库检索完成"))
        return {"knowledge_context": str(retrieved), "stream_messages": messages}
    except Exception as exc:  # noqa: BLE001
        messages.append(StreamMessage.error(f"知识检索失败: {exc}"))
        return {"knowledge_context": f"知识检索失败: {exc}", "stream_messages": messages}


def _build_prompt_node(state: QAState) -> QAState:
    question = str(state.get("question", "")).strip()
    memory_ctx = str(state.get("retrieved_context", "")).strip() or "(none)"
    knowledge_ctx = str(state.get("knowledge_context", "")).strip() or "(none)"
    now_local = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    messages: list[StreamMessage] = []

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
    chat_messages: list[BaseMessage] = []
    if isinstance(history, list):
        chat_messages.extend(m for m in history if isinstance(m, BaseMessage))
    chat_messages.append(HumanMessage(content=prompt))

    return {"final_prompt": prompt, "messages": chat_messages, "stream_messages": messages}


def _tool_display_name(tool_name: str) -> str:
    mapping = {
        "run_python_code": "Python计算器",
        "web_search": "联网搜索",
        "retrieve_pg_knowledge": "知识库检索",
    }
    return mapping.get(tool_name, tool_name)


def _assistant_node_factory(model_with_tools):
    def _assistant_node(state: QAState) -> QAState:
        messages = state.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        stream_messages: list[StreamMessage] = []

        resp = model_with_tools.invoke(messages)
        tool_calls = getattr(resp, "tool_calls", None) or []
        if tool_calls:
            for call in tool_calls:
                tool_name = str(call.get("name", "")).strip() or "unknown_tool"
                args = call.get("args", {})
                display_name = _tool_display_name(tool_name)
                stream_messages.append(
                    StreamMessage.tool_start(
                        tool_name,
                        display_name=display_name,
                        args=args,
                    )
                )
        else:
            # 无工具调用时保留一条轻量状态，避免界面长时间静默。
            stream_messages.append(StreamMessage.progress("正在生成回答"))
        
        return {"messages": [resp], "stream_messages": stream_messages}

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
    stream_messages = []
    
    if not isinstance(messages, list) or not messages:
        stream_messages.append(StreamMessage.error("没有生成答案"))
        return {"answer": "", "stream_messages": stream_messages}

    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            answer = str(msg.content)
            break

    if not answer:
        answer = str(messages[-1])

    # 将最终答案分解为FINAL_ANSWER消息（可用于流式显示分块）
    # 这里在finalize阶段不分块，让GUI层处理分块显示
    stream_messages.append(StreamMessage.final_answer(answer))
    
    return {"answer": answer, "stream_messages": stream_messages}


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
    load_dotenv()

    model = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )

    model_with_tools = model.bind_tools([web_search, retrieve_pg_knowledge, run_python_code])

    graph = StateGraph(QAState)
    graph.add_node("plan", RunnableLambda(_plan_node).with_config(_node_config("plan")))
    graph.add_node(
        "retrieve_knowledge",
        RunnableLambda(_knowledge_node).with_config(_node_config("retrieve_knowledge", ["retrieval"])),
    )
    graph.add_node("build_prompt", RunnableLambda(_build_prompt_node).with_config(_node_config("build_prompt")))
    graph.add_node(
        "assistant",
        RunnableLambda(_assistant_node_factory(model_with_tools)).with_config(_node_config("assistant", ["llm"])),
    )
    graph.add_node(
        "tools",
        ToolNode([web_search, retrieve_pg_knowledge, run_python_code]).with_config(
            _node_config("tools", ["tool-calls"])
        ),
    )
    graph.add_node("finalize", RunnableLambda(_finalize_node).with_config(_node_config("finalize")))

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

    compiled_graph = graph.compile().with_config(_graph_config())

    base_chain = (
        RunnableLambda(
            lambda x: {
                "question": str(x.get("question", "")),
                "retrieved_context": str(x.get("retrieved_context", "")),
                "history": x.get("history", []),
            }
        ).with_config(_node_config("input_adapter"))
        | compiled_graph
        | RunnableLambda(_extract_answer).with_config(_node_config("output_adapter"))
    )

    runtime_chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    # 返回支持流式的包装
    return StreamingLanggraphChain(
        compiled_graph, 
        compiled_graph,
        runtime_chain,
        get_session_history,
        _node_config
    )


class StreamingLanggraphChain:
    """包装LangGraph chain，同时暴露invoke()和stream()接口。"""
    
    def __init__(self, compiled_graph, compiled_app, base_chain, session_history_getter, node_config_getter):
        self._graph = compiled_graph
        self._app = compiled_app
        self._base_chain = base_chain
        self._session_history_getter = session_history_getter
        self._node_config_getter = node_config_getter
    
    def invoke(self, input_data: dict, **kwargs) -> str:
        """同步调用，返回最终答案。代理给base_chain.invoke()。"""
        return self._base_chain.invoke(input_data, **kwargs)
    
    def stream(self, input_data: dict, session_id: str | None = None, **kwargs):
        """流式调用，逐步返回StreamMessage对象。
        
        Args:
            input_data: 输入数据（包含question, retrieved_context等）
            session_id: 会话ID，用于消息历史
            **kwargs: 传递给graph.stream()的其他参数
        
        Yields:
            StreamMessage对象，表示各个处理步骤的中间结果
        """
        # 从kwargs中提取session_id（如果提供的话从config中）
        if not session_id and "config" in kwargs:
            config = kwargs["config"]
            if isinstance(config, dict) and "configurable" in config:
                session_id = config["configurable"].get("session_id")
        
        # 准备输入状态
        state_input = {
            "question": str(input_data.get("question", "")),
            "retrieved_context": str(input_data.get("retrieved_context", "")),
            "history": input_data.get("history", []),
            "stream_messages": [],
        }
        
        # 流式执行graph，收集中间消息
        collected_messages = []
        final_state = None
        yielded_fingerprints: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
        seen_tool_message_ids: set[str] = set()
        tool_name_by_call_id: dict[str, str] = {}

        def _msg_fp(msg: StreamMessage) -> tuple[str, str, tuple[tuple[str, str], ...]]:
            meta_pairs = tuple(sorted((str(k), str(v)) for k, v in msg.metadata.items()))
            return (msg.message_type.value, msg.content, meta_pairs)

        for step_output in self._graph.stream(state_input, config=kwargs.get("config")):
            # step_output 是 {node_name: state_dict} 格式
            for node_name, state in step_output.items():
                final_state = state

                # 增量记录 tool_call_id -> tool_name，供 tools 结果展示。
                state_messages = state.get("messages", [])
                if isinstance(state_messages, list):
                    for item in state_messages:
                        if not isinstance(item, AIMessage):
                            continue
                        for call in (getattr(item, "tool_calls", None) or []):
                            call_id = str(call.get("id", "") or "").strip()
                            call_name = str(call.get("name", "") or "").strip()
                            if call_id and call_name:
                                tool_name_by_call_id[call_id] = call_name

                if "stream_messages" in state and state["stream_messages"]:
                    for msg in state["stream_messages"]:
                        fp = _msg_fp(msg)
                        if fp in yielded_fingerprints:
                            continue
                        yielded_fingerprints.add(fp)
                        collected_messages.append(msg)
                        yield msg

                # 从 tools 节点增量提取工具执行结果，展示真实执行信息。
                if node_name == "tools":
                    state_messages = state.get("messages", [])
                    if isinstance(state_messages, list):
                        for item in state_messages:
                            if not isinstance(item, ToolMessage):
                                continue
                            tool_call_id = str(getattr(item, "tool_call_id", "") or "")
                            content = str(getattr(item, "content", "") or "").strip()
                            if not tool_call_id or tool_call_id in seen_tool_message_ids:
                                continue
                            seen_tool_message_ids.add(tool_call_id)

                            preview = content[:180] + "..." if len(content) > 180 else content
                            resolved_tool_name = tool_name_by_call_id.get(tool_call_id, "tool")
                            tool_result_msg = StreamMessage.tool_result(
                                tool_name=resolved_tool_name,
                                result=preview or "(empty)",
                                tool_call_id=tool_call_id,
                            )
                            fp = _msg_fp(tool_result_msg)
                            if fp in yielded_fingerprints:
                                continue
                            yielded_fingerprints.add(fp)
                            collected_messages.append(tool_result_msg)
                            yield tool_result_msg
        
        # 确保至少返回一个最终答案消息
        if final_state and "answer" in final_state:
            answer = final_state.get("answer", "")
            if answer and (not collected_messages or collected_messages[-1].message_type != MessageType.FINAL_ANSWER):
                yield StreamMessage.final_answer(answer)
