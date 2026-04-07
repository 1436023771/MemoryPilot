from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Callable

from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage

from app.agents.langgraph.history_compression import _compress_history_by_token_budget, _compress_text_to_token_budget, _history_token_limit
from app.agents.langgraph.routing_policy import _detect_route
from app.agents.langgraph.state import QAState
from app.agents.llm_client import invoke_model
from app.agents.skills import (
    DEFAULT_SKILL_NAME,
    process_context_for_skill,
    resolve_prompt_key,
    resolve_skill_name,
    resolve_tool_display_name,
    select_tools_for_skill,
    skill_registry,
)
from app.agents.stream_messages import StreamMessage
from app.agents.tool_definition import parse_tool_arguments
from app.agents.tool_registry import tool_registry
from app.config import get_settings
from app.config.langgraph import context_window_default, rerank_candidates_default, top_k_default
from app.core.prompt_store import render_prompt


logger = logging.getLogger(__name__)


def _extract_selected_skill_from_response(content: str) -> str | None:
    """Extract selected_skill from model output with robust JSON parsing.

    Strategy:
    1) Try strict JSON decode on full text.
    2) If that fails, scan for JSON objects and decode each candidate with raw_decode.
    3) Return the first non-empty selected_skill string found.
    """
    text = str(content or "").strip()
    if not text:
        return None

    decoder = json.JSONDecoder()

    # Preferred path: model returned strict JSON only.
    try:
        parsed = decoder.decode(text)
        if isinstance(parsed, dict):
            value = parsed.get("selected_skill")
            if isinstance(value, str) and value.strip():
                return value.strip()
    except json.JSONDecodeError:
        pass

    # Fallback path: model wrapped JSON with extra commentary.
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue

        if not isinstance(parsed, dict):
            continue

        value = parsed.get("selected_skill")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _safe_invoke_tool(tool_name: str, raw_args: object) -> str:
    """Invoke tool with explicit parse/execute error handling."""
    tool = tool_registry.get_tool(tool_name)
    if tool is None:
        return f"工具执行失败: 未找到工具 {tool_name}"

    try:
        args = parse_tool_arguments(raw_args)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "Tool argument parse failed; tool=%s raw_args_type=%s err=%s",
            tool_name,
            type(raw_args).__name__,
            type(exc).__name__,
            exc_info=True,
        )
        return f"工具执行失败: 参数解析失败 ({type(exc).__name__}: {exc})"

    try:
        return str(tool.invoke(args))
    except (RuntimeError, ValueError, TypeError, OSError, TimeoutError) as exc:
        logger.warning(
            "Tool invoke recoverable error; tool=%s err=%s",
            tool_name,
            type(exc).__name__,
            exc_info=True,
        )
        return f"工具执行失败: {type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Tool invoke unexpected error; tool=%s", tool_name)
        return f"工具执行失败: {type(exc).__name__}: {exc}"


def _plan_node(state: QAState) -> QAState:
    question = str(state.get("question", "")).strip()
    route = _detect_route(question)
    # 仅在需要知识库路径时给出执行信息，减少噪音提示。
    messages = [StreamMessage.progress("将先检索知识库证据")]
    if route != "knowledge":
        messages = []
    return {"route": route, "stream_messages": messages}


def _skill_planning_node(state: QAState) -> QAState:
    """LLM-based skill selection node.

    Let LLM choose the most appropriate skill for the user's question.
    Falls back to 'general' if LLM cannot decide.
    """
    question = str(state.get("question", "")).strip()
    if not question:
        return {"selected_skill": DEFAULT_SKILL_NAME}

    skills_list = skill_registry.list_descriptions()
    if not skills_list:
        return {"selected_skill": DEFAULT_SKILL_NAME}

    decision_prompt = render_prompt(
        "agents.langgraph.skill_planning_prompt",
        skills_list=skills_list,
        question=question,
    )

    try:
        settings = get_settings()

        response = invoke_model(
            settings=settings,
            prompt_or_messages=decision_prompt,
            temperature=0.3,
        )
        content = str(getattr(response, "content", "") or "").strip()

        selected_skill = _extract_selected_skill_from_response(content)
        if not selected_skill:
            logger.warning(
                "Skill planning JSON parse failed, fallback to default; content_len=%s",
                len(content),
            )
            selected_skill = DEFAULT_SKILL_NAME

        return {"selected_skill": resolve_skill_name(selected_skill)}

    except (RuntimeError, ValueError, TypeError, KeyError) as exc:
        logger.warning(
            "Skill planning recoverable error, fallback to default; err=%s",
            type(exc).__name__,
            exc_info=True,
        )
        return {"selected_skill": DEFAULT_SKILL_NAME}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Skill planning unexpected error, fallback to default")
        return {"selected_skill": DEFAULT_SKILL_NAME}


def _knowledge_node(state: QAState) -> QAState:
    question = str(state.get("question", "")).strip()
    if not question:
        return {"knowledge_context": "", "stream_messages": []}

    messages = [StreamMessage.progress("正在检索知识库...")]
    try:
        knowledge_tool = tool_registry.get_tool("retrieve_pg_knowledge")
        if knowledge_tool is None:
            raise RuntimeError("retrieve_pg_knowledge tool is not available")

        retrieved = knowledge_tool.invoke(
            {
                "query": question,
                "top_k": top_k_default(),
                "context_window": context_window_default(),
                "rerank_candidates": rerank_candidates_default(),
            }
        )
        compressed_retrieved = _compress_text_to_token_budget(
            str(retrieved),
            int(_history_token_limit() * 0.3),
        )
        messages.append(StreamMessage.progress("知识库检索完成"))
        return {"knowledge_context": compressed_retrieved, "stream_messages": messages}
    except (RuntimeError, ValueError, TypeError, OSError, TimeoutError) as exc:
        logger.warning(
            "Knowledge retrieval recoverable error; err=%s",
            type(exc).__name__,
            exc_info=True,
        )
        messages.append(StreamMessage.error(f"知识检索失败: {exc}"))
        return {"knowledge_context": f"知识检索失败: {exc}", "stream_messages": messages}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Knowledge retrieval unexpected error")
        messages.append(StreamMessage.error(f"知识检索失败: {exc}"))
        return {"knowledge_context": f"知识检索失败: {exc}", "stream_messages": messages}


def _build_prompt_node_impl(
    state: QAState,
    *,
    render_prompt_fn: Callable[..., str],
    docker_workdir_mount_fn: Callable[[], object],
) -> QAState:
    question = str(state.get("question", "")).strip()
    route = str(state.get("route", "direct") or "direct")
    selected_skill_name = str(state.get("selected_skill", "") or "")
    memory_ctx = str(state.get("retrieved_context", "")).strip() or "(none)"
    knowledge_ctx = str(state.get("knowledge_context", "")).strip() or "(none)"
    now_local = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    mount_path = docker_workdir_mount_fn()
    docker_workdir = "/workspace"
    if mount_path is not None:
        docker_workdir = f"/workspace (host mount: {mount_path})"

    messages: list[StreamMessage] = []

    effective_skill_name = selected_skill_name
    if not effective_skill_name:
        effective_skill_name = "reading-companion" if route == "knowledge" else DEFAULT_SKILL_NAME
    effective_skill_name = resolve_skill_name(effective_skill_name)

    memory_ctx, knowledge_ctx = process_context_for_skill(effective_skill_name, memory_ctx, knowledge_ctx)
    prompt_key = resolve_prompt_key(effective_skill_name, "agents.langgraph.final_user_prompt")

    prompt = render_prompt_fn(
        prompt_key,
        now_local=now_local,
        question=question,
        memory_ctx=memory_ctx,
        knowledge_ctx=knowledge_ctx,
        docker_workdir=docker_workdir,
    )

    history = state.get("history", [])
    chat_messages: list[BaseMessage] = []
    if isinstance(history, list):
        history_msgs = [m for m in history if isinstance(m, BaseMessage)]
        compressed_history = _compress_history_by_token_budget(history_msgs, _history_token_limit())
        chat_messages.extend(compressed_history)
    chat_messages.append(HumanMessage(content=prompt))

    return {"final_prompt": prompt, "messages": chat_messages, "stream_messages": messages}


def _assistant_node_factory(model_factory: Callable):
    """Create assistant node with dynamic tool binding based on skill."""

    def _assistant_node(state: QAState) -> QAState:
        messages = state.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        selected_skill_name = resolve_skill_name(str(state.get("selected_skill", DEFAULT_SKILL_NAME) or DEFAULT_SKILL_NAME))

        filtered_tools = select_tools_for_skill(selected_skill_name)
        filtered_tool_schemas = [
            tool.to_openai_tool() if hasattr(tool, "to_openai_tool") else tool
            for tool in filtered_tools
        ]

        base_model = model_factory()
        model_with_tools = base_model.bind_tools(filtered_tool_schemas)

        stream_messages: list[StreamMessage] = []

        merged_chunk: AIMessageChunk | None = None
        try:
            for chunk in model_with_tools.stream(messages):
                if not isinstance(chunk, AIMessageChunk):
                    continue
                if merged_chunk is None:
                    merged_chunk = chunk
                else:
                    merged_chunk = merged_chunk + chunk
        except (RuntimeError, ValueError, TypeError, OSError, TimeoutError) as exc:
            logger.warning(
                "Model stream failed, fallback to invoke; err=%s",
                type(exc).__name__,
                exc_info=True,
            )
            merged_chunk = None
        except Exception as exc:  # noqa: BLE001
            logger.exception("Model stream unexpected error, fallback to invoke")
            merged_chunk = None

        if merged_chunk is None:
            resp = model_with_tools.invoke(messages)
        else:
            streamed_tool_calls = getattr(merged_chunk, "tool_calls", None) or []
            streamed_tool_call_chunks = getattr(merged_chunk, "tool_call_chunks", None) or []
            if streamed_tool_call_chunks and not streamed_tool_calls:
                resp = model_with_tools.invoke(messages)
            else:
                resp = AIMessage(
                    content=str(getattr(merged_chunk, "content", "") or ""),
                    tool_calls=streamed_tool_calls,
                )

        tool_calls = getattr(resp, "tool_calls", None) or []
        if tool_calls:
            for call in tool_calls:
                tool_name = str(call.get("name", "")).strip() or "unknown_tool"
                args = call.get("args", {})
                display_name = resolve_tool_display_name(selected_skill_name, tool_name)
                stream_messages.append(
                    StreamMessage.tool_start(
                        tool_name,
                        display_name=display_name,
                        args=args,
                    )
                )

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


def _tools_node(state: QAState) -> QAState:
    messages = state.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return {"messages": []}

    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return {"messages": []}

    tool_messages: list[ToolMessage] = []
    for call in tool_calls:
        tool_name = str(call.get("name", "") or "").strip()
        call_id = str(call.get("id", "") or "").strip()
        raw_args = call.get("args", {})

        if not tool_name:
            continue

        result = _safe_invoke_tool(tool_name, raw_args)

        tool_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=call_id or f"missing_call_id_{tool_name}",
            )
        )

    return {"messages": tool_messages}


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

    stream_messages.append(StreamMessage.final_answer(answer))

    return {"answer": answer, "stream_messages": stream_messages}


def _extract_answer(result: dict) -> str:
    if not isinstance(result, dict):
        return str(result)
    return str(result.get("answer", ""))


__all__ = [
    "_plan_node",
    "_skill_planning_node",
    "_knowledge_node",
    "_build_prompt_node_impl",
    "_assistant_node_factory",
    "_should_call_tools",
    "_tools_node",
    "_finalize_node",
    "_extract_answer",
    "_extract_selected_skill_from_response",
]
