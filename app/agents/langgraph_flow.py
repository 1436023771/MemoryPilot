from __future__ import annotations

from datetime import datetime
import os
import re
from typing import Callable
from typing import Annotated
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

from app.agents.tool_registry import tool_registry
from app.agents.stream_messages import StreamMessage, MessageType
from app.agents.skills import (
    DEFAULT_SKILL_NAME,
    process_context_for_skill,
    resolve_prompt_key,
    resolve_skill_name,
    resolve_tool_display_name,
    select_tools_for_skill,
    skill_registry,
)
from app.config.langgraph import (
    langchain_project,
    max_history_tokens,
    DEFAULT_MAX_HISTORY_TOKENS,
    top_k_default,
    context_window_default,
    rerank_candidates_default,
)
from app.config.execution import docker_workdir_mount
from app.config.execution import llmlingua_mcp_enabled
from app.agents.llmlingua_mcp_client import compress_text_via_llmlingua_mcp, compress_history_via_llmlingua_mcp
from app.config import get_settings
from app.core.config import Settings
from app.core.prompt_store import render_prompt


DEFAULT_MAX_HISTORY_TOKENS = 1800


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
    selected_skill: str  # 新增：LLM 选择的 skill 名称
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
    project = langchain_project()
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


def _estimate_text_tokens(text: str) -> int:
    """Rough token estimate without external tokenizer dependency."""
    raw = str(text or "")
    if not raw:
        return 0

    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", raw))
    latin_words = re.findall(r"[a-zA-Z0-9_]+", raw)
    latin_chars = sum(len(w) for w in latin_words)
    symbol_chars = len(re.findall(r"[^\w\s\u4e00-\u9fff]", raw))
    latin_tokens = max(1, latin_chars // 4) if latin_chars > 0 else 0

    # CJK roughly maps close to one token per character; latin text is denser.
    return cjk_count + latin_tokens + max(0, symbol_chars // 6)


def _estimate_message_tokens(msg: BaseMessage) -> int:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return _estimate_text_tokens(content)
    if isinstance(content, list):
        joined = " ".join(str(item) for item in content)
        return _estimate_text_tokens(joined)
    return _estimate_text_tokens(str(content))


def _history_token_limit() -> int:
    return max(300, max_history_tokens())


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[。！？!?；;\.\n])\s*", text or "") if p and p.strip()]
    return parts


def _extract_key_tokens(text: str) -> list[str]:
    """Extract key tokens that should be preserved across compression."""
    raw = str(text or "")
    if not raw:
        return []

    patterns = [
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",  # date-like 2026-03-28
        r"\b\d{1,2}:\d{2}\b",  # time-like 08:30
        r"\b\d+(?:\.\d+)?%\b",  # percentage
        r"\b\d+(?:\.\d+)?(?:k|m|b|w)?\b",  # numbers
        r"\b(?:v|V)?\d+(?:\.\d+){1,3}\b",  # version-like
        r"\b[a-zA-Z_]+\s*=\s*[^\s,;，；]+",  # key=value hints
    ]

    negation_terms = ["不", "不能", "不要", "不可", "仅", "必须", "except", "unless", "not", "only"]

    seen: set[str] = set()
    kept: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, raw):
            token = m.group(0).strip()
            if token and token not in seen:
                seen.add(token)
                kept.append(token)

    for word in negation_terms:
        if word in raw and word not in seen:
            seen.add(word)
            kept.append(word)

    return kept[:12]


def _contains_key_token(text: str, token: str) -> bool:
    if not token:
        return False
    return token in (text or "")


def _compress_text_head_tail_to_token_budget(text: str, max_tokens: int) -> str:
    """Fallback head-tail compression with hard budget guarantee."""
    raw = str(text or "")
    if not raw:
        return raw
    if max_tokens <= 0:
        return ""

    estimated = _estimate_text_tokens(raw)
    if estimated <= max_tokens:
        return raw

    keep_ratio = max(0.08, min(1.0, max_tokens / max(estimated, 1)))
    keep_chars = max(6, int(len(raw) * keep_ratio * 0.95))
    if keep_chars >= len(raw):
        candidate = raw
    elif keep_chars <= 12:
        candidate = raw[:keep_chars].rstrip()
    else:
        head_len = int(keep_chars * 0.7)
        tail_len = max(0, keep_chars - head_len)
        head = raw[:head_len].rstrip()
        tail = raw[-tail_len:].lstrip() if tail_len > 0 else ""
        candidate = f"{head} ... {tail}" if tail else head

    if _estimate_text_tokens(candidate) <= max_tokens:
        return candidate

    # Hard fallback: shrink aggressively until budget is satisfied.
    hard_chars = min(len(raw), max_tokens)
    hard = raw[: max(1, hard_chars)].rstrip()
    while hard and _estimate_text_tokens(hard) > max_tokens:
        next_len = max(1, len(hard) - 1)
        hard = hard[:next_len].rstrip()
        if next_len == 1:
            break
    return hard


def _extractive_compress_with_keys(text: str, max_tokens: int, key_tokens: list[str]) -> str:
    """Extractive sentence compression with key-token-aware scoring."""
    sentences = _split_sentences(text)
    if not sentences:
        return _compress_text_head_tail_to_token_budget(text, max_tokens)

    scored: list[tuple[float, int, str, int]] = []
    for idx, sent in enumerate(sentences):
        sent_tokens = _estimate_text_tokens(sent)
        if sent_tokens <= 0:
            continue
        key_hits = sum(1 for k in key_tokens if _contains_key_token(sent, k))
        has_digit = 1.0 if re.search(r"\d", sent) else 0.0
        has_negation = 1.0 if any(x in sent for x in ("不", "不能", "不要", "不可", "仅", "必须", "not", "only")) else 0.0
        density = min(1.0, sent_tokens / 24)
        score = key_hits * 3.0 + has_digit * 0.8 + has_negation * 1.0 + density * 0.4
        scored.append((score, idx, sent, sent_tokens))

    if not scored:
        return _compress_text_head_tail_to_token_budget(text, max_tokens)

    scored.sort(key=lambda x: (x[0], -x[3]), reverse=True)

    selected_idx: set[int] = set()
    used = 0
    for _score, idx, sent, sent_tokens in scored:
        if sent_tokens > max_tokens:
            continue
        if used + sent_tokens > max_tokens:
            continue
        selected_idx.add(idx)
        used += sent_tokens
        if used >= int(max_tokens * 0.9):
            break

    if not selected_idx:
        best = scored[0][2]
        return _compress_text_head_tail_to_token_budget(best, max_tokens)

    ordered = [sentences[i] for i in sorted(selected_idx)]
    candidate = " ".join(ordered).strip()
    if _estimate_text_tokens(candidate) <= max_tokens:
        return candidate
    return _compress_text_head_tail_to_token_budget(candidate, max_tokens)


def _inject_missing_key_tokens(base: str, key_tokens: list[str], max_tokens: int) -> str:
    """Try appending missing key tokens while staying under budget."""
    result = base.strip()
    for token in key_tokens:
        if _contains_key_token(result, token):
            continue
        trial = f"{result} [{token}]" if result else token
        if _estimate_text_tokens(trial) <= max_tokens:
            result = trial
    return result


def _compress_text_to_token_budget(text: str, max_tokens: int) -> str:
    """Hybrid compression: key-token preservation + local extractive summarization."""
    raw = str(text or "")
    if not raw:
        return raw
    if max_tokens <= 0:
        return ""

    if llmlingua_mcp_enabled():
        key_tokens = _extract_key_tokens(raw)
        mcp_compressed = compress_text_via_llmlingua_mcp(raw, int(max_tokens), preserve_keywords=key_tokens)
        if isinstance(mcp_compressed, str) and mcp_compressed.strip():
            return mcp_compressed

    estimated = _estimate_text_tokens(raw)
    if estimated <= max_tokens:
        return raw

    key_tokens = _extract_key_tokens(raw)
    extracted = _extractive_compress_with_keys(raw, max_tokens, key_tokens)
    with_keys = _inject_missing_key_tokens(extracted, key_tokens, max_tokens)

    if _estimate_text_tokens(with_keys) <= max_tokens:
        return with_keys

    return _compress_text_head_tail_to_token_budget(with_keys, max_tokens)


def _copy_message_with_content(msg: BaseMessage, content: str) -> BaseMessage:
    """Clone message while preserving role-specific metadata."""
    if isinstance(msg, HumanMessage):
        return HumanMessage(content=content, additional_kwargs=getattr(msg, "additional_kwargs", {}))
    if isinstance(msg, AIMessage):
        return AIMessage(content=content, additional_kwargs=getattr(msg, "additional_kwargs", {}))
    if isinstance(msg, SystemMessage):
        return SystemMessage(content=content, additional_kwargs=getattr(msg, "additional_kwargs", {}))
    if isinstance(msg, ToolMessage):
        return ToolMessage(
            content=content,
            tool_call_id=str(getattr(msg, "tool_call_id", "") or ""),
            additional_kwargs=getattr(msg, "additional_kwargs", {}),
        )

    try:
        return msg.__class__(content=content)
    except Exception:  # noqa: BLE001
        return HumanMessage(content=content)


def _message_role_name(msg: BaseMessage) -> str:
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, AIMessage):
        return "assistant"
    if isinstance(msg, SystemMessage):
        return "system"
    if isinstance(msg, ToolMessage):
        return "tool"
    return "user"


def _compress_history_by_token_budget(history: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """Compress message contents to fit budget while keeping all message turns."""
    if not history:
        return []
    if max_tokens <= 0:
        return [_copy_message_with_content(msg, "") for msg in history]

    if llmlingua_mcp_enabled():
        role_payload: list[dict[str, str]] = []
        for msg in history:
            role = _message_role_name(msg)

            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                content = str(content)
            role_payload.append({"role": role, "content": content})

        compressed_payload = compress_history_via_llmlingua_mcp(role_payload, int(max_tokens))
        if isinstance(compressed_payload, list) and len(compressed_payload) == len(history):
            roles_match = True
            for msg, payload in zip(history, compressed_payload, strict=False):
                payload_role = str(payload.get("role", "") or "")
                if payload_role != _message_role_name(msg):
                    roles_match = False
                    break

            if not roles_match:
                compressed_payload = None

        if isinstance(compressed_payload, list) and len(compressed_payload) == len(history):
            compressed_msgs: list[BaseMessage] = []
            for msg, payload in zip(history, compressed_payload, strict=False):
                new_content = str(payload.get("content", "") or "")
                compressed_msgs.append(_copy_message_with_content(msg, new_content))
            return compressed_msgs

    costs = [_estimate_message_tokens(msg) for msg in history]
    total = sum(costs)
    if total <= max_tokens:
        return history

    n = len(history)
    # Keep at least one estimated token budget per message to preserve turns.
    minima = [1] * n
    budget_left = max(0, max_tokens - sum(minima))
    capacities = [max(0, c - 1) for c in costs]

    # Prefer recent messages by assigning larger weights to later indices.
    weights = [i + 1 for i in range(n)]
    weight_sum = sum(weights) or 1
    extras = [0] * n

    for i in range(n):
        if capacities[i] <= 0 or budget_left <= 0:
            continue
        alloc = int(budget_left * (weights[i] / weight_sum))
        use = min(capacities[i], alloc)
        extras[i] = use

    used_extra = sum(extras)
    remaining = max(0, budget_left - used_extra)
    for i in range(n - 1, -1, -1):
        if remaining <= 0:
            break
        spare = capacities[i] - extras[i]
        if spare <= 0:
            continue
        take = min(spare, remaining)
        extras[i] += take
        remaining -= take

    targets = [minima[i] + extras[i] for i in range(n)]
    compressed: list[BaseMessage] = []
    for msg, target in zip(history, targets, strict=False):
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            content = str(content)
        new_content = _compress_text_to_token_budget(content, target)
        compressed.append(_copy_message_with_content(msg, new_content))
    return compressed


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
    
    # Get available skills
    skills_list = skill_registry.list_descriptions()
    if not skills_list:
        return {"selected_skill": DEFAULT_SKILL_NAME}
    
    # Create LLM decision prompt
    decision_prompt = f"""根据用户问题，选择最合适的处理mode。

可用的模式（skills）：
{skills_list}

用户问题：
{question}

请返回JSON格式的决策结果，只返回JSON，不要任何其他文字。
格式: {{"selected_skill": "skill_name"}}
如果无法判断，选择 "general"。
"""
    
    try:
        from langchain_openai import ChatOpenAI

        settings = get_settings()
        model = ChatOpenAI(
            model=settings.model_name,
            temperature=0.3,
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
        
        response = model.invoke(decision_prompt)
        content = str(getattr(response, "content", "") or "").strip()
        
        # Parse JSON response
        import json
        import re
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                selected_skill = str(result.get("selected_skill", DEFAULT_SKILL_NAME)).strip()
            else:
                selected_skill = DEFAULT_SKILL_NAME
        except (json.JSONDecodeError, ValueError):
            selected_skill = DEFAULT_SKILL_NAME

        return {"selected_skill": resolve_skill_name(selected_skill)}
    
    except Exception as e:
        # Fallback to general on any error
        import logging
        logging.warning(f"Skill planning error: {e}, falling back to general")
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

        default_top_k = top_k_default()
        default_context_window = context_window_default()
        default_rerank_candidates = rerank_candidates_default()
        # 先做召回+重排，再把结果回传给回答节点。
        retrieved = knowledge_tool.invoke(
            {
                "query": question,
                "top_k": default_top_k,
                "context_window": default_context_window,
                "rerank_candidates": default_rerank_candidates,
            }
        )
        compressed_retrieved = _compress_text_to_token_budget(
            str(retrieved),
            int(_history_token_limit() * 0.3),
        )
        messages.append(StreamMessage.progress("知识库检索完成"))
        return {"knowledge_context": compressed_retrieved, "stream_messages": messages}
    except Exception as exc:  # noqa: BLE001
        messages.append(StreamMessage.error(f"知识检索失败: {exc}"))
        return {"knowledge_context": f"知识检索失败: {exc}", "stream_messages": messages}


def _build_prompt_node(state: QAState) -> QAState:
    question = str(state.get("question", "")).strip()
    route = str(state.get("route", "direct") or "direct")
    selected_skill_name = str(state.get("selected_skill", "") or "")
    memory_ctx = str(state.get("retrieved_context", "")).strip() or "(none)"
    knowledge_ctx = str(state.get("knowledge_context", "")).strip() or "(none)"
    now_local = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    mount_path = docker_workdir_mount()
    # Docker container side workdir is always /workspace; host path is only a mount source detail.
    docker_workdir = "/workspace"
    if mount_path is not None:
        docker_workdir = f"/workspace (host mount: {mount_path})"

    messages: list[StreamMessage] = []

    # Backward compatibility: if no skill was selected yet, infer from route.
    effective_skill_name = selected_skill_name
    if not effective_skill_name:
        effective_skill_name = "reading-companion" if route == "knowledge" else DEFAULT_SKILL_NAME
    effective_skill_name = resolve_skill_name(effective_skill_name)

    memory_ctx, knowledge_ctx = process_context_for_skill(effective_skill_name, memory_ctx, knowledge_ctx)
    prompt_key = resolve_prompt_key(effective_skill_name, "agents.langgraph.final_user_prompt")

    prompt = render_prompt(
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
    """Create assistant node with dynamic tool binding based on skill.
    
    Args:
        model_factory: Callable that returns a ChatOpenAI model.
                      Used to create models with skill-specific tools.
    """
    def _assistant_node(state: QAState) -> QAState:
        messages = state.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        # Get selected skill and filter tools accordingly
        selected_skill_name = resolve_skill_name(str(state.get("selected_skill", DEFAULT_SKILL_NAME) or DEFAULT_SKILL_NAME))
        
        # Determine which tools to bind
        filtered_tools = select_tools_for_skill(selected_skill_name)
        
        # Create model with skill-specific tools
        base_model = model_factory()
        model_with_tools = base_model.bind_tools(filtered_tools)

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
        except Exception:  # noqa: BLE001
            merged_chunk = None

        if merged_chunk is None:
            resp = model_with_tools.invoke(messages)
        else:
            # 流式场景下部分 provider 仅返回 tool_call_chunks，不一定能可靠组装 tool_calls。
            # 出现该情况时回退到 invoke，保证工具路由正确。
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

    # Create a model factory lambda that can be called at runtime to create fresh models
    model_factory = lambda: ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )

    graph = StateGraph(QAState)
    graph.add_node("plan", RunnableLambda(_plan_node).with_config(_node_config("plan")))
    graph.add_node("skill_planning", RunnableLambda(_skill_planning_node).with_config(_node_config("skill_planning")))
    graph.add_node(
        "retrieve_knowledge",
        RunnableLambda(_knowledge_node).with_config(_node_config("retrieve_knowledge", ["retrieval"])),
    )
    graph.add_node("build_prompt", RunnableLambda(_build_prompt_node).with_config(_node_config("build_prompt")))
    graph.add_node(
        "assistant",
        RunnableLambda(_assistant_node_factory(model_factory)).with_config(_node_config("assistant", ["llm"])),
    )
    graph.add_node(
        "tools",
        ToolNode(tool_registry.get_all_tools()).with_config(_node_config("tools", ["tool-calls"])),
    )
    graph.add_node("finalize", RunnableLambda(_finalize_node).with_config(_node_config("finalize")))

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "skill_planning")
    graph.add_conditional_edges(
        "skill_planning",
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

        session_history = None
        history_messages: list[BaseMessage] = []
        if session_id:
            try:
                session_history = self._session_history_getter(session_id)
                existing = getattr(session_history, "messages", [])
                if isinstance(existing, list):
                    history_messages.extend(m for m in existing if isinstance(m, BaseMessage))
            except Exception:  # noqa: BLE001
                session_history = None

        provided_history = input_data.get("history", [])
        if isinstance(provided_history, list):
            history_messages.extend(m for m in provided_history if isinstance(m, BaseMessage))
        
        # 准备输入状态
        state_input = {
            "question": str(input_data.get("question", "")),
            "retrieved_context": str(input_data.get("retrieved_context", "")),
            "history": history_messages,
            "stream_messages": [],
        }
        
        # 流式执行graph，收集中间消息
        collected_messages = []
        final_state = None
        yielded_fingerprints: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
        seen_tool_message_ids: set[str] = set()
        tool_name_by_call_id: dict[str, str] = {}
        saw_token_delta = False
        answer_chunks: list[str] = []

        def _msg_fp(msg: StreamMessage) -> tuple[str, str, tuple[tuple[str, str], ...]]:
            meta_pairs = tuple(sorted((str(k), str(v)) for k, v in msg.metadata.items()))
            return (msg.message_type.value, msg.content, meta_pairs)

        def _extract_delta_text(chunk_obj) -> str:
            if not isinstance(chunk_obj, AIMessageChunk):
                return ""
            content = getattr(chunk_obj, "content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                        continue
                    if isinstance(item, dict):
                        # OpenAI-compatible content blocks: {"type":"text","text":"..."}
                        text = item.get("text", "")
                        if isinstance(text, str) and text:
                            parts.append(text)
                return "".join(parts)
            return ""

        def _handle_update_state(node_name: str, state: dict) -> list[StreamMessage]:
            nonlocal final_state
            final_state = state

            out_messages: list[StreamMessage] = []

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
                    # token delta 允许重复文本，不做内容去重。
                    is_delta = bool(msg.metadata.get("is_delta", False)) if isinstance(msg.metadata, dict) else False
                    if msg.message_type == MessageType.FINAL_ANSWER and is_delta:
                        out_messages.append(msg)
                        continue

                    # 非增量的 FINAL_ANSWER（通常来自 finalize 节点整段输出）
                    # 在此统一不透传，避免与 token 流重复。
                    # 若当前provider不支持token流，函数末尾会用 final_state.answer 做兜底一次性输出。
                    if msg.message_type == MessageType.FINAL_ANSWER and not is_delta:
                        continue

                    fp = _msg_fp(msg)
                    if fp in yielded_fingerprints:
                        continue
                    yielded_fingerprints.add(fp)
                    out_messages.append(msg)

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
                        out_messages.append(tool_result_msg)

            return out_messages

        graph_stream = None
        try:
            graph_stream = self._graph.stream(
                state_input,
                config=kwargs.get("config"),
                stream_mode=["updates", "messages"],
            )
        except TypeError:
            graph_stream = self._graph.stream(state_input, config=kwargs.get("config"))

        for event in graph_stream:
            # 多 stream_mode 输出：(mode, payload)
            if isinstance(event, tuple) and len(event) == 2 and isinstance(event[0], str):
                mode, payload = event
                if mode == "updates" and isinstance(payload, dict):
                    for node_name, state in payload.items():
                        for msg in _handle_update_state(node_name, state):
                            collected_messages.append(msg)
                            yield msg
                elif mode == "messages":
                    # payload 可能是 (chunk, metadata)
                    metadata = payload[1] if isinstance(payload, tuple) and len(payload) > 1 else {}
                    if isinstance(metadata, dict):
                        node_name = str(metadata.get("langgraph_node", "")).strip()
                        if node_name and node_name != "assistant":
                            continue

                    chunk_obj = payload[0] if isinstance(payload, tuple) and payload else payload
                    delta_text = _extract_delta_text(chunk_obj)
                    if not delta_text:
                        continue
                    saw_token_delta = True
                    answer_chunks.append(delta_text)
                    delta_msg = StreamMessage.final_answer(delta_text, is_delta=True)
                    collected_messages.append(delta_msg)
                    yield delta_msg
                continue

            # 兼容旧格式：updates 直接是 dict
            if isinstance(event, dict):
                for node_name, state in event.items():
                    for msg in _handle_update_state(node_name, state):
                        collected_messages.append(msg)
                        yield msg
                continue

            # 兼容旧格式：messages 直接是 (chunk, metadata)
            if isinstance(event, tuple) and len(event) == 2:
                metadata = event[1]
                if isinstance(metadata, dict):
                    node_name = str(metadata.get("langgraph_node", "")).strip()
                    if node_name and node_name != "assistant":
                        continue

                chunk_obj = event[0]
                delta_text = _extract_delta_text(chunk_obj)
                if not delta_text:
                    continue
                saw_token_delta = True
                answer_chunks.append(delta_text)
                delta_msg = StreamMessage.final_answer(delta_text, is_delta=True)
                collected_messages.append(delta_msg)
                yield delta_msg
        
        final_answer_text = ""
        if saw_token_delta:
            final_answer_text = "".join(answer_chunks)
        elif final_state and "answer" in final_state:
            final_answer_text = str(final_state.get("answer", "") or "")

        # stream模式不会经过RunnableWithMessageHistory，需手动回写短期会话历史。
        if session_history is not None:
            try:
                question = str(input_data.get("question", "")).strip()
                to_add: list[BaseMessage] = []
                if question:
                    to_add.append(HumanMessage(content=question))
                if final_answer_text.strip():
                    to_add.append(AIMessage(content=final_answer_text))
                if to_add:
                    session_history.add_messages(to_add)
            except Exception:  # noqa: BLE001
                pass

        # 确保至少返回一个最终答案消息
        if final_state and "answer" in final_state and not saw_token_delta:
            answer = final_state.get("answer", "")
            if answer and (not collected_messages or collected_messages[-1].message_type != MessageType.FINAL_ANSWER):
                yield StreamMessage.final_answer(answer)
