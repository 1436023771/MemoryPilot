from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage

from app.agents.session_history import SessionHistory
from app.agents.stream_messages import MessageType, StreamMessage


class StreamingLanggraphChain:
    """Wrapper around compiled LangGraph with invoke/stream interfaces."""

    def __init__(self, compiled_graph, compiled_app, base_chain, session_history_getter, node_config_getter):
        self._graph = compiled_graph
        self._app = compiled_app
        self._base_chain = base_chain
        self._session_history_getter = session_history_getter
        self._node_config_getter = node_config_getter

    def _resolve_session_id(self, explicit_session_id: str | None, kwargs: dict) -> str | None:
        if explicit_session_id:
            return explicit_session_id

        config = kwargs.get("config")
        if isinstance(config, dict):
            configurable = config.get("configurable")
            if isinstance(configurable, dict):
                value = configurable.get("session_id")
                if value is not None:
                    return str(value)
        return None

    def _load_history_messages(self, session_id: str | None, input_data: dict) -> tuple[SessionHistory | None, list[BaseMessage]]:
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

        return session_history, history_messages

    def _build_state_input(self, input_data: dict, history_messages: list[BaseMessage]) -> dict:
        return {
            "question": str(input_data.get("question", "")),
            "retrieved_context": str(input_data.get("retrieved_context", "")),
            "history": history_messages,
            "stream_messages": [],
        }

    def _extract_final_answer(self, final_state) -> str:
        if isinstance(final_state, dict):
            answer = str(final_state.get("answer", "") or "")
            if answer.strip():
                return answer

            messages = final_state.get("messages", [])
            if isinstance(messages, list):
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        return str(getattr(msg, "content", "") or "")
        return ""

    def _persist_session_history(self, session_history: SessionHistory | None, question: str, answer: str) -> None:
        if session_history is None:
            return

        try:
            to_add: list[BaseMessage] = []
            normalized_question = str(question or "").strip()
            normalized_answer = str(answer or "").strip()
            if normalized_question:
                to_add.append(HumanMessage(content=normalized_question))
            if normalized_answer:
                to_add.append(AIMessage(content=normalized_answer))
            if to_add:
                session_history.add_messages(to_add)
        except Exception:  # noqa: BLE001
            pass

    def invoke(self, input_data: dict, **kwargs) -> str:
        """Synchronous invoke returning final answer text."""
        session_id = self._resolve_session_id(None, kwargs)
        session_history, history_messages = self._load_history_messages(session_id, input_data)
        state_input = self._build_state_input(input_data, history_messages)

        final_state = self._graph.invoke(state_input, config=kwargs.get("config"))
        answer = self._extract_final_answer(final_state)

        self._persist_session_history(
            session_history=session_history,
            question=str(input_data.get("question", "") or ""),
            answer=answer,
        )
        return answer

    def stream(self, input_data: dict, session_id: str | None = None, **kwargs):
        """Stream intermediate messages and final answer deltas."""
        session_id = self._resolve_session_id(session_id, kwargs)
        session_history, history_messages = self._load_history_messages(session_id, input_data)
        state_input = self._build_state_input(input_data, history_messages)

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
                        text = item.get("text", "")
                        if isinstance(text, str) and text:
                            parts.append(text)
                return "".join(parts)
            return ""

        def _handle_update_state(node_name: str, state: dict) -> list[StreamMessage]:
            nonlocal final_state
            final_state = state

            out_messages: list[StreamMessage] = []

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
                    is_delta = bool(msg.metadata.get("is_delta", False)) if isinstance(msg.metadata, dict) else False
                    if msg.message_type == MessageType.FINAL_ANSWER and is_delta:
                        out_messages.append(msg)
                        continue

                    if msg.message_type == MessageType.FINAL_ANSWER and not is_delta:
                        continue

                    fp = _msg_fp(msg)
                    if fp in yielded_fingerprints:
                        continue
                    yielded_fingerprints.add(fp)
                    out_messages.append(msg)

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
            if isinstance(event, tuple) and len(event) == 2 and isinstance(event[0], str):
                mode, payload = event
                if mode == "updates" and isinstance(payload, dict):
                    for node_name, state in payload.items():
                        for msg in _handle_update_state(node_name, state):
                            collected_messages.append(msg)
                            yield msg
                elif mode == "messages":
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

            if isinstance(event, dict):
                for node_name, state in event.items():
                    for msg in _handle_update_state(node_name, state):
                        collected_messages.append(msg)
                        yield msg
                continue

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

        self._persist_session_history(
            session_history=session_history,
            question=str(input_data.get("question", "") or ""),
            answer=final_answer_text,
        )

        if final_state and "answer" in final_state and not saw_token_delta:
            answer = final_state.get("answer", "")
            if answer and (not collected_messages or collected_messages[-1].message_type != MessageType.FINAL_ANSWER):
                yield StreamMessage.final_answer(answer)


__all__ = ["StreamingLanggraphChain"]
