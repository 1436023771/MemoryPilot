"""LangGraph runtime internals split from monolithic flow file."""

from __future__ import annotations

from typing import Callable

from app.agents.langgraph.factory import build_langgraph_chain_impl
from app.agents.langgraph.history_compression import (
    _compress_history_by_token_budget,
    _compress_text_to_token_budget,
    _estimate_message_tokens,
    _estimate_text_tokens,
    _history_token_limit,
)
from app.agents.langgraph.nodes import _build_prompt_node_impl, _finalize_node
from app.agents.langgraph.state import QAState
from app.agents.langgraph.stream_adapter import StreamingLanggraphChain
from app.agents.session_history import SessionHistory
from app.config.execution import docker_workdir_mount
from app.core.config import Settings
from app.core.prompt_store import render_prompt


def _build_prompt_node(state: QAState) -> QAState:
    return _build_prompt_node_impl(
        state,
        render_prompt_fn=render_prompt,
        docker_workdir_mount_fn=docker_workdir_mount,
    )


def build_langgraph_chain(
    settings: Settings,
    get_session_history: Callable[[str], SessionHistory],
):
    return build_langgraph_chain_impl(
        settings=settings,
        get_session_history=get_session_history,
        build_prompt_node=_build_prompt_node,
    )


__all__ = [
    "build_langgraph_chain",
    "build_langgraph_chain_impl",
    "StreamingLanggraphChain",
    "QAState",
    "_finalize_node",
    "_compress_history_by_token_budget",
    "_compress_text_to_token_budget",
    "_estimate_message_tokens",
    "_estimate_text_tokens",
    "_history_token_limit",
]
