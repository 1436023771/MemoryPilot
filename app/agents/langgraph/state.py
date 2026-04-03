from __future__ import annotations

from typing import Annotated
from typing import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.agents.stream_messages import StreamMessage
from app.config.langgraph import langchain_project


def _add_stream_messages(left: list[StreamMessage], right: list[StreamMessage]) -> list[StreamMessage]:
    """Merge stream message lists for Annotated state field."""
    return left + right


class QAState(TypedDict, total=False):
    question: str
    retrieved_context: str
    history: list[BaseMessage]
    messages: Annotated[list[BaseMessage], add_messages]
    stream_messages: Annotated[list[StreamMessage], _add_stream_messages]
    route: str
    selected_skill: str
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


def _route_from_plan(state: QAState) -> str:
    return str(state.get("route", "direct"))


__all__ = [
    "QAState",
    "_node_config",
    "_graph_config",
    "_route_from_plan",
]
