from __future__ import annotations

from typing import Literal, TypedDict


GRAPH_START = "__start__"
GRAPH_END = "__end__"


class NodeSpec(TypedDict):
    """Declarative node definition for LangGraph assembly."""

    name: str
    factory: str
    extra_tags: list[str]


class DirectEdgeSpec(TypedDict):
    """A direct edge from source to target."""

    kind: Literal["direct"]
    source: str
    target: str


class ConditionalEdgeSpec(TypedDict):
    """A conditional edge with router and route-to-target mapping."""

    kind: Literal["conditional"]
    source: str
    router: str
    mapping: dict[str, str]


EdgeSpec = DirectEdgeSpec | ConditionalEdgeSpec


NODE_TABLE: list[NodeSpec] = [
    {"name": "plan", "factory": "plan", "extra_tags": []},
    {"name": "skill_planning", "factory": "skill_planning", "extra_tags": []},
    {"name": "retrieve_knowledge", "factory": "retrieve_knowledge", "extra_tags": ["retrieval"]},
    {"name": "build_prompt", "factory": "build_prompt", "extra_tags": []},
    {"name": "assistant", "factory": "assistant", "extra_tags": ["llm"]},
    {"name": "tools", "factory": "tools", "extra_tags": ["tool-calls"]},
    {"name": "finalize", "factory": "finalize", "extra_tags": []},
]


EDGE_TABLE: list[EdgeSpec] = [
    {"kind": "direct", "source": GRAPH_START, "target": "plan"},
    {"kind": "direct", "source": "plan", "target": "skill_planning"},
    {
        "kind": "conditional",
        "source": "skill_planning",
        "router": "route_from_plan",
        "mapping": {
            "knowledge": "retrieve_knowledge",
            "direct": "build_prompt",
        },
    },
    {"kind": "direct", "source": "retrieve_knowledge", "target": "build_prompt"},
    {"kind": "direct", "source": "build_prompt", "target": "assistant"},
    {
        "kind": "conditional",
        "source": "assistant",
        "router": "should_call_tools",
        "mapping": {
            "tools": "tools",
            "finalize": "finalize",
        },
    },
    {"kind": "direct", "source": "tools", "target": "assistant"},
    {"kind": "direct", "source": "finalize", "target": GRAPH_END},
]


__all__ = [
    "GRAPH_START",
    "GRAPH_END",
    "NodeSpec",
    "DirectEdgeSpec",
    "ConditionalEdgeSpec",
    "EdgeSpec",
    "NODE_TABLE",
    "EDGE_TABLE",
]
