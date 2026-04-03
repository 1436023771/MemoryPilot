from __future__ import annotations

from typing import Callable

from app.agents.langgraph.graph_spec import EdgeSpec, GRAPH_END, GRAPH_START, NodeSpec
from app.agents.langgraph.nodes import (
    _assistant_node_factory,
    _finalize_node,
    _knowledge_node,
    _plan_node,
    _should_call_tools,
    _skill_planning_node,
    _tools_node,
)
from app.agents.langgraph.state import QAState, _route_from_plan
from app.agents.llm_client import create_chat_model
from app.core.config import Settings

StateNodeFn = Callable[[QAState], QAState]
NodeFactory = Callable[[], StateNodeFn]
RouterFn = Callable[[QAState], str]


def build_node_factories(
    settings: Settings,
    build_prompt_node: StateNodeFn,
) -> dict[str, NodeFactory]:
    """Build all node factories from runtime dependencies."""
    model_factory = lambda: create_chat_model(settings=settings)
    return {
        "plan": lambda: _plan_node,
        "skill_planning": lambda: _skill_planning_node,
        "retrieve_knowledge": lambda: _knowledge_node,
        "build_prompt": lambda: build_prompt_node,
        "assistant": lambda: _assistant_node_factory(model_factory),
        "tools": lambda: _tools_node,
        "finalize": lambda: _finalize_node,
    }


def build_routers() -> dict[str, RouterFn]:
    """Build router function registry for conditional edges."""
    return {
        "route_from_plan": _route_from_plan,
        "should_call_tools": _should_call_tools,
    }


def validate_graph_tables(
    node_table: list[NodeSpec],
    edge_table: list[EdgeSpec],
    node_factories: dict[str, NodeFactory],
    routers: dict[str, RouterFn],
) -> None:
    """Validate graph spec and runtime registries before graph assembly."""
    node_names = [node["name"] for node in node_table]
    node_name_set = set(node_names)

    if len(node_names) != len(node_name_set):
        duplicates = sorted({name for name in node_names if node_names.count(name) > 1})
        raise ValueError(f"Duplicate node names found: {duplicates}")

    for node in node_table:
        factory_key = node["factory"]
        if factory_key not in node_factories:
            raise ValueError(f"Unknown node factory '{factory_key}' for node '{node['name']}'")

    for edge in edge_table:
        source = edge["source"]
        if source != GRAPH_START and source not in node_name_set:
            raise ValueError(f"Unknown edge source '{source}'")

        if edge["kind"] == "direct":
            target = edge["target"]
            if target != GRAPH_END and target not in node_name_set:
                raise ValueError(f"Unknown edge target '{target}'")
            continue

        router_key = edge["router"]
        if router_key not in routers:
            raise ValueError(f"Unknown router '{router_key}' on edge from '{source}'")

        for route_key, target in edge["mapping"].items():
            if target not in node_name_set:
                raise ValueError(
                    f"Unknown conditional target '{target}' for route '{route_key}' from '{source}'"
                )


__all__ = [
    "StateNodeFn",
    "NodeFactory",
    "RouterFn",
    "build_node_factories",
    "build_routers",
    "validate_graph_tables",
]
