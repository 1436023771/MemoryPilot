from __future__ import annotations

from typing import Callable

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from app.agents.langgraph.graph_runtime import build_node_factories, build_routers, validate_graph_tables
from app.agents.langgraph.graph_spec import EDGE_TABLE, GRAPH_END, GRAPH_START, NODE_TABLE
from app.agents.langgraph.state import QAState, _graph_config, _node_config
from app.agents.langgraph.stream_adapter import StreamingLanggraphChain
from app.agents.session_history import SessionHistory
from app.core.config import Settings


def build_langgraph_chain_impl(
    settings: Settings,
    get_session_history: Callable[[str], SessionHistory],
    build_prompt_node: Callable[[QAState], QAState],
):
    """Assemble and compile LangGraph, then wrap with stream-capable adapter."""
    node_factories = build_node_factories(settings=settings, build_prompt_node=build_prompt_node)
    routers = build_routers()

    validate_graph_tables(NODE_TABLE, EDGE_TABLE, node_factories, routers)

    graph = StateGraph(QAState)

    for node in NODE_TABLE:
        node_name = node["name"]
        node_fn = node_factories[node["factory"]]()
        graph.add_node(
            node_name,
            RunnableLambda(node_fn).with_config(_node_config(node_name, node["extra_tags"])),
        )

    for edge in EDGE_TABLE:
        source = START if edge["source"] == GRAPH_START else edge["source"]

        if edge["kind"] == "direct":
            target = END if edge["target"] == GRAPH_END else edge["target"]
            graph.add_edge(source, target)
            continue

        graph.add_conditional_edges(
            source,
            routers[edge["router"]],
            edge["mapping"],
        )

    compiled_graph = graph.compile().with_config(_graph_config())

    return StreamingLanggraphChain(
        compiled_graph,
        compiled_graph,
        None,
        get_session_history,
        _node_config,
    )


__all__ = ["build_langgraph_chain_impl"]
