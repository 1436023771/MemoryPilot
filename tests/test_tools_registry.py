import app.agents.tools as tools_registry
from app.agents.tool_registry import tool_registry


def test_registry_auto_exports_core_tools() -> None:
    assert "web_search" in tools_registry.__all__
    assert "run_python_code" in tools_registry.__all__
    assert "run_docker_command" in tools_registry.__all__
    assert "retrieve_pg_knowledge" in tools_registry.__all__

    assert hasattr(tools_registry, "web_search")
    assert hasattr(tools_registry, "run_python_code")
    assert hasattr(tools_registry, "run_docker_command")
    assert hasattr(tools_registry, "retrieve_pg_knowledge")


def test_registry_keeps_helper_exports() -> None:
    assert "get_search_log" in tools_registry.__all__
    assert "clear_search_log" in tools_registry.__all__
    assert "get_knowledge_retrieval_log" in tools_registry.__all__
    assert "clear_knowledge_retrieval_log" in tools_registry.__all__
    assert "get_docker_exec_log" in tools_registry.__all__
    assert "clear_docker_exec_log" in tools_registry.__all__


def test_tool_registry_filters_to_callable_tools() -> None:
    tool_names = tool_registry.get_tool_names()

    assert "web_search" in tool_names
    assert "run_python_code" in tool_names
    assert "run_docker_command" in tool_names
    assert "retrieve_pg_knowledge" in tool_names
    assert "get_search_log" not in tool_names


def test_tool_registry_can_filter_allowed_tools() -> None:
    selected = tool_registry.get_tools(["web_search", "run_python_code"])

    assert [tool.name for tool in selected] == ["web_search", "run_python_code"]
