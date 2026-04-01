import app.agents.tools as tools_registry


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
