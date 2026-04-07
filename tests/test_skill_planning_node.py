from types import SimpleNamespace

from app.agents.langgraph.nodes import _extract_selected_skill_from_response, _skill_planning_node
from app.agents.skills import DEFAULT_SKILL_NAME


def test_extract_selected_skill_from_strict_json() -> None:
    text = '{"selected_skill": "reading-companion"}'
    assert _extract_selected_skill_from_response(text) == "reading-companion"


def test_extract_selected_skill_from_wrapped_json() -> None:
    text = 'analysis...\n{"selected_skill": "coder"}\nextra text'
    assert _extract_selected_skill_from_response(text) == "coder"


def test_extract_selected_skill_returns_none_for_invalid_payload() -> None:
    text = 'not json { selected_skill: coder } end'
    assert _extract_selected_skill_from_response(text) is None


def test_skill_planning_node_falls_back_when_response_not_parseable(monkeypatch) -> None:
    monkeypatch.setattr("app.agents.langgraph.nodes.get_settings", lambda: object())
    monkeypatch.setattr("app.agents.langgraph.nodes.resolve_skill_name", lambda x: x)
    monkeypatch.setattr("app.agents.langgraph.nodes.skill_registry.list_descriptions", lambda: "general: default")
    monkeypatch.setattr(
        "app.agents.langgraph.nodes.invoke_model",
        lambda **_kwargs: SimpleNamespace(content="garbled response without json"),
    )

    result = _skill_planning_node({"question": "帮我写测试"})
    assert result.get("selected_skill") == DEFAULT_SKILL_NAME


def test_skill_planning_node_extracts_json_with_prefix_suffix(monkeypatch) -> None:
    monkeypatch.setattr("app.agents.langgraph.nodes.get_settings", lambda: object())
    monkeypatch.setattr("app.agents.langgraph.nodes.resolve_skill_name", lambda x: x)
    monkeypatch.setattr("app.agents.langgraph.nodes.skill_registry.list_descriptions", lambda: "coder: write code")
    monkeypatch.setattr(
        "app.agents.langgraph.nodes.invoke_model",
        lambda **_kwargs: SimpleNamespace(content="说明\n{\"selected_skill\": \"coder\"}\n结束"),
    )

    result = _skill_planning_node({"question": "帮我写一个函数"})
    assert result.get("selected_skill") == "coder"
