from __future__ import annotations

from pathlib import Path

from app.config.execution import (
    llmlingua_mcp_enabled,
    llmlingua_mcp_server_url,
    llmlingua_mcp_timeout_seconds,
    llmlingua_model_name,
    llmlingua_model_path,
    llmlingua_shared_server,
)


def test_llmlingua_defaults(monkeypatch) -> None:
    # Use empty values (instead of delenv) so load_dotenv() will not repopulate
    # these keys from a local .env during test execution.
    monkeypatch.setenv("LLMLINGUA_MCP_ENABLED", "")
    monkeypatch.setenv("LLMLINGUA_MCP_SERVER_URL", "")
    monkeypatch.setenv("LLMLINGUA_MCP_TIMEOUT", "")
    monkeypatch.setenv("LLMLINGUA_MODEL_NAME", "")
    monkeypatch.setenv("LLMLINGUA_MODEL_PATH", "")
    monkeypatch.setenv("LLMLINGUA_SHARED_SERVER", "")

    assert llmlingua_mcp_enabled() is True
    assert llmlingua_mcp_server_url() == "http://127.0.0.1:8765/mcp"
    assert llmlingua_mcp_timeout_seconds() == 30
    assert llmlingua_model_name() == "Qwen/Qwen2-1.5B-Instruct"
    assert llmlingua_shared_server() is True

    path = llmlingua_model_path()
    assert path.is_absolute()
    assert path.as_posix().endswith("models/qwen-1.5b")


def test_llmlingua_path_resolve_relative(monkeypatch) -> None:
    monkeypatch.setenv("LLMLINGUA_MODEL_PATH", "./models/custom-qwen")
    path = llmlingua_model_path()
    assert path.is_absolute()
    assert path.as_posix().endswith("models/custom-qwen")


def test_llmlingua_path_resolve_absolute(monkeypatch, tmp_path: Path) -> None:
    custom = tmp_path / "qwen"
    monkeypatch.setenv("LLMLINGUA_MODEL_PATH", str(custom))
    path = llmlingua_model_path()
    assert path == custom.resolve()
