from __future__ import annotations

from app.config.execution import (
    reading_companion_mcp_enabled,
    reading_companion_mcp_server_url,
    reading_companion_mcp_timeout_seconds,
)


def test_reading_companion_mcp_defaults(monkeypatch) -> None:
    monkeypatch.setenv("READING_COMPANION_MCP_ENABLED", "")
    monkeypatch.setenv("READING_COMPANION_MCP_SERVER_URL", "")
    monkeypatch.setenv("READING_COMPANION_MCP_TIMEOUT", "")

    assert reading_companion_mcp_enabled() is False
    assert reading_companion_mcp_server_url() == "http://127.0.0.1:8767/mcp"
    assert reading_companion_mcp_timeout_seconds() == 30


def test_reading_companion_mcp_custom_values(monkeypatch) -> None:
    monkeypatch.setenv("READING_COMPANION_MCP_ENABLED", "true")
    monkeypatch.setenv("READING_COMPANION_MCP_SERVER_URL", "http://127.0.0.1:9001/mcp")
    monkeypatch.setenv("READING_COMPANION_MCP_TIMEOUT", "45")

    assert reading_companion_mcp_enabled() is True
    assert reading_companion_mcp_server_url() == "http://127.0.0.1:9001/mcp"
    assert reading_companion_mcp_timeout_seconds() == 45