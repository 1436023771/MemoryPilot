from __future__ import annotations

import subprocess

import app.agents.command_guard as command_guard
from app.agents.tools import clear_docker_exec_log, get_docker_exec_log, run_docker_command


def test_run_docker_command_requires_enable_flag(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_SANDBOX_ENABLED", "false")
    clear_docker_exec_log()

    result = run_docker_command.invoke({"command": "echo hi"})

    assert "沙箱未启用" in result
    logs = get_docker_exec_log()
    assert len(logs) == 1
    assert logs[0]["mode"] == "shell"


def test_run_docker_command_blocks_dangerous_shell(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_SANDBOX_ENABLED", "true")
    clear_docker_exec_log()
    monkeypatch.setattr(command_guard, "request_command_review", lambda *args, **kwargs: False)

    result = run_docker_command.invoke({"command": "mount -t tmpfs tmpfs /mnt"})

    assert "已拦截" in result
    logs = get_docker_exec_log()
    assert len(logs) == 1
    assert "mount" in logs[0]["command"]


def test_run_docker_command_allows_after_confirmation(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_SANDBOX_ENABLED", "true")
    clear_docker_exec_log()
    monkeypatch.setattr(command_guard, "request_command_review", lambda *args, **kwargs: True)

    def _fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="approved\n",
            stderr="",
        )

    monkeypatch.setattr("app.sandbox.docker_runner.subprocess.run", _fake_run)

    result = run_docker_command.invoke({"command": "mount -t tmpfs tmpfs /mnt"})

    assert "exit_code: 0" in result
    assert "approved" in result
    logs = get_docker_exec_log()
    assert len(logs) == 1
    assert logs[0]["mode"] == "shell"


def test_run_docker_command_success(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_SANDBOX_ENABLED", "true")
    clear_docker_exec_log()

    def _fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="ok\n",
            stderr="",
        )

    monkeypatch.setattr("app.sandbox.docker_runner.subprocess.run", _fake_run)

    result = run_docker_command.invoke({"command": "echo ok"})

    assert "exit_code: 0" in result
    assert "ok" in result
    logs = get_docker_exec_log()
    assert len(logs) == 1
    assert logs[0]["mode"] == "shell"


def test_run_docker_command_uses_mcp_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_MCP_ENABLED", "true")
    clear_docker_exec_log()
    monkeypatch.setattr(
        "app.agents.tools.tools_docker_sandbox.call_docker_command_via_mcp",
        lambda command, timeout_seconds: "exit_code: 0\nstdout:\nmcp-ok\n\nstderr:\n(empty)",
    )

    result = run_docker_command.invoke({"command": "echo from mcp"})

    assert "mcp-ok" in result
    logs = get_docker_exec_log()
    assert len(logs) == 1
    assert "echo from mcp" in logs[0]["command"]
