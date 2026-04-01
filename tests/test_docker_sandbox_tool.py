from __future__ import annotations

import subprocess

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

    result = run_docker_command.invoke({"command": "mount -t tmpfs tmpfs /mnt"})

    assert "被禁止" in result
    logs = get_docker_exec_log()
    assert len(logs) == 1
    assert "mount" in logs[0]["command"]


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

    monkeypatch.setattr("app.agents.tools_docker_sandbox.subprocess.run", _fake_run)

    result = run_docker_command.invoke({"command": "echo ok"})

    assert "exit_code: 0" in result
    assert "ok" in result
    logs = get_docker_exec_log()
    assert len(logs) == 1
    assert logs[0]["mode"] == "shell"
