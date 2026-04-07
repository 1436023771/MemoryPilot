import pytest

import app.agents.command_guard as command_guard
import app.agents.tools.tools_python_exec as py_exec_module
from app.agents.tools import clear_python_exec_log, get_python_exec_log, run_python_code


def test_run_python_code_success(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_SANDBOX_ENABLED", "true")
    monkeypatch.setattr(
        py_exec_module,
        "_run_python_in_docker_impl",
        lambda code: "exit_code: 0\nstdout:\n5\n\nstderr:\n(empty)",
    )

    clear_python_exec_log()
    result = run_python_code.invoke({"code": "print(2 + 3)"})

    assert "exit_code: 0" in result
    assert "5" in result

    logs = get_python_exec_log()
    assert len(logs) == 1
    assert "print(2 + 3)" in logs[0]["code"]
    assert "exit_code: 0" in logs[0]["result"]


def test_run_python_code_blocks_destructive_calls() -> None:
    clear_python_exec_log()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(command_guard, "request_command_review", lambda *args, **kwargs: False)
    result = run_python_code.invoke({"code": "import shutil\nshutil.rmtree('/')"})
    monkeypatch.undo()

    assert "已拦截" in result

    logs = get_python_exec_log()
    assert len(logs) == 1
    assert "shutil.rmtree('/')" in logs[0]["code"]
    assert "已拦截" in logs[0]["result"]


def test_run_python_code_allows_after_confirmation(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_SANDBOX_ENABLED", "true")
    monkeypatch.setattr(command_guard, "request_command_review", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        py_exec_module,
        "_run_python_in_docker_impl",
        lambda code: "exit_code: 0\nstdout:\nallowed\n\nstderr:\n(empty)",
    )

    clear_python_exec_log()
    result = run_python_code.invoke({"code": "import shutil\nshutil.rmtree('/')"})

    assert "exit_code: 0" in result
    assert "allowed" in result


def test_run_python_code_allows_subprocess_in_sandbox(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_SANDBOX_ENABLED", "true")
    monkeypatch.setattr(
        py_exec_module,
        "_run_python_in_docker_impl",
        lambda code: "exit_code: 0\nstdout:\nsubprocess-ok\n\nstderr:\n(empty)",
    )

    clear_python_exec_log()
    result = run_python_code.invoke({"code": "import subprocess\nprint('ok')"})

    assert "exit_code: 0" in result
    assert "subprocess-ok" in result


def test_run_python_code_requires_docker_enabled(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_SANDBOX_ENABLED", "false")
    clear_python_exec_log()

    result = run_python_code.invoke({"code": "print('hello')"})

    assert "沙箱未启用" in result
