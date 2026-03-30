import app.agents.tools_python_exec as py_exec_module
from app.agents.tools import clear_python_exec_log, get_python_exec_log, run_python_code


def test_run_python_code_success(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_SANDBOX_ENABLED", "true")
    monkeypatch.setattr(
        py_exec_module,
        "run_python_in_docker",
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


def test_run_python_code_blocks_dangerous_calls() -> None:
    clear_python_exec_log()
    result = run_python_code.invoke({"code": "import subprocess\nprint('x')"})

    assert "已拒绝执行" in result

    logs = get_python_exec_log()
    assert len(logs) == 1
    assert "import subprocess" in logs[0]["code"]
    assert "已拒绝执行" in logs[0]["result"]


def test_run_python_code_requires_docker_enabled(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_SANDBOX_ENABLED", "false")
    clear_python_exec_log()

    result = run_python_code.invoke({"code": "print('hello')"})

    assert "沙箱未启用" in result
