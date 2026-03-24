from app.agents.tools import clear_python_exec_log, get_python_exec_log, run_python_code


def test_run_python_code_success() -> None:
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
