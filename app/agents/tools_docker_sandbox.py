"""Docker sandbox tool wrappers backed by reusable sandbox runner."""

from __future__ import annotations

from langchain.tools import tool
from app.agents.docker_mcp_client import call_docker_command_via_mcp, call_python_in_docker_via_mcp
from app.agents.execution_config import docker_mcp_enabled
from app.sandbox.docker_runner import execute_docker_shell, execute_python_in_docker

_docker_exec_log: list[dict] = []


def get_docker_exec_log() -> list[dict]:
    return _docker_exec_log.copy()


def clear_docker_exec_log() -> None:
    global _docker_exec_log
    _docker_exec_log = []


def record_docker_exec(mode: str, command: str, result: str) -> None:
    global _docker_exec_log
    _docker_exec_log.append({"mode": mode, "command": command, "result": result})


@tool
def run_docker_command(command: str, timeout_seconds: int = 30) -> str:
    """在 Docker 沙箱中执行 shell 命令（高风险操作优先使用）。

    Args:
        command: 在容器内执行的命令，例如 `ls -la`、`find . -name '*.log'`、`curl -I https://example.com`。
        timeout_seconds: 最大执行时长（秒）。

    Returns:
        统一格式的执行结果：exit_code/stdout/stderr。

    Use when:
        - 需要系统命令、文件批处理、下载、环境探测等操作。
        - 不适合只用文本推理完成的问题。

    Notes:
        - 该工具受资源/安全限制；若沙箱未开启会返回明确错误。
    """
    return _run_docker_command_impl(command, timeout_seconds)


def _run_docker_command_impl(command: str, timeout_seconds: int = 30) -> str:
    """Internal implementation of run_docker_command."""
    normalized = (command or "").strip()
    if docker_mcp_enabled():
        result = call_docker_command_via_mcp(command=normalized, timeout_seconds=timeout_seconds)
    else:
        result = execute_docker_shell(command=normalized, timeout_seconds=timeout_seconds)
    record_docker_exec("shell", normalized, result)
    return result


@tool
def run_python_in_docker(code: str, timeout_seconds: int = 30) -> str:
    """在 Docker 沙箱中执行 Python 代码进行精确计算与逻辑验证。

    Args:
        code: 在容器内执行的 Python 代码片段。
        timeout_seconds: 最大执行时长（秒）。

    Returns:
        统一格式的执行结果：exit_code/stdout/stderr。

    Use when:
        - 需要数值计算、数据处理、库函数验证。
        - 需要调用 subprocess、requests、数据库、外部 API。
        - 文本推理无法解决的问题。

    Notes:
        - 代码受危险模式检查保护（仅阻止根目录删除等破坏操作）。
        - subprocess/requests/socket 等已允许（运行在 Docker 容器内部）。
    """
    return _run_python_in_docker_impl(code, timeout_seconds)


def _run_python_in_docker_impl(code: str, timeout_seconds: int = 30) -> str:
    """Internal implementation of run_python_in_docker."""
    normalized = (code or "").strip()
    if docker_mcp_enabled():
        result = call_python_in_docker_via_mcp(code=normalized, timeout_seconds=timeout_seconds)
    else:
        result = execute_python_in_docker(code=normalized, timeout_seconds=timeout_seconds)
    record_docker_exec("python", normalized, result)
    return result


__all__ = [
    "run_docker_command",
    "run_python_in_docker",
    "_run_docker_command_impl",
    "_run_python_in_docker_impl",
    "get_docker_exec_log",
    "clear_docker_exec_log",
    "record_docker_exec",
]
