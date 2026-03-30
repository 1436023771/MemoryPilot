"""
Python execution tool for precise calculation and logic verification.
"""

from __future__ import annotations

import re

from langchain.tools import tool
from app.agents.tools_docker_sandbox import run_python_in_docker

_MAX_PY_CODE_CHARS = 8000
_MAX_TOOL_OUTPUT_CHARS = 6000
_python_exec_log: list[dict] = []


def get_python_exec_log() -> list[dict]:
    """Return a copy of current turn Python execution log."""
    return _python_exec_log.copy()


def clear_python_exec_log() -> None:
    """Clear Python execution log at the start of each turn."""
    global _python_exec_log
    _python_exec_log = []


def record_python_exec(code: str, result: str) -> None:
    """Record executed code and tool result for GUI sidebar display."""
    global _python_exec_log
    _python_exec_log.append({"code": code, "result": result})


def _contains_dangerous_python(code: str) -> bool:
    """Block obvious high-risk operations."""
    deny_patterns = [
        r"\bimport\s+subprocess\b",
        r"\bfrom\s+subprocess\s+import\b",
        r"\bos\.system\s*\(",
        r"\bsubprocess\.",
        r"\bpty\.",
        r"\bimport\s+socket\b",
        r"\bfrom\s+socket\s+import\b",
        r"\bsocket\.",
        r"\bimport\s+requests\b",
        r"\bfrom\s+requests\s+import\b",
        r"\brequests\.",
        r"\bimport\s+httpx\b",
        r"\bfrom\s+httpx\s+import\b",
        r"\bhttpx\.",
        r"\burllib\.request\.",
        r"\bimport\s+shutil\b",
        r"\bfrom\s+shutil\s+import\b",
        r"\bshutil\.rmtree\s*\(",
    ]
    lowered = code.lower()
    return any(re.search(pattern, lowered) for pattern in deny_patterns)


@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return exit_code/stdout/stderr."""
    normalized = (code or "").strip()
    if not normalized:
        result = "Python 执行失败: 代码为空"
        record_python_exec(normalized, result)
        return result

    if len(normalized) > _MAX_PY_CODE_CHARS:
        result = f"Python 执行失败: 代码过长（>{_MAX_PY_CODE_CHARS} 字符）"
        record_python_exec(normalized, result)
        return result

    if _contains_dangerous_python(normalized):
        result = "Python 执行失败: 检测到高风险调用（系统命令/网络/进程操作），已拒绝执行"
        record_python_exec(normalized, result)
        return result

    result = run_python_in_docker(normalized)
    record_python_exec(normalized, result)
    return result


__all__ = ["run_python_code", "get_python_exec_log", "clear_python_exec_log", "record_python_exec"]
