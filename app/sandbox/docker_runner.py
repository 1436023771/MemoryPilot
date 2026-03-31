from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile

from app.agents.execution_config import (
    docker_cpu_limit,
    docker_exec_timeout_seconds,
    docker_max_command_chars,
    docker_memory_limit,
    docker_network_mode,
    docker_pids_limit,
    docker_sandbox_enabled,
    docker_sandbox_image,
    docker_workdir_mount,
)

_MAX_TOOL_OUTPUT_CHARS = 6000


def _trim_text(text: str, max_chars: int = _MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _format_result(return_code: int, stdout: str, stderr: str) -> str:
    return (
        f"exit_code: {return_code}\n"
        f"stdout:\n{_trim_text(stdout.strip()) or '(empty)'}\n\n"
        f"stderr:\n{_trim_text(stderr.strip()) or '(empty)'}"
    )


def _contains_dangerous_shell(command: str) -> bool:
    deny_patterns = [
        r"\bmount\b",
        r"\bumount\b",
        r"\bmkfs\w*\b",
        r"\bdd\s+if=.*\sof=/dev/",
        r":\(\)\s*\{\s*:\|:\s*&\s*\};:",
    ]
    lowered = command.lower()
    return any(re.search(pattern, lowered) for pattern in deny_patterns)


def _docker_base_cmd(image: str, mount_dir: Path) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--cpus",
        docker_cpu_limit(),
        "--memory",
        docker_memory_limit(),
        "--pids-limit",
        str(docker_pids_limit()),
        "--network",
        docker_network_mode(),
        "-v",
        f"{mount_dir}:/workspace",
        "-w",
        "/workspace",
        image,
        "sh",
        "-lc",
    ]


def _prepare_mount_dir() -> Path:
    configured = docker_workdir_mount()
    if configured is not None:
        configured.mkdir(parents=True, exist_ok=True)
        return configured
    return Path(tempfile.mkdtemp(prefix="agent_docker_workspace_"))


def execute_docker_shell(command: str, timeout_seconds: int = 30) -> str:
    normalized = (command or "").strip()
    if not docker_sandbox_enabled():
        return "Docker 执行失败: DOCKER_SANDBOX_ENABLED=false，沙箱未启用"

    if not normalized:
        return "Docker 执行失败: 命令为空"

    max_chars = docker_max_command_chars()
    if len(normalized) > max_chars:
        return f"Docker 执行失败: 命令过长（>{max_chars} 字符）"

    if _contains_dangerous_shell(normalized):
        return "Docker 执行失败: 检测到被禁止的高危命令"

    timeout_value = max(5, int(timeout_seconds or docker_exec_timeout_seconds()))
    mount_dir = _prepare_mount_dir()
    image = docker_sandbox_image()
    cmd = _docker_base_cmd(image=image, mount_dir=mount_dir)
    cmd.append(normalized)

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_value + 8,
            check=False,
            env=os.environ.copy(),
        )
    except FileNotFoundError:
        return "Docker 执行失败: 未找到 docker 命令，请确认 Docker 已安装并在 PATH 中"
    except subprocess.TimeoutExpired as exc:
        partial_stdout = _trim_text((exc.stdout or "").strip())
        partial_stderr = _trim_text((exc.stderr or "").strip())
        return (
            f"Docker 执行超时: 超过 {timeout_value} 秒\n"
            f"stdout:\n{partial_stdout or '(empty)'}\n\n"
            f"stderr:\n{partial_stderr or '(empty)'}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"Docker 执行失败: {exc}"

    return _format_result(completed.returncode, completed.stdout or "", completed.stderr or "")


def execute_python_in_docker(code: str, timeout_seconds: int = 30) -> str:
    normalized = (code or "").strip()
    if not docker_sandbox_enabled():
        return "Python 执行失败: DOCKER_SANDBOX_ENABLED=false，沙箱未启用"

    if not normalized:
        return "Python 执行失败: 代码为空"

    timeout_value = max(5, int(timeout_seconds or docker_exec_timeout_seconds()))
    mount_dir = _prepare_mount_dir()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(normalized + "\n")
        script_path = Path(handle.name)

    image = docker_sandbox_image()
    cmd = [
        "docker",
        "run",
        "--rm",
        "--cpus",
        docker_cpu_limit(),
        "--memory",
        docker_memory_limit(),
        "--pids-limit",
        str(docker_pids_limit()),
        "--network",
        docker_network_mode(),
        "-v",
        f"{mount_dir}:/workspace",
        "-v",
        f"{script_path}:/sandbox/script.py:ro",
        "-w",
        "/workspace",
        image,
        "sh",
        "-lc",
        "python /sandbox/script.py",
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_value + 8,
            check=False,
        )
    except FileNotFoundError:
        return "Python 执行失败: 未找到 docker 命令，请确认 Docker 已安装并在 PATH 中"
    except subprocess.TimeoutExpired as exc:
        partial_stdout = _trim_text((exc.stdout or "").strip())
        partial_stderr = _trim_text((exc.stderr or "").strip())
        return (
            f"Python 执行超时: 超过 {timeout_value} 秒\n"
            f"stdout:\n{partial_stdout or '(empty)'}\n\n"
            f"stderr:\n{partial_stderr or '(empty)'}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"Python 执行失败: {exc}"
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    return _format_result(completed.returncode, completed.stdout or "", completed.stderr or "")
