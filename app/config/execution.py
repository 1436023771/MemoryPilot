from __future__ import annotations

from pathlib import Path

from app.config import get_env_bool, get_env_int, get_env_str


def docker_sandbox_enabled() -> bool:
    return get_env_bool("DOCKER_SANDBOX_ENABLED", default=False)


def docker_sandbox_image() -> str:
    raw = get_env_str("DOCKER_SANDBOX_IMAGE", "mcr.microsoft.com/devcontainers/python:3.11-bookworm")
    return raw or "mcr.microsoft.com/devcontainers/python:3.11-bookworm"


def docker_exec_timeout_seconds(default: int = 30) -> int:
    return get_env_int("DOCKER_EXEC_TIMEOUT", default=default, min_value=5)


def docker_memory_limit() -> str:
    raw = get_env_str("DOCKER_MEMORY_LIMIT", "512m")
    return raw or "512m"


def docker_cpu_limit() -> str:
    raw = get_env_str("DOCKER_CPU_LIMIT", "1")
    return raw or "1"


def docker_pids_limit() -> int:
    return get_env_int("DOCKER_PIDS_LIMIT", default=128, min_value=16)


def docker_network_mode() -> str:
    raw = get_env_str("DOCKER_NETWORK_MODE", "bridge", lower=True) or "bridge"
    if raw not in {"bridge", "none"}:
        raise ValueError("DOCKER_NETWORK_MODE must be 'bridge' or 'none'.")
    return raw


def docker_workdir_mount() -> Path | None:
    raw = get_env_str("DOCKER_SANDBOX_WORKDIR", "")
    if not raw:
        return None

    path = Path(raw).expanduser()
    if not path.is_absolute():
        workspace_root = Path(__file__).resolve().parents[2]
        path = workspace_root / path
    return path.resolve()


def docker_max_command_chars() -> int:
    return get_env_int("DOCKER_MAX_COMMAND_CHARS", default=4000, min_value=200)


def docker_mcp_enabled() -> bool:
    return get_env_bool("DOCKER_MCP_ENABLED", default=False)


def docker_mcp_command() -> str:
    raw = get_env_str("DOCKER_MCP_COMMAND", "python -m app.mcp.docker_sandbox_server")
    return raw or "python -m app.mcp.docker_sandbox_server"


def docker_mcp_timeout_seconds() -> int:
    return get_env_int("DOCKER_MCP_TIMEOUT", default=30, min_value=5)


def llmlingua_mcp_enabled() -> bool:
    return get_env_bool("LLMLINGUA_MCP_ENABLED", default=True)


def llmlingua_mcp_server_url() -> str:
    raw = get_env_str("LLMLINGUA_MCP_SERVER_URL", "http://127.0.0.1:8765/mcp")
    return raw or "http://127.0.0.1:8765/mcp"


def llmlingua_mcp_timeout_seconds() -> int:
    return get_env_int("LLMLINGUA_MCP_TIMEOUT", default=30, min_value=5)


def llmlingua_model_name() -> str:
    raw = get_env_str("LLMLINGUA_MODEL_NAME", "Qwen/Qwen2-1.5B-Instruct")
    return raw or "Qwen/Qwen2-1.5B-Instruct"


def llmlingua_model_path() -> Path:
    raw = get_env_str("LLMLINGUA_MODEL_PATH", "./models/qwen-1.5b") or "./models/qwen-1.5b"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        workspace_root = Path(__file__).resolve().parents[2]
        path = workspace_root / path
    return path.resolve()


def llmlingua_shared_server() -> bool:
    return get_env_bool("LLMLINGUA_SHARED_SERVER", default=True)


def reading_companion_mcp_enabled() -> bool:
    return get_env_bool("READING_COMPANION_MCP_ENABLED", default=False)


def reading_companion_mcp_server_url() -> str:
    raw = get_env_str("READING_COMPANION_MCP_SERVER_URL", "http://127.0.0.1:8767/mcp")
    return raw or "http://127.0.0.1:8767/mcp"


def reading_companion_mcp_timeout_seconds() -> int:
    return get_env_int("READING_COMPANION_MCP_TIMEOUT", default=30, min_value=5)
