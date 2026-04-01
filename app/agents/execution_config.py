from __future__ import annotations

import os
from pathlib import Path

from app.core.config import get_env_bool, get_env_int


def docker_sandbox_enabled() -> bool:
    return get_env_bool("DOCKER_SANDBOX_ENABLED", default=False)


def docker_sandbox_image() -> str:
    raw = os.getenv("DOCKER_SANDBOX_IMAGE", "mcr.microsoft.com/devcontainers/python:3.11-bookworm").strip()
    return raw or "mcr.microsoft.com/devcontainers/python:3.11-bookworm"


def docker_exec_timeout_seconds(default: int = 30) -> int:
    return get_env_int("DOCKER_EXEC_TIMEOUT", default=default, min_value=5)


def docker_memory_limit() -> str:
    raw = os.getenv("DOCKER_MEMORY_LIMIT", "512m").strip()
    return raw or "512m"


def docker_cpu_limit() -> str:
    raw = os.getenv("DOCKER_CPU_LIMIT", "1").strip()
    return raw or "1"


def docker_pids_limit() -> int:
    return get_env_int("DOCKER_PIDS_LIMIT", default=128, min_value=16)


def docker_network_mode() -> str:
    raw = os.getenv("DOCKER_NETWORK_MODE", "bridge").strip().lower() or "bridge"
    if raw not in {"bridge", "none"}:
        raise ValueError("DOCKER_NETWORK_MODE must be 'bridge' or 'none'.")
    return raw


def docker_workdir_mount() -> Path | None:
    raw = os.getenv("DOCKER_SANDBOX_WORKDIR", "").strip()
    if not raw:
        return None

    path = Path(raw).expanduser()
    if not path.is_absolute():
        workspace_root = Path(__file__).resolve().parents[2]
        path = workspace_root / path
    return path.resolve()


def docker_max_command_chars() -> int:
    return get_env_int("DOCKER_MAX_COMMAND_CHARS", default=4000, min_value=200)
