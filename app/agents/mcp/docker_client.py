from __future__ import annotations

import asyncio
import importlib
import os
import shlex
from typing import Any

from app.config.execution import docker_mcp_command, docker_mcp_timeout_seconds


def _extract_text_from_mcp_result(result: Any) -> str:
    content = getattr(result, "content", None)
    if not isinstance(content, list):
        return str(result)

    parts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
            continue
        if isinstance(item, dict):
            mapped = str(item.get("text", "")).strip()
            if mapped:
                parts.append(mapped)
    merged = "\n".join(parts).strip()
    return merged or str(result)


def _build_stdio_params():
    try:
        from mcp import StdioServerParameters
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("mcp package is not available") from exc

    raw_command = docker_mcp_command().strip()
    parts = shlex.split(raw_command)
    if not parts:
        raise RuntimeError("DOCKER_MCP_COMMAND is empty")

    return StdioServerParameters(
        command=parts[0],
        args=parts[1:],
        env=os.environ.copy(),
    )


async def _call_tool_async(tool_name: str, arguments: dict[str, Any]) -> str:
    try:
        mcp_module = importlib.import_module("mcp")
        stdio_module = importlib.import_module("mcp.client.stdio")
        ClientSession = getattr(mcp_module, "ClientSession")
        stdio_client = getattr(stdio_module, "stdio_client")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("mcp client modules are unavailable") from exc

    params = _build_stdio_params()
    timeout_seconds = docker_mcp_timeout_seconds()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments=arguments),
                timeout=timeout_seconds,
            )
            return _extract_text_from_mcp_result(result)


def _call_tool_sync(tool_name: str, arguments: dict[str, Any]) -> str:
    try:
        return asyncio.run(_call_tool_async(tool_name, arguments))
    except Exception as exc:  # noqa: BLE001
        return f"MCP 调用失败: {exc}"


def call_docker_command_via_mcp(command: str, timeout_seconds: int = 30) -> str:
    return _call_tool_sync(
        "run_docker_command",
        {"command": command, "timeout_seconds": int(timeout_seconds)},
    )


def call_python_in_docker_via_mcp(code: str, timeout_seconds: int = 30) -> str:
    return _call_tool_sync(
        "run_python_in_docker",
        {"code": code, "timeout_seconds": int(timeout_seconds)},
    )


__all__ = [
    "call_docker_command_via_mcp",
    "call_python_in_docker_via_mcp",
]
