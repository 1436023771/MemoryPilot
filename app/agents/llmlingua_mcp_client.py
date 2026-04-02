from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any

from app.agents.execution_config import llmlingua_mcp_server_url, llmlingua_mcp_timeout_seconds


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


def _safe_json_loads(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:  # noqa: BLE001
        return None
    if raw.startswith("```") and raw.endswith("```"):
        body = raw.strip("`").strip()
        if body.lower().startswith("json"):
            body = body[4:].strip()
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001
            return None
    return None


async def _call_tool_via_streamable_http(tool_name: str, arguments: dict[str, Any]) -> str:
    try:
        mcp_module = importlib.import_module("mcp")
        ClientSession = getattr(mcp_module, "ClientSession")

        http_module = None
        client_ctx_factory = None

        try:
            http_module = importlib.import_module("mcp.client.streamable_http")
            # Prefer the newer API name first; it is more stable in mcp>=1.x.
            client_ctx_factory = getattr(http_module, "streamablehttp_client", None)
            if client_ctx_factory is None:
                client_ctx_factory = getattr(http_module, "streamable_http_client", None)
        except Exception:  # noqa: BLE001
            http_module = None

        if client_ctx_factory is None:
            raise RuntimeError("mcp streamable-http client module is unavailable")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("mcp client modules are unavailable") from exc

    timeout_seconds = llmlingua_mcp_timeout_seconds()
    server_url = llmlingua_mcp_server_url()

    async with client_ctx_factory(server_url) as streams:
        if isinstance(streams, tuple) and len(streams) >= 2:
            read_stream = streams[0]
            write_stream = streams[1]
        else:
            raise RuntimeError("Unexpected streamable-http client return value")

        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments=arguments),
                timeout=timeout_seconds,
            )
            return _extract_text_from_mcp_result(result)


async def _call_tool_async(tool_name: str, arguments: dict[str, Any]) -> str:
    return await _call_tool_via_streamable_http(tool_name, arguments)


def _call_tool_sync(tool_name: str, arguments: dict[str, Any]) -> str:
    try:
        return asyncio.run(_call_tool_async(tool_name, arguments))
    except Exception as exc:  # noqa: BLE001
        return f"LLMLingua MCP 调用失败: {exc}"


def health_check_via_llmlingua_mcp() -> dict[str, Any] | None:
    raw = _call_tool_sync("health_check", {})
    parsed = _safe_json_loads(raw)
    return parsed


def warmup_model_via_llmlingua_mcp() -> dict[str, Any] | None:
    raw = _call_tool_sync("warmup_model", {})
    parsed = _safe_json_loads(raw)
    return parsed


def compress_text_via_llmlingua_mcp(
    text: str,
    target_tokens: int,
    preserve_keywords: list[str] | None = None,
) -> str | None:
    raw = _call_tool_sync(
        "compress_prompt",
        {
            "text": str(text or ""),
            "target_tokens": int(target_tokens),
            "preserve_keywords": list(preserve_keywords or []),
        },
    )

    payload = _safe_json_loads(raw)
    if payload and isinstance(payload.get("compressed_text"), str):
        return str(payload["compressed_text"])

    # mcp result sometimes returns plain text payload.
    if raw and not raw.startswith("LLMLingua MCP 调用失败"):
        return raw
    return None


def compress_history_via_llmlingua_mcp(
    messages: list[dict[str, str]],
    target_tokens: int,
) -> list[dict[str, str]] | None:
    raw = _call_tool_sync(
        "compress_history",
        {
            "messages": list(messages or []),
            "target_tokens": int(target_tokens),
        },
    )
    payload = _safe_json_loads(raw)
    if not payload:
        return None

    compressed = payload.get("compressed_messages")
    if not isinstance(compressed, list):
        return None

    out: list[dict[str, str]] = []
    for item in compressed:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user") or "user")
        content = str(item.get("content", "") or "")
        out.append({"role": role, "content": content})
    return out


__all__ = [
    "health_check_via_llmlingua_mcp",
    "warmup_model_via_llmlingua_mcp",
    "compress_text_via_llmlingua_mcp",
    "compress_history_via_llmlingua_mcp",
]
