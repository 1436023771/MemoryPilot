from __future__ import annotations

import asyncio
import importlib
from typing import Any

from app.config.execution import reading_companion_mcp_server_url, reading_companion_mcp_timeout_seconds


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


async def _call_tool_via_streamable_http(tool_name: str, arguments: dict[str, Any]) -> str:
    try:
        mcp_module = importlib.import_module("mcp")
        ClientSession = getattr(mcp_module, "ClientSession")

        http_module = importlib.import_module("mcp.client.streamable_http")
        client_ctx_factory = getattr(http_module, "streamablehttp_client", None)
        if client_ctx_factory is None:
            client_ctx_factory = getattr(http_module, "streamable_http_client", None)
        if client_ctx_factory is None:
            raise RuntimeError("mcp streamable-http client module is unavailable")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("mcp client modules are unavailable") from exc

    timeout_seconds = reading_companion_mcp_timeout_seconds()
    server_url = reading_companion_mcp_server_url()

    async with client_ctx_factory(server_url) as streams:
        if not (isinstance(streams, tuple) and len(streams) >= 2):
            raise RuntimeError("Unexpected streamable-http client return value")

        read_stream = streams[0]
        write_stream = streams[1]
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments=arguments),
                timeout=timeout_seconds,
            )
            return _extract_text_from_mcp_result(result)


def _call_tool_sync(tool_name: str, arguments: dict[str, Any]) -> str:
    try:
        return asyncio.run(_call_tool_via_streamable_http(tool_name, arguments))
    except Exception as exc:  # noqa: BLE001
        return f"Reading Companion MCP 调用失败: {exc}"


def retrieve_reading_context_via_mcp(
    query: str,
    top_k: int = 0,
    book_id: str = "",
    chapter: str = "",
    context_window: int = -1,
    rerank_candidates: int = 0,
) -> str:
    return _call_tool_sync(
        "retrieve_reading_context",
        {
            "query": str(query or ""),
            "top_k": int(top_k),
            "book_id": str(book_id or ""),
            "chapter": str(chapter or ""),
            "context_window": int(context_window),
            "rerank_candidates": int(rerank_candidates),
        },
    )


__all__ = ["retrieve_reading_context_via_mcp"]