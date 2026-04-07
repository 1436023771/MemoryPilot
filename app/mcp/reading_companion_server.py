from __future__ import annotations

import argparse

from app.agents.tools.tools_pg_knowledge import _retrieve_pg_knowledge_impl
from app.config.knowledge import pgvector_table

try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # noqa: BLE001
    FastMCP = None  # type: ignore[assignment]


if FastMCP is not None:
    mcp = FastMCP("reading-companion")

    @mcp.tool()
    def health_check() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "reading-companion",
            "knowledge_table": pgvector_table(),
        }

    @mcp.tool()
    def retrieve_reading_context(
        query: str,
        top_k: int = 0,
        book_id: str = "",
        chapter: str = "",
        context_window: int = -1,
        rerank_candidates: int = 0,
    ) -> str:
        """Retrieve reading-related context from pgvector knowledge base."""
        return _retrieve_pg_knowledge_impl(
            query=query,
            top_k=top_k,
            book_id=book_id,
            chapter=chapter,
            context_window=context_window,
            rerank_candidates=rerank_candidates,
        )


def main() -> None:
    if FastMCP is None:
        raise RuntimeError("mcp package is required. Install with: pip install mcp")

    parser = argparse.ArgumentParser(description="Reading companion MCP server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http", "sse"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--path", default="/mcp")
    args = parser.parse_args()

    try:
        mcp.settings.host = args.host
        mcp.settings.port = int(args.port)
        mcp.settings.streamable_http_path = args.path
    except Exception:  # noqa: BLE001
        pass

    if args.transport == "stdio":
        mcp.run()
        return

    try:
        mcp.run(transport=args.transport, mount_path=args.path)
    except TypeError:
        mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()