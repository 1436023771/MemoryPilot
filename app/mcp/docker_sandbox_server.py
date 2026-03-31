from __future__ import annotations

from app.sandbox.docker_runner import execute_docker_shell, execute_python_in_docker

try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # noqa: BLE001
    FastMCP = None  # type: ignore[assignment]


if FastMCP is not None:
    mcp = FastMCP("docker-sandbox")

    @mcp.tool()
    def run_docker_command(command: str, timeout_seconds: int = 30) -> str:
        """Run shell command in docker sandbox."""
        return execute_docker_shell(command=command, timeout_seconds=timeout_seconds)

    @mcp.tool()
    def run_python_in_docker(code: str, timeout_seconds: int = 30) -> str:
        """Run python code in docker sandbox."""
        return execute_python_in_docker(code=code, timeout_seconds=timeout_seconds)


def main() -> None:
    if FastMCP is None:
        raise RuntimeError("mcp package is required. Install with: pip install mcp")
    mcp.run()


if __name__ == "__main__":
    main()
