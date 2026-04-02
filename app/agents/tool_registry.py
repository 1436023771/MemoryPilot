from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import app.agents.tools as tools_module


@dataclass(frozen=True)
class ToolRegistryEntry:
    name: str
    tool: object


class ToolRegistry:
    """Registry for LangChain tools exposed by app.agents.tools."""

    def __init__(self, module=tools_module):
        self._module = module
        self._tool_map = self._build_tool_map()

    def _build_tool_map(self) -> dict[str, object]:
        tool_map: dict[str, object] = {}
        exported_names = getattr(self._module, "__all__", [])
        for exported_name in exported_names:
            candidate = getattr(self._module, exported_name, None)
            tool_name = getattr(candidate, "name", None)
            if not tool_name or not hasattr(candidate, "invoke"):
                continue

            normalized_name = str(tool_name).strip() or exported_name
            tool_map[normalized_name] = candidate
        return tool_map

    def get_tool_names(self) -> list[str]:
        return list(self._tool_map.keys())

    def get_all_tools(self) -> list[object]:
        return list(self._tool_map.values())

    def get_tool(self, tool_name: str):
        return self._tool_map.get(tool_name)

    def get_tools(self, allowed_tool_names: Iterable[str] | None = None) -> list[object]:
        if allowed_tool_names is None:
            return self.get_all_tools()

        selected: list[object] = []
        seen: set[str] = set()
        for name in allowed_tool_names:
            normalized = str(name).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            tool = self._tool_map.get(normalized)
            if tool is not None:
                selected.append(tool)
        return selected

    def display_name(self, tool_name: str) -> str:
        defaults = {
            "run_python_code": "Python计算器",
            "run_docker_command": "Docker沙箱",
            "web_search": "联网搜索",
            "retrieve_pg_knowledge": "知识库检索",
        }
        return defaults.get(tool_name, tool_name)


@lru_cache(maxsize=1)
def get_tool_registry() -> ToolRegistry:
    return ToolRegistry()


tool_registry = get_tool_registry()
