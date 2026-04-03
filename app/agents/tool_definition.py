from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., str]

    def invoke(self, arguments: dict[str, Any] | None = None) -> str:
        args = arguments or {}
        if not isinstance(args, dict):
            raise ValueError(f"Tool {self.name} expects a dict input.")
        return str(self.handler(**args))

    def __call__(self, *args, **kwargs):
        return self.handler(*args, **kwargs)

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


def parse_tool_arguments(raw_args: Any) -> dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        text = raw_args.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Tool arguments must be valid JSON object string.") from exc
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Tool arguments JSON must decode to an object.")

    raise ValueError("Unsupported tool arguments type.")
