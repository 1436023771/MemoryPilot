from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import yaml

_PROMPT_FILE = Path(__file__).with_name("prompts.yaml")
_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
_PROMPT_CACHE: dict[str, str] | None = None


def _load_prompt_map() -> dict[str, str]:
    global _PROMPT_CACHE
    if _PROMPT_CACHE is not None:
        return _PROMPT_CACHE

    if _PROMPT_FILE.exists():
        raw = yaml.safe_load(_PROMPT_FILE.read_text(encoding="utf-8"))
    else:
        fallback_file = Path(__file__).with_name("prompts.json")
        raw = json.loads(fallback_file.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("prompt file must be an object of key/value strings")

    prompt_map: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("prompt keys and values must both be strings")
        prompt_map[key] = value

    _PROMPT_CACHE = prompt_map
    return prompt_map


def get_prompt_template(key: str) -> str:
    prompts = _load_prompt_map()
    if key not in prompts:
        raise KeyError(f"Prompt key not found: {key}")
    return prompts[key]


def render_prompt(key: str, **variables: Any) -> str:
    template = get_prompt_template(key)

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            raise KeyError(f"Missing prompt variable '{name}' for key '{key}'")
        return str(variables[name])

    return _PLACEHOLDER_PATTERN.sub(_replace, template)
