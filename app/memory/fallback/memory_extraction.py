from __future__ import annotations

import json
import re
from typing import Callable

from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.prompt_store import render_prompt

try:
    from langsmith import tracing_context
except Exception:  # noqa: BLE001
    tracing_context = None

FactTuple = tuple[str, str, str]


def extract_structured_facts_regex(
    user_text: str,
    format_fact: Callable[[str, str], str],
    normalize_text: Callable[[str], str],
) -> list[FactTuple]:
    """Legacy regex extraction kept as fallback path."""
    patterns: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"我叫\s*([^，。,.!?！？\s]+)"), "name"),
        (re.compile(r"我是\s*([^，。,.!?！？\s]+)"), "name"),
        (re.compile(r"我喜欢\s*([^。！？!?]+)"), "like"),
        (re.compile(r"我偏好\s*([^。！？!?]+)"), "like"),
        (re.compile(r"我不喜欢\s*([^。！？!?]+)"), "dislike"),
        (re.compile(r"我的目标是\s*([^。！？!?]+)"), "goal"),
    ]

    extracted: list[FactTuple] = []
    for pattern, key in patterns:
        for match in pattern.finditer(user_text):
            value = normalize_text(match.group(1))
            if not value:
                continue
            value = value.rstrip("。.!！?？").strip()
            if not value:
                continue
            extracted.append((key, value, format_fact(key, value)))

    seen: set[tuple[str, str]] = set()
    deduped: list[FactTuple] = []
    for key, value, text in extracted:
        identity = (key, value)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append((key, value, text))
    return deduped


def extract_candidate_facts_single_turn(
    user_text: str,
    format_fact: Callable[[str, str], str],
    normalize_text: Callable[[str], str],
) -> list[FactTuple]:
    """Legacy single-turn LLM extraction kept as fallback path."""
    try:
        settings = get_settings()
        model = ChatOpenAI(
            model=settings.model_name,
            temperature=0.1,
            api_key=settings.api_key,
            base_url=settings.base_url,
        )

        prompt = render_prompt("memory.single_turn_extraction", user_text=user_text)

        if tracing_context is not None:
            with tracing_context(enabled=False):
                response = model.invoke(prompt)
        else:
            response = model.invoke(prompt)
        if hasattr(response, "content"):
            json_str = str(response.content).strip()
        else:
            json_str = str(response).strip()

        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        data = json.loads(json_str)
        if not isinstance(data, list):
            data = []

        key_mapping = {
            "name": "name",
            "user_name": "name",
            "like": "like",
            "preference": "like",
            "dislike": "dislike",
            "goal": "goal",
            "objective": "goal",
            "skill": "skill",
            "background": "background",
        }

        extracted: list[FactTuple] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip().lower()
            value = str(item.get("value", "")).strip()
            if not key or not value:
                continue
            normalized_key = key_mapping.get(key, key)
            extracted.append((normalized_key, value, format_fact(normalized_key, value)))

        seen: set[tuple[str, str]] = set()
        deduped: list[FactTuple] = []
        for key, value, text in extracted:
            identity = (key, value)
            if identity in seen:
                continue
            seen.add(identity)
            deduped.append((key, value, text))
        return deduped
    except (json.JSONDecodeError, ValueError, TypeError, Exception) as e:
        print(f"[Warning] Fallback single-turn extraction failed ({e}), falling back to regex extraction")
        return extract_structured_facts_regex(user_text, format_fact, normalize_text)
