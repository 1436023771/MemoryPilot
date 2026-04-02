"""Runtime helpers for skill-aware orchestration."""

from __future__ import annotations

from app.agents.tool_registry import tool_registry
from app.agents.skills.skill_base import Skill
from app.agents.skills.skill_registry import skill_registry


DEFAULT_SKILL_NAME = "general"


def resolve_skill(skill_name: str | None) -> Skill | None:
    normalized = str(skill_name or "").strip()
    if normalized:
        selected = skill_registry.get_skill(normalized)
        if selected:
            return selected
    return skill_registry.get_skill(DEFAULT_SKILL_NAME)


def resolve_skill_name(skill_name: str | None) -> str:
    skill = resolve_skill(skill_name)
    if skill is None:
        return DEFAULT_SKILL_NAME
    return skill.config.name


def select_tools_for_skill(skill_name: str | None) -> list[object]:
    skill = resolve_skill(skill_name)
    if skill and skill.config.available_tools:
        return tool_registry.get_tools(skill.config.available_tools)
    return tool_registry.get_all_tools()


def resolve_tool_display_name(skill_name: str | None, tool_name: str) -> str:
    skill = resolve_skill(skill_name)
    if skill and skill.config.tool_display_names:
        custom_name = skill.config.tool_display_names.get(tool_name)
        if custom_name:
            return custom_name
    return tool_registry.display_name(tool_name)


def resolve_prompt_key(skill_name: str | None, fallback_prompt_key: str) -> str:
    skill = resolve_skill(skill_name)
    if skill and skill.config.system_prompt_override:
        return skill.config.system_prompt_override
    return fallback_prompt_key


def process_context_for_skill(
    skill_name: str | None,
    memory_ctx: str,
    knowledge_ctx: str,
) -> tuple[str, str]:
    skill = resolve_skill(skill_name)
    if not skill:
        return memory_ctx, knowledge_ctx

    processed = skill.process_context(memory_ctx, knowledge_ctx)
    return processed.get("memory_ctx", memory_ctx), processed.get("knowledge_ctx", knowledge_ctx)
