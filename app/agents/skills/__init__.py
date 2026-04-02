"""
Agent skill system: framework for specialized agent behaviors.

Skills define:
- Custom prompts
- Tool restrictions
- Context preprocessing
- Custom tool display names

Available skills are dynamically selected at runtime based on LLM decision.
"""

from .skill_base import Skill, SkillConfig
from .skill_registry import skill_registry, SkillRegistry
from .runtime import (
    DEFAULT_SKILL_NAME,
    process_context_for_skill,
    resolve_prompt_key,
    resolve_skill,
    resolve_skill_name,
    resolve_tool_display_name,
    select_tools_for_skill,
)

# Import skills to trigger registration
from . import general, reading_companion

__all__ = [
    "Skill",
    "SkillConfig",
    "skill_registry",
    "SkillRegistry",
    "DEFAULT_SKILL_NAME",
    "resolve_skill",
    "resolve_skill_name",
    "resolve_prompt_key",
    "resolve_tool_display_name",
    "select_tools_for_skill",
    "process_context_for_skill",
]
