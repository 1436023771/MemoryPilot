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

# Import skills to trigger registration
from . import general, reading_companion

__all__ = [
    "Skill",
    "SkillConfig",
    "skill_registry",
    "SkillRegistry",
]
