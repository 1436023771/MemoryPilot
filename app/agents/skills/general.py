"""
General purpose skill for diverse tasks.

Default skill for coding, system administration, research, and general-purpose tool usage.
Allows all available tools and uses standard execution strategy.
"""

from .skill_base import Skill, SkillConfig
from .skill_registry import skill_registry


class GeneralSkill(Skill):
    """General-purpose skill for diverse tasks."""
    
    def __init__(self):
        """Initialize general skill."""
        config = SkillConfig(
            name="general",
            description=(
                "General-purpose execution: coding, command execution, research, "
                "web searches, and tool orchestration for diverse tasks."
            ),
            system_prompt_override="agents.langgraph.final_user_prompt",
            available_tools=[
                "web_search",
                "retrieve_pg_knowledge",
                "run_python_code",
                "run_docker_command",
                "translate_light_novel_ja_to_zh",
                "translate_light_novel_batch",
            ],
            tool_display_names={
                "web_search": "联网搜索",
                "retrieve_pg_knowledge": "知识库检索",
                "run_python_code": "Python计算器",
                "run_docker_command": "Docker沙箱",
                "translate_light_novel_ja_to_zh": "轻小说日译中",
                "translate_light_novel_batch": "轻小说批量翻译",
            },
        )
        super().__init__(config)


# Register skill globally
skill_registry.register(GeneralSkill())
