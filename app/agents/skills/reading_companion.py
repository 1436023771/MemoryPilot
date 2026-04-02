"""
Reading companion skill for book/narrative analysis.

Specialized for questions about books, chapters, characters, timelines, and narrative structure.
"""

from .skill_base import Skill, SkillConfig
from .skill_registry import skill_registry


# Reading companion keywords for fallback keyword-based detection
READING_KEYWORDS = [
    "书",
    "章节",
    "chapter",
    "剧情",
    "plot",
    "角色",
    "character",
    "人物",
    "设定",
    "timeline",
    "时间线",
    "time line",
    "narrative",
    "叙事",
]


class ReadingCompanionSkill(Skill):
    """Skill for book reading companionship and narrative analysis."""
    
    def __init__(self):
        """Initialize reading companion skill."""
        config = SkillConfig(
            name="reading-companion",
            description=(
                "Book analysis and reading companionship: chapters, characters, timelines, "
                "plot structure, narrative analysis, and book-related context retrieval."
            ),
            system_prompt_override="agents.langgraph.reading_companion_prompt",
            available_tools=[
                "retrieve_pg_knowledge",  # Book/knowledge base retrieval
                "run_python_code",        # Data processing for narrative analysis
                "run_docker_command",     # Advanced processing if needed
                # Note: web_search is excluded to prioritize internal knowledge base
            ],
            tool_display_names={
                "retrieve_pg_knowledge": "书籍检索",
                "run_python_code": "数据处理",
                "run_docker_command": "高级计算",
            },
        )
        super().__init__(config)
    
    def should_activate(self, question: str) -> bool:
        """Check if question mentions reading-related keywords (fallback detection).
        
        Args:
            question: User's question.
            
        Returns:
            True if question contains reading keywords.
        """
        q = (question or "").lower()
        return any(keyword in q for keyword in READING_KEYWORDS)


# Register skill globally
skill_registry.register(ReadingCompanionSkill())
