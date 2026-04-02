"""
Base skill class and configuration for agent skill system.

Each skill represents a specialized behavior/capability for the agent.
Skills can define:
- System prompt override
- Allowed tools (tool filtering)
- Custom tool display names
- Context preprocessing logic
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class SkillConfig:
    """Skill configuration object."""
    
    name: str
    """Unique identifier for the skill (e.g., 'reading-companion', 'general')."""
    
    description: str
    """Human-readable description (used for LLM skill selection)."""
    
    system_prompt_override: Optional[str] = None
    """Custom system prompt template key (e.g., 'agents.langgraph.reading_companion_prompt').
    If None, uses default prompt."""
    
    available_tools: Optional[List[str]] = None
    """List of allowed tool names. If None, all tools are available.
    Example: ['retrieve_pg_knowledge', 'run_python_code', 'run_docker_command']"""
    
    tool_display_names: Dict[str, str] = field(default_factory=dict)
    """Custom display names for tools.
    Example: {'retrieve_pg_knowledge': '书籍检索'}"""
    
    context_processor: Optional[Callable[[str, str], Dict[str, str]]] = None
    """Optional function to preprocess memory_ctx and knowledge_ctx.
    Signature: (memory_ctx: str, knowledge_ctx: str) -> Dict[str, str]
    Returns dict with keys 'memory_ctx' and 'knowledge_ctx'."""


class Skill(ABC):
    """Base class for all agent skills."""
    
    def __init__(self, config: SkillConfig):
        """Initialize skill with configuration.
        
        Args:
            config: SkillConfig object defining the skill's properties.
        """
        self.config = config
    
    def should_activate(self, question: str) -> bool:
        """Optional method: whether this skill should be activated based on the question.
        
        This is a fallback mechanism for keyword-based detection.
        In the main flow, LLM decision takes precedence.
        
        Args:
            question: User's question.
            
        Returns:
            True if skill should be activated.
        """
        return False
    
    def process_context(
        self, 
        memory_ctx: str, 
        knowledge_ctx: str
    ) -> Dict[str, str]:
        """Preprocess memory and knowledge context for this skill.
        
        This allows skills to modify or reorder context before prompt rendering.
        
        Args:
            memory_ctx: Retrieved long-term memory context.
            knowledge_ctx: Retrieved knowledge base context.
            
        Returns:
            Dict with keys 'memory_ctx' and 'knowledge_ctx' (modified).
        """
        if self.config.context_processor:
            return self.config.context_processor(memory_ctx, knowledge_ctx)
        return {"memory_ctx": memory_ctx, "knowledge_ctx": knowledge_ctx}
    
    def get_tool_config(self) -> Dict[str, str]:
        """Get tool display configuration for this skill.
        
        Returns:
            Dict mapping tool names to display names.
        """
        return self.config.tool_display_names
