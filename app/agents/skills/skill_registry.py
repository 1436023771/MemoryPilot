"""
Global skill registry for agent skill system.

Skills are registered at module initialization and accessed via the global registry.
"""

from typing import Dict, List, Optional

from .skill_base import Skill


class SkillRegistry:
    """Central registry for all agent skills."""
    
    _skills: Dict[str, Skill] = {}
    _initialized: bool = False
    
    @classmethod
    def register(cls, skill: Skill) -> Skill:
        """Register a skill with the registry.
        
        Args:
            skill: Skill instance to register.
            
        Returns:
            The registered skill (for decorator usage).
        """
        cls._skills[skill.config.name] = skill
        return skill
    
    @classmethod
    def get_skill(cls, name: str) -> Optional[Skill]:
        """Retrieve a skill by name.
        
        Args:
            name: Skill name (e.g., 'reading-companion').
            
        Returns:
            Skill instance, or None if not found.
        """
        return cls._skills.get(name)
    
    @classmethod
    def list_skills(cls) -> List[str]:
        """Get list of all registered skill names.
        
        Returns:
            Sorted list of skill names.
        """
        return sorted(cls._skills.keys())
    
    @classmethod
    def list_descriptions(cls) -> str:
        """Generate a formatted string of skill descriptions for LLM.
        
        Format: "- skill_name: skill_description"
        Used to help LLM understand available skills during selection.
        
        Returns:
            Multi-line string with skill descriptions.
        """
        lines = []
        for name in sorted(cls._skills.keys()):
            skill = cls._skills[name]
            lines.append(f"- {name}: {skill.config.description}")
        return "\n".join(lines)
    
    @classmethod
    def get_active_skills(cls) -> List[str]:
        """Get list of skills suitable for current context.
        
        For now, returns all registered skills.
        Can be extended in future for context-aware filtering.
        
        Returns:
            List of skill names.
        """
        return cls.list_skills()


# Global instance
skill_registry = SkillRegistry()
