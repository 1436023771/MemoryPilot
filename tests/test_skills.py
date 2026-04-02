"""
Unit tests for agent skill framework.

Tests skill registration, configuration, and selection logic.
"""

import pytest
from app.agents.skills import skill_registry, SkillRegistry, Skill, SkillConfig
from app.agents.skills.runtime import (
    DEFAULT_SKILL_NAME,
    process_context_for_skill,
    resolve_prompt_key,
    resolve_skill_name,
    resolve_tool_display_name,
    select_tools_for_skill,
)


class TestSkillRegistry:
    """Test SkillRegistry functionality."""

    def test_list_skills_returns_registered_skills(self):
        """Verify registry contains both skills."""
        skills = skill_registry.list_skills()
        assert "reading-companion" in skills
        assert "general" in skills
        assert len(skills) >= 2

    def test_get_skill_returns_correct_skill(self):
        """Verify skill retrieval by name."""
        reading = skill_registry.get_skill("reading-companion")
        assert reading is not None
        assert reading.config.name == "reading-companion"
        
        general = skill_registry.get_skill("general")
        assert general is not None
        assert general.config.name == "general"

    def test_get_skill_returns_none_for_unknown_skill(self):
        """Verify unknown skill returns None."""
        unknown = skill_registry.get_skill("nonexistent-skill")
        assert unknown is None

    def test_list_descriptions_returns_formatted_string(self):
        """Verify skill descriptions are formatted correctly."""
        descriptions = skill_registry.list_descriptions()
        assert isinstance(descriptions, str)
        assert "reading-companion:" in descriptions
        assert "general:" in descriptions
        assert "\n" in descriptions


class TestReadingCompanionSkill:
    """Test ReadingCompanionSkill configuration and behavior."""

    def test_reading_companion_config(self):
        """Verify reading companion skill configuration."""
        skill = skill_registry.get_skill("reading-companion")
        assert skill is not None
        assert skill.config.name == "reading-companion"
        assert "book" in skill.config.description.lower()
        assert skill.config.system_prompt_override == "agents.langgraph.reading_companion_prompt"

    def test_reading_companion_tool_restrictions(self):
        """Verify reading companion has restricted tool set."""
        skill = skill_registry.get_skill("reading-companion")
        allowed_tools = skill.config.available_tools
        
        # Should NOT have web_search
        assert "web_search" not in allowed_tools
        
        # Should have knowledge, python, docker
        assert "retrieve_pg_knowledge" in allowed_tools
        assert "run_python_code" in allowed_tools
        assert "run_docker_command" in allowed_tools

    def test_reading_companion_should_activate_with_keywords(self):
        """Verify keyword-based activation."""
        skill = skill_registry.get_skill("reading-companion")
        
        # Should activate for reading keywords
        assert skill.should_activate("讲讲这本书的情节") is True
        assert skill.should_activate("有什么章节") is True
        assert skill.should_activate("人物关系是什么") is True
        assert skill.should_activate("book chapter timeline") is True
        
        # Should not activate for non-reading questions
        assert skill.should_activate("如何编写Python程序") is False
        assert skill.should_activate("请执行这个命令") is False

    def test_reading_companion_tool_display_names(self):
        """Verify custom tool display names."""
        skill = skill_registry.get_skill("reading-companion")
        display_names = skill.config.tool_display_names
        
        assert display_names.get("retrieve_pg_knowledge") == "书籍检索"
        assert display_names.get("run_python_code") == "数据处理"
        assert display_names.get("run_docker_command") == "高级计算"


class TestGeneralSkill:
    """Test GeneralSkill configuration and behavior."""

    def test_general_skill_config(self):
        """Verify general skill configuration."""
        skill = skill_registry.get_skill("general")
        assert skill is not None
        assert skill.config.name == "general"
        assert "general" in skill.config.description.lower()
        assert skill.config.system_prompt_override == "agents.langgraph.final_user_prompt"

    def test_general_skill_allows_all_tools(self):
        """Verify general skill allows all tools."""
        skill = skill_registry.get_skill("general")
        allowed_tools = skill.config.available_tools
        
        # Should have all tools
        assert "web_search" in allowed_tools
        assert "retrieve_pg_knowledge" in allowed_tools
        assert "run_python_code" in allowed_tools
        assert "run_docker_command" in allowed_tools

    def test_general_skill_tool_display_names(self):
        """Verify general skill uses Chinese display names."""
        skill = skill_registry.get_skill("general")
        display_names = skill.config.tool_display_names
        
        assert display_names.get("web_search") == "联网搜索"
        assert display_names.get("retrieve_pg_knowledge") == "知识库检索"
        assert display_names.get("run_python_code") == "Python计算器"
        assert display_names.get("run_docker_command") == "Docker沙箱"


class TestSkillContextProcessing:
    """Test context processing capabilities."""

    def test_default_context_processor_returns_unchanged(self):
        """Verify default context processor returns input unchanged."""
        skill = skill_registry.get_skill("general")
        
        memory_ctx = "test memory"
        knowledge_ctx = "test knowledge"
        
        result = skill.process_context(memory_ctx, knowledge_ctx)
        assert result["memory_ctx"] == memory_ctx
        assert result["knowledge_ctx"] == knowledge_ctx

    def test_context_processor_with_custom_processor(self):
        """Verify custom context processor is called."""
        def custom_processor(memory_ctx, knowledge_ctx):
            return {
                "memory_ctx": f"[CUSTOM] {memory_ctx}",
                "knowledge_ctx": f"[CUSTOM] {knowledge_ctx}",
            }
        
        config = SkillConfig(
            name="test-skill",
            description="test",
            context_processor=custom_processor,
        )
        skill = Skill(config)
        
        result = skill.process_context("memory", "knowledge")
        assert result["memory_ctx"] == "[CUSTOM] memory"
        assert result["knowledge_ctx"] == "[CUSTOM] knowledge"


class TestSkillRuntimeHelpers:
    """Test skill runtime helper functions."""

    def test_resolve_skill_name_fallbacks_to_general(self):
        assert resolve_skill_name("non-existent-skill") == DEFAULT_SKILL_NAME
        assert resolve_skill_name("") == DEFAULT_SKILL_NAME

    def test_select_tools_for_skill_applies_restrictions(self):
        reading_tools = [tool.name for tool in select_tools_for_skill("reading-companion")]
        general_tools = [tool.name for tool in select_tools_for_skill("general")]

        assert "web_search" not in reading_tools
        assert "retrieve_pg_knowledge" in reading_tools
        assert "run_python_code" in reading_tools
        assert "run_docker_command" in reading_tools

        assert "web_search" in general_tools
        assert "retrieve_pg_knowledge" in general_tools

    def test_resolve_tool_display_name_prefers_skill_custom_name(self):
        assert resolve_tool_display_name("reading-companion", "run_python_code") == "数据处理"
        assert resolve_tool_display_name("reading-companion", "run_docker_command") == "高级计算"

    def test_resolve_prompt_key_uses_skill_override(self):
        assert (
            resolve_prompt_key("reading-companion", "agents.langgraph.final_user_prompt")
            == "agents.langgraph.reading_companion_prompt"
        )
        assert (
            resolve_prompt_key("general", "agents.langgraph.final_user_prompt")
            == "agents.langgraph.final_user_prompt"
        )

    def test_process_context_for_skill_keeps_default_behavior(self):
        memory_ctx, knowledge_ctx = process_context_for_skill("general", "m", "k")
        assert memory_ctx == "m"
        assert knowledge_ctx == "k"
