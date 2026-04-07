from __future__ import annotations

from pathlib import Path

from app.agents.skills import skill_registry


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1] / ".github" / "skills"


def test_every_registered_skill_has_standard_folder_format() -> None:
    root = _skills_root()
    assert root.exists(), "Missing standardized skills root: .github/skills"

    for skill_name in skill_registry.list_skills():
        skill_dir = root / skill_name
        assert skill_dir.exists(), f"Missing skill folder for '{skill_name}': {skill_dir}"
        assert (skill_dir / "skill.md").exists(), f"Missing required file: {skill_dir / 'skill.md'}"
        assert (skill_dir / "scripts").exists(), f"Missing optional scripts directory: {skill_dir / 'scripts'}"
        assert (skill_dir / "knowledge").exists(), f"Missing optional knowledge directory: {skill_dir / 'knowledge'}"
