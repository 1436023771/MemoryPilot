"""Unit tests for batch light novel translation tool."""

import tempfile
from pathlib import Path

import pytest

from app.agents.tools.tools_lightnovel_batch import (
    translate_light_novel_batch,
)


def test_translate_batch_source_dir_not_exists() -> None:
    """Verify error when source directory does not exist."""
    result = translate_light_novel_batch.invoke({
        "source_dir": "/nonexistent/path",
        "target_dir": "/tmp/out",
    })
    assert "翻译失败" in result
    assert "源目录不存在" in result


def test_translate_batch_creates_target_dir(tmp_path) -> None:
    """Verify target directory is created if it doesn't exist."""
    source = tmp_path / "src"
    target = tmp_path / "dst" / "nested"
    source.mkdir()
    
    # Write a test file
    test_file = source / "test.txt"
    test_file.write_text("彼女はそっと頷いた。", encoding="utf-8")
    
    result = translate_light_novel_batch.invoke({
        "source_dir": str(source),
        "target_dir": str(target),
        "style": "faithful",
    })
    
    # Target directory should be created
    assert target.exists()
    assert "翻译完成" in result


def test_translate_batch_processes_multiple_files(tmp_path, monkeypatch) -> None:
    """Verify batch translation processes multiple files."""
    import app.agents.tools.tools_lightnovel_batch as batch_module
    
    source = tmp_path / "src"
    target = tmp_path / "dst"
    source.mkdir()
    
    # Create test files
    (source / "story1.txt").write_text("彼女はそっと頷いた。", encoding="utf-8")
    (source / "story2.md").write_text("彼は静かに歩いていった。", encoding="utf-8")
    (source / "ignore.json").write_text('{"text": "彼女の声が聞こえた。"}', encoding="utf-8")
    
    # Mock the single-file translator to return predictable output
    def mock_translate(text, style="faithful", model=None):
        if "彼女" in text:
            return "她轻轻点了点头。"
        if "彼は" in text:
            return "他静静地走了。"
        if "彼女の声" in text:
            return "我听到了她的声音。"
        return "翻译成功。"
    
    monkeypatch.setattr(batch_module, "_translate_light_novel_impl", mock_translate)
    
    result = translate_light_novel_batch.invoke({
        "source_dir": str(source),
        "target_dir": str(target),
        "file_extensions": "txt,md,json",
    })
    
    # Check results
    assert "翻译完成" in result
    assert (target / "story1.txt").exists()
    assert (target / "story1.txt").read_text(encoding="utf-8") == "她轻轻点了点头。"
    assert (target / "story2.md").exists()


def test_translate_batch_respects_file_extensions(tmp_path, monkeypatch) -> None:
    """Verify only specified file extensions are processed."""
    import app.agents.tools.tools_lightnovel_batch as batch_module
    
    source = tmp_path / "src"
    target = tmp_path / "dst"
    source.mkdir()
    
    (source / "story.txt").write_text("彼女はそっと頷いた。", encoding="utf-8")
    (source / "note.md").write_text("彼は静かに歩いていった。", encoding="utf-8")
    (source / "data.json").write_text('{"text": "test"}', encoding="utf-8")
    
    monkeypatch.setattr(batch_module, "_translate_light_novel_impl", lambda text, style="faithful", model=None: "✓")
    
    # Only process txt files
    result = translate_light_novel_batch.invoke({
        "source_dir": str(source),
        "target_dir": str(target),
        "file_extensions": "txt",
    })
    
    assert (target / "story.txt").exists()
    assert not (target / "note.md").exists()
    assert not (target / "data.json").exists()


def test_translate_batch_preserves_directory_structure(tmp_path, monkeypatch) -> None:
    """Verify directory structure is preserved in target."""
    import app.agents.tools.tools_lightnovel_batch as batch_module
    
    source = tmp_path / "src"
    target = tmp_path / "dst"
    source.mkdir()
    (source / "vol1").mkdir()
    (source / "vol1" / "ch1").mkdir()
    
    (source / "vol1" / "ch1" / "story.txt").write_text("彼女はそっと頷いた。", encoding="utf-8")
    
    monkeypatch.setattr(batch_module, "_translate_light_novel_impl", lambda text, style="faithful", model=None: "✓")
    
    translate_light_novel_batch.invoke({
        "source_dir": str(source),
        "target_dir": str(target),
    })
    
    # Check nested structure is preserved
    assert (target / "vol1" / "ch1" / "story.txt").exists()


def test_translate_batch_handles_read_error(tmp_path) -> None:
    """Verify errors during file reading are handled gracefully."""
    source = tmp_path / "src"
    target = tmp_path / "dst"
    source.mkdir()
    
    # Create a file but don't give read permission
    test_file = source / "test.txt"
    test_file.write_text("彼女はそっと頷いた。", encoding="utf-8")
    test_file.chmod(0o000)
    
    try:
        result = translate_light_novel_batch.invoke({
            "source_dir": str(source),
            "target_dir": str(target),
        })
        # Should handle permission error gracefully
        assert "翻译完成" in result or "失败" in result
    finally:
        # Restore permission for cleanup
        test_file.chmod(0o644)


def test_translate_batch_returns_summary_with_stats(tmp_path, monkeypatch) -> None:
    """Verify summary includes counts and statistics."""
    import app.agents.tools.tools_lightnovel_batch as batch_module
    
    source = tmp_path / "src"
    target = tmp_path / "dst"
    source.mkdir()
    
    (source / "a.txt").write_text("彼女。", encoding="utf-8")
    (source / "b.txt").write_text("彼。", encoding="utf-8")
    
    monkeypatch.setattr(batch_module, "_translate_light_novel_impl", lambda text, style="faithful", model=None: "✓")
    
    result = translate_light_novel_batch.invoke({
        "source_dir": str(source),
        "target_dir": str(target),
    })
    
    assert "成功:" in result
    assert "失败:" in result
    assert "字符" in result
    assert f"目标目录: {target}" in result
