from pathlib import Path

from app.write_memory import append_memory_facts, extract_candidate_facts


def test_extract_candidate_facts_from_user_input() -> None:
    facts = extract_candidate_facts("我叫小李。我喜欢Python。我的目标是实现长期记忆")

    assert "用户姓名：小李；我是小李。" in facts
    assert "用户喜欢：Python；我喜欢Python。" in facts
    assert "用户目标：实现长期记忆；我的目标是实现长期记忆。" in facts


def test_append_memory_facts_with_dedup(tmp_path: Path) -> None:
    memory_file = tmp_path / "long_term_memory.txt"
    memory_file.write_text("# 自动写入记忆\n用户姓名：小李；我是小李。\n", encoding="utf-8")

    written = append_memory_facts(memory_file, ["用户姓名：小李。", "我喜欢Python"]) 

    content = memory_file.read_text(encoding="utf-8")
    assert written == ["用户喜欢：Python；我喜欢Python。"]
    assert content.count("用户姓名：小李；我是小李。") == 1
    assert "用户喜欢：Python；我喜欢Python。" in content


def test_append_memory_facts_updates_single_value_fields(tmp_path: Path) -> None:
    memory_file = tmp_path / "long_term_memory.txt"
    memory_file.write_text("# 自动写入记忆\n用户姓名：小李；我是小李。\n", encoding="utf-8")

    written = append_memory_facts(memory_file, ["我叫李华"])

    content = memory_file.read_text(encoding="utf-8")
    assert written == ["用户姓名：李华；我是李华。"]
    assert "用户姓名：小李；我是小李。" not in content
    assert "用户姓名：李华；我是李华。" in content
