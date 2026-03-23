from pathlib import Path

from app.memory.write_memory import MemoryFact, append_memory_facts, extract_candidate_facts


def test_extract_candidate_facts_from_user_input() -> None:
    facts = extract_candidate_facts("我叫小李。我喜欢Python。我的目标是实现长期记忆")
    pairs = {(fact.key, fact.value) for fact in facts}

    assert ("name", "小李") in pairs
    assert ("like", "Python") in pairs
    assert ("goal", "实现长期记忆") in pairs


def test_append_memory_facts_with_dedup(tmp_path: Path) -> None:
    memory_file = tmp_path / "long_term_memory.txt"
    memory_file.write_text("# 自动写入记忆\nname|小李\n", encoding="utf-8")

    written = append_memory_facts(
        memory_file,
        [
            MemoryFact(key="name", value="小李", text="用户姓名：小李。"),
            MemoryFact(key="like", value="Python", text="用户喜欢：Python。"),
        ],
    )

    content = memory_file.read_text(encoding="utf-8")
    assert written == ["用户喜欢：Python。"]
    assert content.count("name|小李") == 1
    assert "like|Python" in content


def test_append_memory_facts_updates_single_value_fields(tmp_path: Path) -> None:
    memory_file = tmp_path / "long_term_memory.txt"
    memory_file.write_text("# 自动写入记忆\nname|小李\n", encoding="utf-8")

    written = append_memory_facts(
        memory_file,
        [MemoryFact(key="name", value="李华", text="用户姓名：李华。")],
    )

    content = memory_file.read_text(encoding="utf-8")
    assert written == ["用户姓名：李华。"]
    assert "name|小李" not in content
    assert "name|李华" in content
