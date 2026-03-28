"""测试批量 narrative 提取功能"""
from __future__ import annotations

import pytest

from app.knowledge.narrative_extraction import (
    build_narrative_fields,
    build_narrative_fields_batch,
    analyze_contents_batch_with_cache,
    _NARRATIVE_CACHE,
)


def test_batch_analyze_returns_correct_length(monkeypatch):
    """验证批量分析返回正确的结果数量"""
    # 模拟 LLM 响应
    def mock_analyze_batch(contents):
        return [
            {
                "narrative_context": "present",
                "time_markers": [f"time_{i}"],
                "character_mentions": [f"char_{i}"],
                "relationship_edges": [],
            }
            for i in range(len(contents))
        ]

    monkeypatch.setattr(
        "app.knowledge.narrative_extraction._call_llm_for_batch_analysis",
        mock_analyze_batch,
    )
    _NARRATIVE_CACHE.clear()

    contents = ["content1", "content2", "content3"]
    results = analyze_contents_batch_with_cache(contents)

    assert len(results) == 3
    assert results[0]["narrative_context"] == "present"
    assert results[1]["time_markers"] == ["time_1"]
    assert results[2]["character_mentions"] == ["char_2"]


def test_batch_build_narrative_fields(monkeypatch):
    """验证批量构建 NarrativeFields"""

    def mock_analyze_batch(contents):
        return [
            {
                "narrative_context": "present",
                "time_markers": ["第一天", "第二天"],
                "character_mentions": ["角色A", "角色B"],
                "relationship_edges": [{"source": "角色A", "target": "角色B", "relation": "对立"}],
            }
            for _ in contents
        ]

    monkeypatch.setattr(
        "app.knowledge.narrative_extraction._call_llm_for_batch_analysis",
        mock_analyze_batch,
    )
    _NARRATIVE_CACHE.clear()

    contents = ["piece1", "piece2"]
    fields_list = build_narrative_fields_batch(
        book_id="test_book",
        chapter="test_chapter",
        chunk_indices=[0, 1],
        chunk_orders=[1, 2],
        contents=contents,
    )

    assert len(fields_list) == 2
    assert fields_list[0].chunk_order == 1
    assert fields_list[1].chunk_order == 2
    assert all(f.narrative_context == "present" for f in fields_list)
    assert all(len(f.character_mentions) == 2 for f in fields_list)
    assert all(len(f.relationship_edges) == 1 for f in fields_list)


def test_batch_with_cache_hit(monkeypatch):
    """验证批量调用时缓存命中的情况"""
    call_count = 0

    def mock_analyze_batch(contents):
        nonlocal call_count
        call_count += 1
        return [
            {
                "narrative_context": "present",
                "time_markers": [],
                "character_mentions": [],
                "relationship_edges": [],
            }
            for _ in contents
        ]

    monkeypatch.setattr(
        "app.knowledge.narrative_extraction._call_llm_for_batch_analysis",
        mock_analyze_batch,
    )
    _NARRATIVE_CACHE.clear()

    # 第一次调用：3 个内容都要分析
    contents1 = ["content_a", "content_b", "content_c"]
    results1 = analyze_contents_batch_with_cache(contents1)
    assert call_count == 1  # 只调用 1 次 LLM
    assert len(results1) == 3

    # 第二次调用：相同的 3 个内容，应该命中缓存
    results2 = analyze_contents_batch_with_cache(contents1)
    assert call_count == 1  # 没有再调用 LLM
    assert results2 == results1

    # 第三次调用：2 个新内容 + 1 个已缓存的内容
    contents3 = ["content_a", "content_new_1", "content_new_2"]
    results3 = analyze_contents_batch_with_cache(contents3)
    assert call_count == 2  # 新增了 1 次 LLM 调用（处理 2 个新内容）
    assert len(results3) == 3
    assert results3[0] == results1[0]  # content_a 应该和之前的缓存结果相同


def test_batch_vs_single_consistency(monkeypatch):
    """验证批量调用和单个调用的结果一致性"""

    def mock_analyze_batch(contents):
        return [
            {
                "narrative_context": "present",
                "time_markers": ["time" + str(i)],
                "character_mentions": ["char" + str(i)],
                "relationship_edges": [],
            }
            for i in range(len(contents))
        ]

    monkeypatch.setattr(
        "app.knowledge.narrative_extraction._call_llm_for_batch_analysis",
        mock_analyze_batch,
    )

    def mock_analyze_single(content):
        return {
            "narrative_context": "present",
            "time_markers": ["time_single"],
            "character_mentions": ["char_single"],
            "relationship_edges": [],
        }

    monkeypatch.setattr(
        "app.knowledge.narrative_extraction._call_llm_for_analysis",
        mock_analyze_single,
    )
    _NARRATIVE_CACHE.clear()

    contents = ["test1", "test2"]

    # 分别用批量和单个方式构建
    batch_fields = build_narrative_fields_batch(
        book_id="book",
        chapter="chap",
        chunk_indices=[0, 1],
        chunk_orders=[1, 2],
        contents=contents,
    )

    # 两种方式都应该返回 NarrativeFields 对象
    assert all(hasattr(f, "narrative_context") for f in batch_fields)
    assert all(hasattr(f, "chunk_order") for f in batch_fields)
    assert batch_fields[0].chunk_order == 1
    assert batch_fields[1].chunk_order == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
