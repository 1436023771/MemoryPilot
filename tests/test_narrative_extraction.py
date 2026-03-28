import app.knowledge.narrative_extraction as narrative_extraction


def test_build_narrative_fields_is_deterministic(monkeypatch) -> None:
    content = "第三天清晨，李雷在城门口遇到了韩梅梅。后来他们一起出发。"

    monkeypatch.setattr(
        narrative_extraction,
        "_analyze_content_with_cache",
        lambda _content: {
            "narrative_context": "present",
            "time_markers": ["第三天", "清晨"],
            "character_mentions": ["李雷", "韩梅梅"],
            "relationship_edges": [
                {"source": "李雷", "target": "韩梅梅", "relation": "相遇"},
            ],
        },
    )

    a = narrative_extraction.build_narrative_fields(
        book_id="demo-book",
        chapter="chapter-1",
        chunk_index=3,
        chunk_order=11,
        content=content,
    )
    b = narrative_extraction.build_narrative_fields(
        book_id="demo-book",
        chapter="chapter-1",
        chunk_index=3,
        chunk_order=11,
        content=content,
    )

    assert a == b
    assert a.timeline_order == 11
    assert a.scene_id.startswith("demo-book:chapter-1:scene-")
    assert a.event_id.startswith("demo-book:chapter-1:event-")
    assert "第三天" in a.time_markers
    assert a.character_mentions == ["李雷", "韩梅梅"]
    assert a.relationship_edges == [{"source": "李雷", "target": "韩梅梅", "relation": "相遇"}]
