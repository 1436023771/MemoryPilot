from app.agents.tools import (
    clear_knowledge_retrieval_log,
    get_knowledge_retrieval_log,
    retrieve_pg_knowledge,
)
from app.agents.tools.tools_pg_knowledge import _analyze_query_with_cache, _apply_role_timeline_rerank


def test_retrieve_pg_knowledge_requires_dsn(monkeypatch) -> None:
    clear_knowledge_retrieval_log()
    monkeypatch.setattr("app.agents.tools.tools_pg_knowledge.resolve_pg_dsn", lambda _dsn: "")

    result = retrieve_pg_knowledge.invoke({"query": "项目的记忆机制是什么"})

    assert "缺少数据库连接配置" in result

    logs = get_knowledge_retrieval_log()
    assert len(logs) == 1
    assert "记忆机制" in logs[0]["query"]
    assert "缺少数据库连接配置" in logs[0]["result"]


def test_local_rerank_prefers_character_continuity() -> None:
    hits = [
        {
            "score": 0.90,
            "character_mentions": ["张三", "李四"],
            "timeline_order": 5,
            "time_markers": ["后来"],
        },
        {
            "score": 0.95,
            "character_mentions": ["王五"],
            "timeline_order": 6,
            "time_markers": ["后来"],
        },
    ]

    analysis = {
        "characters": ["张三", "李四"],
        "timeline_intent": "evolution",
        "relation_intent": True,
        "confidence": 0.9,
        "source": "test",
    }
    reranked, aux = _apply_role_timeline_rerank(hits, analysis)

    assert aux["local_rerank_used"] is True
    assert "张三" in aux["query_characters"]
    assert reranked[0]["character_mentions"] == ["张三", "李四"]
    assert reranked[0]["local_character_score"] >= reranked[1]["local_character_score"]


def test_local_rerank_timeline_query_prefers_timeline_metadata() -> None:
    hits = [
        {
            "score": 0.80,
            "character_mentions": ["张三"],
            "timeline_order": 0,
            "time_markers": [],
        },
        {
            "score": 0.70,
            "character_mentions": ["张三"],
            "timeline_order": 12,
            "time_markers": ["第三天"],
        },
    ]

    analysis = {
        "characters": ["张三"],
        "timeline_intent": "ordering",
        "relation_intent": False,
        "confidence": 0.8,
        "source": "test",
    }
    reranked, aux = _apply_role_timeline_rerank(hits, analysis)

    assert aux["timeline_intent"] == "ordering"
    assert reranked[0]["timeline_order"] == 12
    assert reranked[0]["local_timeline_score"] > reranked[1]["local_timeline_score"]


def test_query_analysis_cache_reuses_first_result(monkeypatch) -> None:
    calls = {"n": 0}

    def _fake_llm(query: str, candidate_characters: list[str]):
        calls["n"] += 1
        return {
            "characters": ["张三"],
            "timeline_intent": "ordering",
            "relation_intent": True,
            "confidence": 0.77,
            "source": "llm",
        }

    clear_knowledge_retrieval_log()
    monkeypatch.setattr("app.agents.tools.tools_pg_knowledge._analyze_query_with_llm", _fake_llm)

    a = _analyze_query_with_cache("张三后来做了什么", "book-a", ["张三", "李四"])
    b = _analyze_query_with_cache("张三后来做了什么", "book-a", ["张三", "李四", "王五"])

    assert calls["n"] == 1
    assert a == b
    assert a["characters"] == ["张三"]
