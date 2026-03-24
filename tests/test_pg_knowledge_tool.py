from app.agents.tools import (
    clear_knowledge_retrieval_log,
    get_knowledge_retrieval_log,
    retrieve_pg_knowledge,
)


def test_retrieve_pg_knowledge_requires_dsn() -> None:
    clear_knowledge_retrieval_log()

    result = retrieve_pg_knowledge.invoke({"query": "项目的记忆机制是什么"})

    assert "PGVECTOR_DSN" in result

    logs = get_knowledge_retrieval_log()
    assert len(logs) == 1
    assert "记忆机制" in logs[0]["query"]
    assert "PGVECTOR_DSN" in logs[0]["result"]
