from app.agents.tools.tools_web_search import _build_query_variants


def test_build_query_variants_keeps_base_first() -> None:
    query = "最新 美国 CPI 数据 和 非农 就业"
    variants = _build_query_variants(query)

    assert variants
    assert variants[0] == "最新 美国 CPI 数据 和 非农 就业"


def test_build_query_variants_removes_generic_recency_words() -> None:
    query = "latest iPhone release date and price"
    variants = _build_query_variants(query)

    assert any("latest" not in v.lower() for v in variants[1:])


def test_build_query_variants_dedup_and_limit() -> None:
    query = "最近，最近！！  OpenAI   GPT-5.3   发布 了吗？？"
    variants = _build_query_variants(query)

    lowered = [v.lower().strip() for v in variants]
    assert len(variants) <= 5
    assert len(lowered) == len(set(lowered))
