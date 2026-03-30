import pytest

from app.core.config import get_env_bool, get_env_float, get_env_int, get_settings


def test_deepseek_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # 模拟 DeepSeek 环境变量，验证配置解析结果。
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("MODEL_NAME", "deepseek-chat")
    monkeypatch.setenv("TEMPERATURE", "0.3")

    settings = get_settings()

    assert settings.provider == "deepseek"
    assert settings.api_key == "test-key"
    assert settings.base_url == "https://api.deepseek.com/v1"
    assert settings.model_name == "deepseek-chat"
    assert settings.temperature == 0.3


def test_openai_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # 模拟 OpenAI 环境变量，验证不同提供方切换正确。
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("MODEL_NAME", "gpt-4.1-mini")

    settings = get_settings()

    assert settings.provider == "openai"
    assert settings.api_key == "test-openai-key"
    assert settings.base_url is None
    assert settings.model_name == "gpt-4.1-mini"


def test_get_env_int_and_float(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_TOP_K_DEFAULT", "7")
    monkeypatch.setenv("KNOWLEDGE_BLEND_WEIGHT_LLM", "0.75")

    assert get_env_int("KNOWLEDGE_TOP_K_DEFAULT", default=5, min_value=1) == 7
    assert get_env_float("KNOWLEDGE_BLEND_WEIGHT_LLM", default=0.6, min_value=0.0) == 0.75


def test_get_env_int_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_TOP_K_DEFAULT", "abc")

    with pytest.raises(ValueError, match="KNOWLEDGE_TOP_K_DEFAULT"):
        get_env_int("KNOWLEDGE_TOP_K_DEFAULT", default=5, min_value=1)


def test_get_env_bool_accepts_common_literals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_SYNC_INCREMENTAL", "yes")
    assert get_env_bool("KNOWLEDGE_SYNC_INCREMENTAL", default=False) is True

    monkeypatch.setenv("KNOWLEDGE_SYNC_INCREMENTAL", "off")
    assert get_env_bool("KNOWLEDGE_SYNC_INCREMENTAL", default=True) is False
