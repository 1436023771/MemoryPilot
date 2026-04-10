import app.agents.tools.tools_lightnovel_translate as trans_module
from app.agents.tools import (
    clear_translate_log,
    get_translate_log,
    translate_light_novel_ja_to_zh,
)


def test_translate_lightnovel_success(monkeypatch) -> None:
    monkeypatch.setattr(
        trans_module,
        "_ollama_translate",
        lambda text, style, model=None: "她轻轻点了点头。",
    )

    clear_translate_log()
    result = translate_light_novel_ja_to_zh.invoke(
        {
            "text": "彼女はそっと頷いた。",
            "style": "faithful",
        }
    )

    assert result == "她轻轻点了点头。"

    logs = get_translate_log()
    assert len(logs) == 1
    assert logs[0]["style"] == "faithful"
    assert "彼女はそっと頷いた" in logs[0]["text"]


def test_translate_lightnovel_empty_text() -> None:
    clear_translate_log()
    result = translate_light_novel_ja_to_zh.invoke({"text": "   "})

    assert "文本为空" in result
    logs = get_translate_log()
    assert len(logs) == 1
    assert "翻译失败" in logs[0]["result"]


def test_translate_lightnovel_ollama_error(monkeypatch) -> None:
    def _raise_error(text, style, model=None):
        raise RuntimeError("模型未返回有效译文")

    monkeypatch.setattr(trans_module, "_ollama_translate", _raise_error)

    clear_translate_log()
    result = translate_light_novel_ja_to_zh.invoke(
        {"text": "行こう。", "style": "fluent"}
    )

    assert "翻译失败" in result
    assert "模型未返回有效译文" in result


def test_extract_translation_strips_wrappers() -> None:
    wrapped = "```\n译文：\"你好，世界。\"\n```"
    assert trans_module._extract_translation(wrapped) == "你好，世界。"


def test_normalize_model_candidates_dedup(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_JA_ZH_MODEL", "primary-model")
    monkeypatch.setenv("OLLAMA_JA_ZH_FALLBACK_MODELS", "fallback-a, fallback-b, primary-model")

    models = trans_module._normalize_model_candidates()
    assert models == ["primary-model", "fallback-a", "fallback-b"]


def test_ollama_translate_returns_diagnostic_when_all_empty(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_JA_ZH_MODEL", "primary-model")
    monkeypatch.setenv("OLLAMA_JA_ZH_FALLBACK_MODELS", "fallback-model")

    def _fake_post(url, payload, timeout_seconds):
        if url.endswith("/api/chat"):
            return {
                "message": {"role": "assistant", "content": ""},
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 1,
            }
        return {
            "response": "",
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 1,
        }

    monkeypatch.setattr(trans_module, "_post_json", _fake_post)

    result = trans_module._translate_light_novel_impl("彼女はそっと頷いた。")
    assert "模型未返回有效译文" in result
    assert "primary-model" in result
    assert "fallback-model" in result


def test_ollama_translate_prefers_generate_with_im_template(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_JA_ZH_MODEL", "primary-model")
    monkeypatch.setenv("OLLAMA_JA_ZH_FALLBACK_MODELS", "")

    calls: list[tuple[str, dict]] = []

    def _fake_post(url, payload, timeout_seconds):
        calls.append((url, payload))
        if url.endswith("/api/generate"):
            return {"response": "她轻轻点了点头。", "done_reason": "stop", "eval_count": 6}
        return {"message": {"content": ""}, "done_reason": "stop", "eval_count": 1}

    monkeypatch.setattr(trans_module, "_post_json", _fake_post)

    translated = trans_module._ollama_translate("彼女はそっと頷いた。", "faithful")
    assert translated == "她轻轻点了点头。"

    first_url, first_payload = calls[0]
    assert first_url.endswith("/api/generate")
    assert first_payload.get("raw") is True
    assert "<|im_start|>system" in first_payload.get("prompt", "")
