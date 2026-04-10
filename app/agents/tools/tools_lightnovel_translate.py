"""Local Ollama tool for Japanese-to-Chinese light novel translation."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from app.agents.tool_definition import ToolDefinition

_MAX_TEXT_CHARS = 6000
_DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
_DEFAULT_MODEL = "sakura-14b-qwen2.5-v1.0-iq4xs:latest"
_DEFAULT_TIMEOUT_SECONDS = 60
_DEFAULT_FALLBACK_MODELS = ""
_translate_log: list[dict] = []


def get_translate_log() -> list[dict]:
    """Return a copy of current turn translation log."""
    return _translate_log.copy()


def clear_translate_log() -> None:
    """Clear translation log at the start of each turn."""
    global _translate_log
    _translate_log = []


def record_translation(text: str, style: str, result: str) -> None:
    """Record source text, style, and translated output for inspection."""
    global _translate_log
    _translate_log.append({"text": text, "style": style, "result": result})


def _normalize_style(style: str) -> str:
    normalized = str(style or "").strip().lower()
    if normalized in {"literal", "faithful", "fluent"}:
        return normalized
    return "faithful"


def _build_messages(text: str, style: str) -> list[dict[str, str]]:
    style_desc = {
        "literal": "尽量直译，严格贴近原文句式和信息",
        "faithful": "忠实自然，保留原意与语气，读感通顺",
        "fluent": "优先中文流畅度，在不改变事实的前提下适度润色",
    }[style]

    system_prompt = (
        "你是轻小说日译中助手。"
        "任务是把日文翻译成简体中文。"
        "必须只输出译文，不要解释、不要注释、不要额外前后缀。"
        "保留人名、地名、专有名词的一致性。"
    )
    user_prompt = (
        f"翻译风格：{style_desc}。\n"
        "请将下面日文翻译成简体中文，只输出译文：\n"
        f"{text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_chat_messages_variants(text: str, style: str) -> list[list[dict[str, str]]]:
    base = _build_messages(text=text, style=style)
    compact = [
        {
            "role": "system",
            "content": "你是轻小说日译中助手。只输出简体中文译文。",
        },
        {
            "role": "user",
            "content": f"把这句日文翻译成中文：{text}",
        },
    ]
    structured = [
        {
            "role": "system",
            "content": "你是翻译助手。只返回译文，不要解释。",
        },
        {
            "role": "user",
            "content": (
                "翻译任务\n"
                "源语言: 日文\n"
                "目标语言: 简体中文\n"
                f"风格: {style}\n"
                f"文本: {text}\n"
                "仅输出译文"
            ),
        },
    ]
    return [base, compact, structured]


def _build_generate_prompts(text: str, style: str) -> list[str]:
    style_desc = {
        "literal": "尽量直译，严格贴近原文句式和信息",
        "faithful": "忠实自然，保留原意与语气，读感通顺",
        "fluent": "优先中文流畅度，在不改变事实的前提下适度润色",
    }[style]

    return [
        (
            "<|im_start|>system\n"
            "你是轻小说日译中助手。\n"
            "任务是把日文翻译成简体中文。\n"
            "必须只输出译文，不要解释、不要注释、不要额外前后缀。\n"
            "保留人名、地名、专有名词的一致性。\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"翻译风格：{style_desc}\n"
            "请将下面日文翻译成简体中文，只输出译文：\n"
            f"{text}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        (
            "<|im_start|>system\n"
            "你是翻译助手。只返回简体中文译文，不要解释。\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"把这句日文翻译成简体中文：{text}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    ]


def _post_json(url: str, payload: dict, timeout_seconds: int) -> dict:
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:  # noqa: S310
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _extract_translation(content: str) -> str:
    text = str(content or "").strip()
    if not text:
        return ""

    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    text = re.sub(r"^(译文|翻译)\s*[:：]\s*", "", text)
    text = text.strip().strip('"').strip("'").strip()
    return text


def _normalize_model_candidates(model: str | None = None) -> list[str]:
    primary = (model or os.getenv("OLLAMA_JA_ZH_MODEL", _DEFAULT_MODEL)).strip() or _DEFAULT_MODEL
    raw_fallback = os.getenv("OLLAMA_JA_ZH_FALLBACK_MODELS", _DEFAULT_FALLBACK_MODELS)
    fallback = [item.strip() for item in raw_fallback.split(",") if item.strip()]

    merged = [primary] + fallback
    deduped: list[str] = []
    seen: set[str] = set()
    for item in merged:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _extract_text_from_response(data: dict) -> str:
    if not isinstance(data, dict):
        return ""

    message = data.get("message", {})
    if isinstance(message, dict):
        content = str(message.get("content", "") or "")
        if content.strip():
            return content

    response = str(data.get("response", "") or "")
    if response.strip():
        return response

    return ""


def _summarize_attempt(endpoint: str, model_name: str, data: dict) -> str:
    if not isinstance(data, dict):
        return f"{endpoint}:{model_name}:invalid_response"

    done_reason = str(data.get("done_reason", "")).strip() or "unknown"
    eval_count = data.get("eval_count", "?")
    prompt_eval_count = data.get("prompt_eval_count", "?")
    return (
        f"{endpoint}:{model_name}:done_reason={done_reason},"
        f"prompt_eval={prompt_eval_count},eval={eval_count}"
    )


def _ollama_translate(text: str, style: str, model: str | None = None) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE_URL).strip() or _DEFAULT_OLLAMA_BASE_URL
    timeout_seconds = int(os.getenv("OLLAMA_TRANSLATE_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT_SECONDS)).strip() or _DEFAULT_TIMEOUT_SECONDS)

    model_candidates = _normalize_model_candidates(model=model)
    attempt_summaries: list[str] = []

    for model_name in model_candidates:
        # Primary path: generate + Qwen-style templated prompt.
        for prompt in _build_generate_prompts(text=text, style=style):
            payload = {
                "model": model_name,
                "stream": False,
                "prompt": prompt,
                "raw": True,
                "options": {"temperature": 0.2, "num_predict": 512},
            }
            data = _post_json(f"{base_url.rstrip('/')}/api/generate", payload, timeout_seconds)
            translated = _extract_translation(_extract_text_from_response(data))
            if translated:
                return translated
            attempt_summaries.append(_summarize_attempt("generate", model_name, data))

        # Minimal fallback: chat once for compatibility with other models.
        payload = {
            "model": model_name,
            "stream": False,
            "messages": _build_messages(text=text, style=style),
            "options": {"temperature": 0.2, "num_predict": 512},
        }
        data = _post_json(f"{base_url.rstrip('/')}/api/chat", payload, timeout_seconds)
        translated = _extract_translation(_extract_text_from_response(data))
        if translated:
            return translated
        attempt_summaries.append(_summarize_attempt("chat", model_name, data))

    summary_text = "; ".join(attempt_summaries[:8]) if attempt_summaries else "no-attempts"
    raise RuntimeError(
        "模型未返回有效译文。"
        f"已尝试模型: {', '.join(model_candidates)}。"
        f"诊断: {summary_text}。"
        "可在 .env 配置 OLLAMA_JA_ZH_FALLBACK_MODELS=模型1,模型2 进行回退。"
    )


def _translate_light_novel_impl(text: str, style: str = "faithful", model: str | None = None) -> str:
    source = (text or "").strip()
    if not source:
        result = "翻译失败: 文本为空"
        record_translation(source, _normalize_style(style), result)
        return result

    if len(source) > _MAX_TEXT_CHARS:
        result = f"翻译失败: 文本过长（>{_MAX_TEXT_CHARS} 字符）"
        record_translation(source, _normalize_style(style), result)
        return result

    normalized_style = _normalize_style(style)
    try:
        result = _ollama_translate(source, normalized_style, model=model)
        record_translation(source, normalized_style, result)
        return result
    except urllib.error.URLError as exc:
        error = f"翻译失败: 无法连接本地 Ollama 服务 ({exc})"
        record_translation(source, normalized_style, error)
        return error
    except Exception as exc:  # noqa: BLE001
        error = f"翻译失败: {exc}"
        record_translation(source, normalized_style, error)
        return error


translate_light_novel_ja_to_zh = ToolDefinition(
    name="translate_light_novel_ja_to_zh",
    description="Translate Japanese light novel text to Simplified Chinese via local Ollama.",
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "style": {
                "type": "string",
                "enum": ["literal", "faithful", "fluent"],
                "description": "Translation style. Defaults to faithful.",
            },
            "model": {
                "type": "string",
                "description": "Optional Ollama model override.",
            },
        },
        "required": ["text"],
    },
    handler=_translate_light_novel_impl,
)


__all__ = [
    "translate_light_novel_ja_to_zh",
    "get_translate_log",
    "clear_translate_log",
    "record_translation",
]
