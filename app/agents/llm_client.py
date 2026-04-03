from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import Settings

try:
    from langsmith import tracing_context
except Exception:  # noqa: BLE001
    tracing_context = None


def create_chat_model(settings: Settings, temperature: float | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature if temperature is None else temperature,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )


def invoke_model(
    settings: Settings,
    prompt_or_messages: Any,
    *,
    temperature: float | None = None,
    disable_tracing: bool = False,
):
    model = create_chat_model(settings=settings, temperature=temperature)
    if disable_tracing and tracing_context is not None:
        with tracing_context(enabled=False):
            return model.invoke(prompt_or_messages)
    return model.invoke(prompt_or_messages)


async def ainvoke_model(
    settings: Settings,
    prompt_or_messages: Any,
    *,
    temperature: float | None = None,
    disable_tracing: bool = False,
):
    model = create_chat_model(settings=settings, temperature=temperature)
    if disable_tracing and tracing_context is not None:
        with tracing_context(enabled=False):
            return await model.ainvoke(prompt_or_messages)
    return await model.ainvoke(prompt_or_messages)
