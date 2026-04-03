from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class SessionHistory(Protocol):
    messages: list[BaseMessage]

    def add_messages(self, messages: list[BaseMessage]) -> None:
        ...


@dataclass
class InMemorySessionHistory:
    """Project-local in-memory session history container."""

    messages: list[BaseMessage] = field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)

    def add_user_message(self, message: HumanMessage | str) -> None:
        if isinstance(message, HumanMessage):
            self.messages.append(message)
            return
        self.messages.append(HumanMessage(content=str(message)))

    def add_ai_message(self, message: AIMessage | str) -> None:
        if isinstance(message, AIMessage):
            self.messages.append(message)
            return
        self.messages.append(AIMessage(content=str(message)))
