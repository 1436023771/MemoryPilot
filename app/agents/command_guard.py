from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal
import threading


CommandKind = Literal["shell", "python"]


@dataclass(frozen=True)
class CommandReviewRequest:
    kind: CommandKind
    payload: str
    reason: str


_review_handler: Callable[[CommandReviewRequest], bool] | None = None
_review_handler_lock = threading.Lock()


def register_command_review_handler(handler: Callable[[CommandReviewRequest], bool] | None) -> None:
    global _review_handler
    with _review_handler_lock:
        _review_handler = handler


def request_command_review(kind: CommandKind, payload: str, reason: str) -> bool:
    with _review_handler_lock:
        handler = _review_handler

    if handler is None:
        return False

    try:
        return bool(handler(CommandReviewRequest(kind=kind, payload=payload, reason=reason)))
    except Exception:  # noqa: BLE001
        return False
