from __future__ import annotations

import re

from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage

from app.agents.mcp.llmlingua_client import compress_history_via_llmlingua_mcp, compress_text_via_llmlingua_mcp
from app.config.execution import llmlingua_mcp_enabled
from app.config.langgraph import max_history_tokens


def _estimate_text_tokens(text: str) -> int:
    """Rough token estimate without external tokenizer dependency."""
    raw = str(text or "")
    if not raw:
        return 0

    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", raw))
    latin_words = re.findall(r"[a-zA-Z0-9_]+", raw)
    latin_chars = sum(len(w) for w in latin_words)
    symbol_chars = len(re.findall(r"[^\w\s\u4e00-\u9fff]", raw))
    latin_tokens = max(1, latin_chars // 4) if latin_chars > 0 else 0

    # CJK roughly maps close to one token per character; latin text is denser.
    return cjk_count + latin_tokens + max(0, symbol_chars // 6)


def _estimate_message_tokens(msg: BaseMessage) -> int:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return _estimate_text_tokens(content)
    if isinstance(content, list):
        joined = " ".join(str(item) for item in content)
        return _estimate_text_tokens(joined)
    return _estimate_text_tokens(str(content))


def _history_token_limit() -> int:
    return max(300, max_history_tokens())


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[。！？!?；;\.\n])\s*", text or "") if p and p.strip()]
    return parts


def _extract_key_tokens(text: str) -> list[str]:
    """Extract key tokens that should be preserved across compression."""
    raw = str(text or "")
    if not raw:
        return []

    patterns = [
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",  # date-like 2026-03-28
        r"\b\d{1,2}:\d{2}\b",  # time-like 08:30
        r"\b\d+(?:\.\d+)?%\b",  # percentage
        r"\b\d+(?:\.\d+)?(?:k|m|b|w)?\b",  # numbers
        r"\b(?:v|V)?\d+(?:\.\d+){1,3}\b",  # version-like
        r"\b[a-zA-Z_]+\s*=\s*[^\s,;，；]+",  # key=value hints
    ]

    negation_terms = ["不", "不能", "不要", "不可", "仅", "必须", "except", "unless", "not", "only"]

    seen: set[str] = set()
    kept: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, raw):
            token = m.group(0).strip()
            if token and token not in seen:
                seen.add(token)
                kept.append(token)

    for word in negation_terms:
        if word in raw and word not in seen:
            seen.add(word)
            kept.append(word)

    return kept[:12]


def _contains_key_token(text: str, token: str) -> bool:
    if not token:
        return False
    return token in (text or "")


def _compress_text_head_tail_to_token_budget(text: str, max_tokens: int) -> str:
    """Fallback head-tail compression with hard budget guarantee."""
    raw = str(text or "")
    if not raw:
        return raw
    if max_tokens <= 0:
        return ""

    estimated = _estimate_text_tokens(raw)
    if estimated <= max_tokens:
        return raw

    keep_ratio = max(0.08, min(1.0, max_tokens / max(estimated, 1)))
    keep_chars = max(6, int(len(raw) * keep_ratio * 0.95))
    if keep_chars >= len(raw):
        candidate = raw
    elif keep_chars <= 12:
        candidate = raw[:keep_chars].rstrip()
    else:
        head_len = int(keep_chars * 0.7)
        tail_len = max(0, keep_chars - head_len)
        head = raw[:head_len].rstrip()
        tail = raw[-tail_len:].lstrip() if tail_len > 0 else ""
        candidate = f"{head} ... {tail}" if tail else head

    if _estimate_text_tokens(candidate) <= max_tokens:
        return candidate

    # Hard fallback: shrink aggressively until budget is satisfied.
    hard_chars = min(len(raw), max_tokens)
    hard = raw[: max(1, hard_chars)].rstrip()
    while hard and _estimate_text_tokens(hard) > max_tokens:
        next_len = max(1, len(hard) - 1)
        hard = hard[:next_len].rstrip()
        if next_len == 1:
            break
    return hard


def _extractive_compress_with_keys(text: str, max_tokens: int, key_tokens: list[str]) -> str:
    """Extractive sentence compression with key-token-aware scoring."""
    sentences = _split_sentences(text)
    if not sentences:
        return _compress_text_head_tail_to_token_budget(text, max_tokens)

    scored: list[tuple[float, int, str, int]] = []
    for idx, sent in enumerate(sentences):
        sent_tokens = _estimate_text_tokens(sent)
        if sent_tokens <= 0:
            continue
        key_hits = sum(1 for k in key_tokens if _contains_key_token(sent, k))
        has_digit = 1.0 if re.search(r"\d", sent) else 0.0
        has_negation = 1.0 if any(x in sent for x in ("不", "不能", "不要", "不可", "仅", "必须", "not", "only")) else 0.0
        density = min(1.0, sent_tokens / 24)
        score = key_hits * 3.0 + has_digit * 0.8 + has_negation * 1.0 + density * 0.4
        scored.append((score, idx, sent, sent_tokens))

    if not scored:
        return _compress_text_head_tail_to_token_budget(text, max_tokens)

    scored.sort(key=lambda x: (x[0], -x[3]), reverse=True)

    selected_idx: set[int] = set()
    used = 0
    for _score, idx, sent, sent_tokens in scored:
        if sent_tokens > max_tokens:
            continue
        if used + sent_tokens > max_tokens:
            continue
        selected_idx.add(idx)
        used += sent_tokens
        if used >= int(max_tokens * 0.9):
            break

    if not selected_idx:
        best = scored[0][2]
        return _compress_text_head_tail_to_token_budget(best, max_tokens)

    ordered = [sentences[i] for i in sorted(selected_idx)]
    candidate = " ".join(ordered).strip()
    if _estimate_text_tokens(candidate) <= max_tokens:
        return candidate
    return _compress_text_head_tail_to_token_budget(candidate, max_tokens)


def _inject_missing_key_tokens(base: str, key_tokens: list[str], max_tokens: int) -> str:
    """Try appending missing key tokens while staying under budget."""
    result = base.strip()
    for token in key_tokens:
        if _contains_key_token(result, token):
            continue
        trial = f"{result} [{token}]" if result else token
        if _estimate_text_tokens(trial) <= max_tokens:
            result = trial
    return result


def _compress_text_to_token_budget(text: str, max_tokens: int) -> str:
    """Hybrid compression: key-token preservation + local extractive summarization."""
    raw = str(text or "")
    if not raw:
        return raw
    if max_tokens <= 0:
        return ""

    if llmlingua_mcp_enabled():
        key_tokens = _extract_key_tokens(raw)
        mcp_compressed = compress_text_via_llmlingua_mcp(raw, int(max_tokens), preserve_keywords=key_tokens)
        if isinstance(mcp_compressed, str) and mcp_compressed.strip():
            return mcp_compressed

    estimated = _estimate_text_tokens(raw)
    if estimated <= max_tokens:
        return raw

    key_tokens = _extract_key_tokens(raw)
    extracted = _extractive_compress_with_keys(raw, max_tokens, key_tokens)
    with_keys = _inject_missing_key_tokens(extracted, key_tokens, max_tokens)

    if _estimate_text_tokens(with_keys) <= max_tokens:
        return with_keys

    return _compress_text_head_tail_to_token_budget(with_keys, max_tokens)


def _copy_message_with_content(msg: BaseMessage, content: str) -> BaseMessage:
    """Clone message while preserving role-specific metadata."""
    if isinstance(msg, HumanMessage):
        return HumanMessage(content=content, additional_kwargs=getattr(msg, "additional_kwargs", {}))
    if isinstance(msg, AIMessage):
        return AIMessage(content=content, additional_kwargs=getattr(msg, "additional_kwargs", {}))
    if isinstance(msg, SystemMessage):
        return SystemMessage(content=content, additional_kwargs=getattr(msg, "additional_kwargs", {}))
    if isinstance(msg, ToolMessage):
        return ToolMessage(
            content=content,
            tool_call_id=str(getattr(msg, "tool_call_id", "") or ""),
            additional_kwargs=getattr(msg, "additional_kwargs", {}),
        )

    try:
        return msg.__class__(content=content)
    except Exception:  # noqa: BLE001
        return HumanMessage(content=content)


def _message_role_name(msg: BaseMessage) -> str:
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, AIMessage):
        return "assistant"
    if isinstance(msg, SystemMessage):
        return "system"
    if isinstance(msg, ToolMessage):
        return "tool"
    return "user"


def _compress_history_by_token_budget(history: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """Compress message contents to fit budget while keeping all message turns."""
    if not history:
        return []
    if max_tokens <= 0:
        return [_copy_message_with_content(msg, "") for msg in history]

    if llmlingua_mcp_enabled():
        role_payload: list[dict[str, str]] = []
        for msg in history:
            role = _message_role_name(msg)

            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                content = str(content)
            role_payload.append({"role": role, "content": content})

        compressed_payload = compress_history_via_llmlingua_mcp(role_payload, int(max_tokens))
        if isinstance(compressed_payload, list) and len(compressed_payload) == len(history):
            roles_match = True
            for msg, payload in zip(history, compressed_payload, strict=False):
                payload_role = str(payload.get("role", "") or "")
                if payload_role != _message_role_name(msg):
                    roles_match = False
                    break

            if not roles_match:
                compressed_payload = None

        if isinstance(compressed_payload, list) and len(compressed_payload) == len(history):
            compressed_msgs: list[BaseMessage] = []
            for msg, payload in zip(history, compressed_payload, strict=False):
                new_content = str(payload.get("content", "") or "")
                compressed_msgs.append(_copy_message_with_content(msg, new_content))
            return compressed_msgs

    costs = [_estimate_message_tokens(msg) for msg in history]
    total = sum(costs)
    if total <= max_tokens:
        return history

    n = len(history)
    # Keep at least one estimated token budget per message to preserve turns.
    minima = [1] * n
    budget_left = max(0, max_tokens - sum(minima))
    capacities = [max(0, c - 1) for c in costs]

    # Prefer recent messages by assigning larger weights to later indices.
    weights = [i + 1 for i in range(n)]
    weight_sum = sum(weights) or 1
    extras = [0] * n

    for i in range(n):
        if capacities[i] <= 0 or budget_left <= 0:
            continue
        alloc = int(budget_left * (weights[i] / weight_sum))
        use = min(capacities[i], alloc)
        extras[i] = use

    used_extra = sum(extras)
    remaining = max(0, budget_left - used_extra)
    for i in range(n - 1, -1, -1):
        if remaining <= 0:
            break
        spare = capacities[i] - extras[i]
        if spare <= 0:
            continue
        take = min(spare, remaining)
        extras[i] += take
        remaining -= take

    targets = [minima[i] + extras[i] for i in range(n)]
    compressed: list[BaseMessage] = []
    for msg, target in zip(history, targets, strict=False):
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            content = str(content)
        new_content = _compress_text_to_token_budget(content, target)
        compressed.append(_copy_message_with_content(msg, new_content))
    return compressed


__all__ = [
    "_history_token_limit",
    "_compress_text_to_token_budget",
    "_compress_history_by_token_budget",
]
