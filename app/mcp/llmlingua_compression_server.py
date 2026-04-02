from __future__ import annotations

import argparse
import importlib
import os
import re
from pathlib import Path
from typing import Any

try:
    _fastmcp_mod = importlib.import_module("mcp.server.fastmcp")
    FastMCP = getattr(_fastmcp_mod, "FastMCP")
except Exception:  # noqa: BLE001
    FastMCP = None  # type: ignore[assignment]


_RUNTIME: dict[str, Any] = {
    "initialized": False,
    "tokenizer": None,
    "compressor": None,
    "init_error": "",
}


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_model_name() -> str:
    raw = os.getenv("LLMLINGUA_MODEL_NAME", "Qwen/Qwen2-1.5B-Instruct").strip()
    return raw or "Qwen/Qwen2-1.5B-Instruct"


def _env_model_path() -> Path:
    raw = os.getenv("LLMLINGUA_MODEL_PATH", "./models/qwen-1.5b").strip() or "./models/qwen-1.5b"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _workspace_root() / path
    return path.resolve()


def _env_compression_rate() -> float:
    raw = os.getenv("LLMLINGUA_COMPRESSION_RATE", "0.5").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 0.5
    return max(0.1, min(0.9, value))


def _estimate_text_tokens(text: str) -> int:
    raw = str(text or "")
    if not raw:
        return 0

    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", raw))
    latin_words = re.findall(r"[a-zA-Z0-9_]+", raw)
    latin_chars = sum(len(w) for w in latin_words)
    symbol_chars = len(re.findall(r"[^\w\s\u4e00-\u9fff]", raw))
    latin_tokens = max(1, latin_chars // 4) if latin_chars > 0 else 0
    return cjk_count + latin_tokens + max(0, symbol_chars // 6)


def _fallback_compress(text: str, target_tokens: int) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    if target_tokens <= 0:
        return ""

    estimated = _estimate_text_tokens(raw)
    if estimated <= target_tokens:
        return raw

    keep_ratio = max(0.08, min(1.0, target_tokens / max(estimated, 1)))
    keep_chars = max(8, int(len(raw) * keep_ratio * 0.95))
    if keep_chars >= len(raw):
        return raw
    if keep_chars <= 16:
        return raw[:keep_chars].rstrip()

    head_len = int(keep_chars * 0.72)
    tail_len = max(0, keep_chars - head_len)
    head = raw[:head_len].rstrip()
    tail = raw[-tail_len:].lstrip() if tail_len else ""
    candidate = f"{head} ... {tail}" if tail else head

    # Hard bound guard.
    while candidate and _estimate_text_tokens(candidate) > target_tokens and len(candidate) > 1:
        candidate = candidate[:-1].rstrip()
    return candidate


def _init_runtime() -> None:
    if _RUNTIME["initialized"]:
        return

    model_name = _env_model_name()
    model_path = _env_model_path()
    model_path.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    compressor = None
    init_error = ""

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(model_path),
            trust_remote_code=True,
        )
    except Exception as exc:  # noqa: BLE001
        init_error = f"tokenizer init failed: {exc}"

    try:
        llmlingua_mod = importlib.import_module("llmlingua")
        PromptCompressor = getattr(llmlingua_mod, "PromptCompressor")

        # Force CPU mode so the shared server works on machines without CUDA.
        compressor = PromptCompressor(model_name=model_name, device_map="cpu")
    except Exception as exc:  # noqa: BLE001
        if init_error:
            init_error = f"{init_error}; compressor init failed: {exc}"
        else:
            init_error = f"compressor init failed: {exc}"

    _RUNTIME["tokenizer"] = tokenizer
    _RUNTIME["compressor"] = compressor
    _RUNTIME["init_error"] = init_error
    _RUNTIME["initialized"] = True


def _token_count(text: str) -> int:
    _init_runtime()
    tokenizer = _RUNTIME.get("tokenizer")
    if tokenizer is None:
        return _estimate_text_tokens(text)
    try:
        tokens = tokenizer.encode(str(text or ""), add_special_tokens=False)
        return int(len(tokens))
    except Exception:  # noqa: BLE001
        return _estimate_text_tokens(text)


def _compress_with_llmlingua(text: str, target_tokens: int, preserve_keywords: list[str]) -> str:
    _init_runtime()
    compressor = _RUNTIME.get("compressor")
    if compressor is None:
        return _fallback_compress(text, target_tokens)

    original = str(text or "")
    if not original:
        return ""

    original_tokens = max(1, _token_count(original))
    target_rate = max(0.05, min(0.95, target_tokens / original_tokens))

    # Different llmlingua versions expose slightly different compress APIs.
    # Try a few known call shapes in order and fall back safely.
    attempts: list[tuple[list[Any], dict[str, Any]]] = [
        (
            [original],
            {
                "target_token": int(target_tokens),
                "target_compression": target_rate,
                "preserve_keywords": preserve_keywords,
            },
        ),
        (
            [original],
            {
                "target_token": int(target_tokens),
                "preserve_keywords": preserve_keywords,
            },
        ),
        (
            [original],
            {
                "target_compression": target_rate,
                "preserve_keywords": preserve_keywords,
            },
        ),
    ]

    for args, kwargs in attempts:
        try:
            result = compressor.compress_prompt(*args, **kwargs)
        except TypeError:
            continue
        except Exception:
            continue

        if isinstance(result, dict):
            for key in ("compressed_prompt", "prompt", "text"):
                val = result.get(key)
                if isinstance(val, str) and val.strip():
                    return val
        if isinstance(result, str) and result.strip():
            return result

    return _fallback_compress(original, target_tokens)


def _score_message_for_reorder(content: str, idx: int, n: int) -> float:
    text = str(content or "")
    recency = (idx + 1) / max(1, n)
    has_digit = 1.0 if re.search(r"\d", text) else 0.0
    has_negation = 1.0 if any(x in text for x in ("不", "不能", "不要", "不可", "仅", "必须", "not", "only")) else 0.0
    density = min(1.0, _estimate_text_tokens(text) / 40)
    return recency * 1.5 + has_digit * 0.8 + has_negation * 1.0 + density * 0.3


def _allocate_tokens(costs: list[int], target_tokens: int) -> list[int]:
    if not costs:
        return []
    n = len(costs)
    minima = [1] * n
    budget_left = max(0, int(target_tokens) - sum(minima))
    capacities = [max(0, c - 1) for c in costs]

    weights = [i + 1 for i in range(n)]
    weight_sum = sum(weights) or 1
    extras = [0] * n

    for i in range(n):
        if capacities[i] <= 0 or budget_left <= 0:
            continue
        alloc = int(budget_left * (weights[i] / weight_sum))
        extras[i] = min(capacities[i], alloc)

    used = sum(extras)
    remaining = max(0, budget_left - used)
    for i in range(n - 1, -1, -1):
        if remaining <= 0:
            break
        spare = capacities[i] - extras[i]
        if spare <= 0:
            continue
        take = min(spare, remaining)
        extras[i] += take
        remaining -= take

    return [minima[i] + extras[i] for i in range(n)]


if FastMCP is not None:
    mcp = FastMCP("llmlingua-compression")

    @mcp.tool()
    def warmup_model() -> dict[str, Any]:
        _init_runtime()
        return {
            "status": "ok",
            "initialized": bool(_RUNTIME["initialized"]),
            "tokenizer_ready": _RUNTIME.get("tokenizer") is not None,
            "compressor_ready": _RUNTIME.get("compressor") is not None,
            "init_error": str(_RUNTIME.get("init_error", "")),
        }

    @mcp.tool()
    def health_check() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_name": _env_model_name(),
            "model_path": str(_env_model_path()),
            "compression_rate": _env_compression_rate(),
            "initialized": bool(_RUNTIME["initialized"]),
            "tokenizer_ready": _RUNTIME.get("tokenizer") is not None,
            "compressor_ready": _RUNTIME.get("compressor") is not None,
            "init_error": str(_RUNTIME.get("init_error", "")),
        }

    @mcp.tool()
    def estimate_tokens(text: str) -> dict[str, int]:
        return {"token_count": _token_count(str(text or ""))}

    @mcp.tool()
    def compress_prompt(
        text: str,
        target_tokens: int,
        preserve_keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        raw = str(text or "")
        budget = max(1, int(target_tokens))
        keywords = list(preserve_keywords or [])

        compressed = _compress_with_llmlingua(raw, budget, keywords)
        return {
            "compressed_text": compressed,
            "original_tokens": _token_count(raw),
            "compressed_tokens": _token_count(compressed),
        }

    @mcp.tool()
    def reorder_context(messages: list[dict[str, Any]], target_tokens: int) -> dict[str, Any]:
        normalized: list[dict[str, str]] = []
        for item in messages or []:
            role = str((item or {}).get("role", "user") or "user")
            content = str((item or {}).get("content", "") or "")
            normalized.append({"role": role, "content": content})

        n = len(normalized)
        scored: list[tuple[float, int, dict[str, str]]] = []
        for idx, item in enumerate(normalized):
            scored.append((_score_message_for_reorder(item["content"], idx, n), idx, item))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        reordered = [item for _, _, item in scored]

        costs = [_token_count(item["content"]) for item in reordered]
        allocations = _allocate_tokens(costs, int(target_tokens))

        out: list[dict[str, str]] = []
        for item, budget in zip(reordered, allocations, strict=False):
            content = _compress_with_llmlingua(item["content"], int(budget), [])
            out.append({"role": item["role"], "content": content})

        return {
            "reordered_messages": out,
            "token_allocation": allocations,
            "total_before": sum(costs),
            "total_after": sum(_token_count(x["content"]) for x in out),
        }

    @mcp.tool()
    def compress_history(messages: list[dict[str, Any]], target_tokens: int) -> dict[str, Any]:
        data = reorder_context(messages=messages, target_tokens=target_tokens)
        return {"compressed_messages": data.get("reordered_messages", [])}


def main() -> None:
    if FastMCP is None:
        raise RuntimeError("mcp package is required. Install with: pip install mcp")

    parser = argparse.ArgumentParser(description="LLMLingua MCP compression server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http", "sse"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--path", default="/mcp")
    parser.add_argument("--preload", action="store_true", help="Preload tokenizer/compressor at server startup")
    args = parser.parse_args()

    # Configure network settings for transport-enabled runs.
    try:
        mcp.settings.host = args.host
        mcp.settings.port = int(args.port)
        mcp.settings.streamable_http_path = args.path
    except Exception:  # noqa: BLE001
        pass

    if args.preload:
        _init_runtime()

    if args.transport == "stdio":
        mcp.run()
        return

    # mcp>=1.0 run() supports transport + mount_path.
    # host/port/path come from mcp.settings.
    try:
        mcp.run(transport=args.transport, mount_path=args.path)
    except TypeError:
        mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
