from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path


@dataclass(frozen=True)
class MemoryChunk:
    """一条可被检索的长期记忆片段。"""

    text: str


def _tokenize(text: str) -> set[str]:
    """将文本切分为去重 token，用于简单关键词召回。"""
    normalized = text.lower()
    word_tokens = set(re.findall(r"[a-z0-9]+", normalized))
    cjk_tokens = {ch for ch in normalized if "\u4e00" <= ch <= "\u9fff"}
    return word_tokens | cjk_tokens


def _split_chunks(raw_text: str) -> list[MemoryChunk]:
    """按空行切分记忆块，跳过标题行。"""
    chunks: list[MemoryChunk] = []
    for block in re.split(r"\n\s*\n", raw_text):
        cleaned = block.strip()
        if not cleaned:
            continue

        content_lines = [line.strip() for line in cleaned.splitlines() if line.strip() and not line.strip().startswith("#")]
        content = " ".join(content_lines).strip()
        if not content:
            continue

        chunks.append(MemoryChunk(text=content))
    return chunks


def load_memory_chunks(memory_file: Path) -> list[MemoryChunk]:
    """从本地文件加载只读记忆。"""
    if not memory_file.exists():
        return []

    raw_text = memory_file.read_text(encoding="utf-8")
    return _split_chunks(raw_text)


def retrieve_memory_context(query: str, chunks: list[MemoryChunk], top_k: int = 3) -> str:
    """根据 query 检索最相关记忆并拼接为上下文字符串。"""
    if not chunks:
        return ""

    query_tokens = _tokenize(query)
    if not query_tokens:
        return ""

    scored: list[tuple[int, MemoryChunk]] = []
    for chunk in chunks:
        overlap = len(query_tokens & _tokenize(chunk.text))
        if overlap > 0:
            scored.append((overlap, chunk))

    if not scored:
        return ""

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[: max(1, top_k)]

    lines = [f"- {item.text}" for _, item in selected]
    return "\n".join(lines)
