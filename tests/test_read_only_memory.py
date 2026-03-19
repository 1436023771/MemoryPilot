from pathlib import Path

from app.read_only_memory import MemoryChunk, load_memory_chunks, retrieve_memory_context


def test_load_memory_chunks_from_file(tmp_path: Path) -> None:
    memory_file = tmp_path / "memory.txt"
    memory_file.write_text(
        "# 标题\n\n用户偏好：中文输出\n\n目标：实现长期记忆\n",
        encoding="utf-8",
    )

    chunks = load_memory_chunks(memory_file)

    assert len(chunks) == 2
    assert chunks[0].text == "用户偏好：中文输出"


def test_retrieve_memory_context_returns_top_hits() -> None:
    chunks = [
        MemoryChunk(text="用户偏好：中文回答"),
        MemoryChunk(text="项目目标：实现长期记忆"),
        MemoryChunk(text="技术栈：LangChain"),
    ]

    context = retrieve_memory_context("我想要中文回答", chunks, top_k=2)

    assert "用户偏好：中文回答" in context


def test_load_memory_chunks_keeps_content_below_headers(tmp_path: Path) -> None:
    memory_file = tmp_path / "memory_with_headers.txt"
    memory_file.write_text(
        "# 用户身份\n我是小李。\n\n# 目标\n实现长期记忆。\n",
        encoding="utf-8",
    )

    chunks = load_memory_chunks(memory_file)
    joined = "\n".join(chunk.text for chunk in chunks)

    assert "我是小李。" in joined
    assert "实现长期记忆。" in joined
