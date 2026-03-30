import argparse
from pathlib import Path

from app.agents.chains import build_qa_chain
from app.agents.chains import get_session_history
from app.core.config import get_settings
from app.memory.read_only_memory import load_memory_chunks, retrieve_memory_context
from app.memory.sqlite_memory import (
    load_memory_chunks_from_sqlite,
    retrieve_memory_context_hybrid_from_sqlite,
    write_facts_to_sqlite,
)
from app.memory.write_memory import (
    append_memory_facts,
    extract_candidate_facts,
    extract_candidate_facts_from_dialogue,
)


def parse_args() -> argparse.Namespace:
    # 命令行参数：支持单轮提问、交互模式和会话 ID。
    parser = argparse.ArgumentParser(description="Run the LangChain QA demo with short-term memory")
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--session-id", default="default", help="Session id used for chat memory")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive multi-turn chat (type 'exit' to quit)",
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Enable read-only long-term memory retrieval before each response",
    )
    parser.add_argument(
        "--memory-file",
        default="memory/long_term_memory.txt",
        help="Long-term memory text file path (used when --memory-backend=txt)",
    )
    parser.add_argument(
        "--memory-backend",
        choices=["sqlite", "txt"],
        default="sqlite",
        help="Long-term memory storage backend",
    )
    parser.add_argument(
        "--memory-db",
        default="memory/long_term_memory.db",
        help="SQLite database path (used when --memory-backend=sqlite)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many memory chunks to retrieve when RAG is enabled",
    )
    parser.add_argument(
        "--write-memory",
        action="store_true",
        help="Extract facts from user input and append them into long-term memory file",
    )
    parser.add_argument(
        "--show-memory-write",
        action="store_true",
        help="Print facts written to long-term memory in current turn",
    )
    return parser.parse_args()


def _build_retrieved_context(
    question: str,
    memory_chunks,
    use_rag: bool,
    top_k: int,
    memory_backend: str,
    memory_db: Path,
) -> str:
    """按需从只读记忆中检索上下文。"""
    if not use_rag:
        return ""

    # 第三阶段：SQLite 默认采用混合检索（关键词 + 向量相似度）。
    if memory_backend == "sqlite":
        return retrieve_memory_context_hybrid_from_sqlite(memory_db, question, top_k=top_k)

    return retrieve_memory_context(question, memory_chunks, top_k=top_k)


def _load_memory_chunks(memory_backend: str, memory_file: Path, memory_db: Path):
    """按后端类型加载长期记忆片段。"""
    if memory_backend == "sqlite":
        return load_memory_chunks_from_sqlite(memory_db)
    return load_memory_chunks(memory_file)


def _maybe_write_long_term_memory(
    question: str,
    session_id: str,
    memory_file: Path,
    memory_db: Path,
    memory_backend: str,
    write_memory: bool,
    show_memory_write: bool,
):
    """按需将本轮用户输入提取为长期记忆并写入文件。"""
    if not write_memory:
        return

    history = get_session_history(session_id)
    candidate_facts = extract_candidate_facts_from_dialogue(history.messages, max_turns=6)
    if not candidate_facts:
        candidate_facts = extract_candidate_facts(question)

    if memory_backend == "sqlite":
        written_facts = write_facts_to_sqlite(memory_db, candidate_facts)
    else:
        written_facts = append_memory_facts(memory_file, candidate_facts)

    if show_memory_write and written_facts:
        print("[Memory] wrote facts:")
        for fact in written_facts:
            print(f"- {fact}")


def run_single_turn(
    question: str,
    session_id: str,
    use_rag: bool,
    memory_chunks,
    top_k: int,
    write_memory: bool,
    memory_file: Path,
    memory_db: Path,
    memory_backend: str,
    show_memory_write: bool,
) -> None:
    # 单轮模式：构建链后仅调用一次。
    settings = get_settings()
    chain = build_qa_chain(settings)

    retrieved_context = _build_retrieved_context(
        question=question,
        memory_chunks=memory_chunks,
        use_rag=use_rag,
        top_k=top_k,
        memory_backend=memory_backend,
        memory_db=memory_db,
    )

    # 通过 configurable.session_id 指定记忆上下文。
    response = chain.invoke(
        {"question": question, "retrieved_context": retrieved_context},
        config={"configurable": {"session_id": session_id}},
    )
    print(str(response))

    _maybe_write_long_term_memory(
        question=question,
        session_id=session_id,
        memory_file=memory_file,
        memory_db=memory_db,
        memory_backend=memory_backend,
        write_memory=write_memory,
        show_memory_write=show_memory_write,
    )


def run_interactive(
    session_id: str,
    use_rag: bool,
    memory_chunks,
    top_k: int,
    write_memory: bool,
    memory_file: Path,
    memory_db: Path,
    memory_backend: str,
    show_memory_write: bool,
) -> None:
    # 交互模式：同一进程内循环对话，可连续复用短期记忆。
    settings = get_settings()
    chain = build_qa_chain(settings)
    print(f"Interactive chat started (session_id={session_id}). Type 'exit' to quit.")

    while True:
        # 读取用户输入并处理退出命令。
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not user_input:
            continue

        # 每一轮都带上同一个 session_id，模型即可读取历史上下文。
        retrieved_context = _build_retrieved_context(
            question=user_input,
            memory_chunks=memory_chunks,
            use_rag=use_rag,
            top_k=top_k,
            memory_backend=memory_backend,
            memory_db=memory_db,
        )

        response = chain.invoke(
            {"question": user_input, "retrieved_context": retrieved_context},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"Assistant: {str(response)}")

        _maybe_write_long_term_memory(
            question=user_input,
            session_id=session_id,
            memory_file=memory_file,
            memory_db=memory_db,
            memory_backend=memory_backend,
            write_memory=write_memory,
            show_memory_write=show_memory_write,
        )

        # 写入后刷新本地缓存，使下一轮检索可读到新记忆。
        memory_chunks = _load_memory_chunks(memory_backend, memory_file, memory_db)


def main() -> None:
    # 主入口：按参数决定进入交互模式或单轮模式。
    args = parse_args()

    memory_file = Path(args.memory_file)
    memory_db = Path(args.memory_db)
    memory_chunks = _load_memory_chunks(args.memory_backend, memory_file, memory_db)

    if args.interactive:
        run_interactive(
            session_id=args.session_id,
            use_rag=args.use_rag,
            memory_chunks=memory_chunks,
            top_k=args.top_k,
            write_memory=args.write_memory,
            memory_file=memory_file,
            memory_db=memory_db,
            memory_backend=args.memory_backend,
            show_memory_write=args.show_memory_write,
        )
        return

    question = args.question or "What is LangChain used for?"
    run_single_turn(
        question=question,
        session_id=args.session_id,
        use_rag=args.use_rag,
        memory_chunks=memory_chunks,
        top_k=args.top_k,
        write_memory=args.write_memory,
        memory_file=memory_file,
        memory_db=memory_db,
        memory_backend=args.memory_backend,
        show_memory_write=args.show_memory_write,
    )


if __name__ == "__main__":
    main()
