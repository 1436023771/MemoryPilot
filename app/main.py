import argparse
from pathlib import Path

from app.chains import build_qa_chain
from app.config import get_settings
from app.read_only_memory import load_memory_chunks, retrieve_memory_context
from app.write_memory import append_memory_facts, extract_candidate_facts


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
        help="Read-only long-term memory file path",
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


def _build_retrieved_context(question: str, memory_chunks, use_rag: bool, top_k: int) -> str:
    """按需从只读记忆中检索上下文。"""
    if not use_rag:
        return ""
    return retrieve_memory_context(question, memory_chunks, top_k=top_k)


def _maybe_write_long_term_memory(
    question: str,
    memory_file: Path,
    write_memory: bool,
    show_memory_write: bool,
):
    """按需将本轮用户输入提取为长期记忆并写入文件。"""
    if not write_memory:
        return

    candidate_facts = extract_candidate_facts(question)
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
    )

    # 通过 configurable.session_id 指定记忆上下文。
    response = chain.invoke(
        {"question": question, "retrieved_context": retrieved_context},
        config={"configurable": {"session_id": session_id}},
    )
    print(response)

    _maybe_write_long_term_memory(
        question=question,
        memory_file=memory_file,
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
        )

        response = chain.invoke(
            {"question": user_input, "retrieved_context": retrieved_context},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"Assistant: {response}")

        _maybe_write_long_term_memory(
            question=user_input,
            memory_file=memory_file,
            write_memory=write_memory,
            show_memory_write=show_memory_write,
        )

        # 写入后刷新本地缓存，使下一轮检索可读到新记忆。
        memory_chunks = load_memory_chunks(memory_file)


def main() -> None:
    # 主入口：按参数决定进入交互模式或单轮模式。
    args = parse_args()

    memory_file = Path(args.memory_file)
    memory_chunks = load_memory_chunks(memory_file)

    if args.interactive:
        run_interactive(
            session_id=args.session_id,
            use_rag=args.use_rag,
            memory_chunks=memory_chunks,
            top_k=args.top_k,
            write_memory=args.write_memory,
            memory_file=memory_file,
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
        show_memory_write=args.show_memory_write,
    )


if __name__ == "__main__":
    main()
