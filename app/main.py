import argparse
from pathlib import Path

from app.chains import build_qa_chain
from app.config import get_settings
from app.read_only_memory import load_memory_chunks, retrieve_memory_context


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
    return parser.parse_args()


def _build_retrieved_context(question: str, memory_chunks, use_rag: bool, top_k: int) -> str:
    """按需从只读记忆中检索上下文。"""
    if not use_rag:
        return ""
    return retrieve_memory_context(question, memory_chunks, top_k=top_k)


def run_single_turn(question: str, session_id: str, use_rag: bool, memory_chunks, top_k: int) -> None:
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


def run_interactive(session_id: str, use_rag: bool, memory_chunks, top_k: int) -> None:
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


def main() -> None:
    # 主入口：按参数决定进入交互模式或单轮模式。
    args = parse_args()

    memory_chunks = load_memory_chunks(Path(args.memory_file))

    if args.interactive:
        run_interactive(
            session_id=args.session_id,
            use_rag=args.use_rag,
            memory_chunks=memory_chunks,
            top_k=args.top_k,
        )
        return

    question = args.question or "What is LangChain used for?"
    run_single_turn(
        question=question,
        session_id=args.session_id,
        use_rag=args.use_rag,
        memory_chunks=memory_chunks,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
