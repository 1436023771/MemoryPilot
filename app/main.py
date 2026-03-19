import argparse

from app.chains import build_qa_chain
from app.config import get_settings


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
    return parser.parse_args()


def run_single_turn(question: str, session_id: str) -> None:
    # 单轮模式：构建链后仅调用一次。
    settings = get_settings()
    chain = build_qa_chain(settings)

    # 通过 configurable.session_id 指定记忆上下文。
    response = chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )
    print(response)


def run_interactive(session_id: str) -> None:
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
        response = chain.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"Assistant: {response}")


def main() -> None:
    # 主入口：按参数决定进入交互模式或单轮模式。
    args = parse_args()
    if args.interactive:
        run_interactive(session_id=args.session_id)
        return

    question = args.question or "What is LangChain used for?"
    run_single_turn(question=question, session_id=args.session_id)


if __name__ == "__main__":
    main()
