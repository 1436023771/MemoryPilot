from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

from app.chains import build_qa_chain
from app.config import get_settings
from app.read_only_memory import load_memory_chunks, retrieve_memory_context
from app.sqlite_memory import (
    retrieve_memory_context_hybrid_from_sqlite,
    write_facts_to_sqlite,
)
from app.write_memory import append_memory_facts, extract_candidate_facts


class ChatWindow:
    """简单桌面对话窗口，默认启用长短期记忆。"""

    def __init__(self) -> None:
        self.session_id = "gui-default"
        self.use_rag = True
        self.write_memory = True
        self.top_k = 3

        # 默认使用 SQLite 后端，保留 txt 兼容路径。
        self.memory_backend = "sqlite"
        self.memory_db = Path("memory/long_term_memory.db")
        self.memory_file = Path("memory/long_term_memory.txt")
        self.memory_chunks = load_memory_chunks(self.memory_file)

        settings = get_settings()
        self.chain = build_qa_chain(settings)

        self.root = tk.Tk()
        self.root.title("Agent Chat Demo")
        self.root.geometry("900x620")

        self._build_layout()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        status_text = (
            f"session_id={self.session_id} | backend={self.memory_backend} "
            f"| rag=on | write_memory=on"
        )
        status = ttk.Label(container, text=status_text)
        status.pack(anchor=tk.W, pady=(0, 8))

        self.chat_box = scrolledtext.ScrolledText(
            container,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Menlo", 12),
        )
        self.chat_box.pack(fill=tk.BOTH, expand=True)

        self.chat_box.tag_configure("you", foreground="#1f6feb")
        self.chat_box.tag_configure("assistant", foreground="#0f5132")
        self.chat_box.tag_configure("memory", foreground="#6a737d")
        self.chat_box.tag_configure("error", foreground="#b42318")

        input_row = ttk.Frame(container)
        input_row.pack(fill=tk.X, pady=(8, 0))

        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(input_row, textvariable=self.input_var)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind("<Return>", self._on_send)

        self.send_btn = ttk.Button(input_row, text="发送", command=self._on_send)
        self.send_btn.pack(side=tk.LEFT, padx=(8, 0))

        self._append_message(
            "assistant",
            "你好，我已经开启长期记忆（RAG）和短期会话记忆。你可以直接开始对话。",
        )

    def _append_message(self, role: str, text: str) -> None:
        self.chat_box.configure(state=tk.NORMAL)
        tag = role
        prefix_map = {
            "you": "You",
            "assistant": "Assistant",
            "memory": "Memory",
            "error": "Error",
        }
        prefix = prefix_map.get(role, "Info")
        self.chat_box.insert(tk.END, f"{prefix}: {text}\n\n", tag)
        self.chat_box.configure(state=tk.DISABLED)
        self.chat_box.see(tk.END)

    def _build_retrieved_context(self, question: str) -> str:
        if not self.use_rag:
            return ""

        if self.memory_backend == "sqlite":
            return retrieve_memory_context_hybrid_from_sqlite(
                self.memory_db,
                question,
                top_k=self.top_k,
            )

        return retrieve_memory_context(question, self.memory_chunks, top_k=self.top_k)

    def _write_long_term_memory(self, question: str) -> list[str]:
        if not self.write_memory:
            return []

        candidate_facts = extract_candidate_facts(question)
        if self.memory_backend == "sqlite":
            return write_facts_to_sqlite(self.memory_db, candidate_facts)

        written = append_memory_facts(self.memory_file, candidate_facts)
        self.memory_chunks = load_memory_chunks(self.memory_file)
        return written

    def _on_send(self, _event=None) -> None:
        user_input = self.input_var.get().strip()
        if not user_input:
            return

        self.input_var.set("")
        self._append_message("you", user_input)
        self.send_btn.configure(state=tk.DISABLED)

        threading.Thread(target=self._process_turn, args=(user_input,), daemon=True).start()

    def _process_turn(self, user_input: str) -> None:
        try:
            retrieved_context = self._build_retrieved_context(user_input)
            response = self.chain.invoke(
                {"question": user_input, "retrieved_context": retrieved_context},
                config={"configurable": {"session_id": self.session_id}},
            )
            written_facts = self._write_long_term_memory(user_input)
            self.root.after(0, self._on_turn_finished, str(response), written_facts, None)
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, self._on_turn_finished, "", [], str(exc))

    def _on_turn_finished(self, response: str, written_facts: list[str], error: str | None) -> None:
        if error:
            self._append_message("error", error)
            self.send_btn.configure(state=tk.NORMAL)
            self.input_entry.focus_set()
            return

        self._append_message("assistant", response)
        if written_facts:
            self._append_message("memory", "wrote facts:")
            for fact in written_facts:
                self._append_message("memory", f"- {fact}")

        self.send_btn.configure(state=tk.NORMAL)
        self.input_entry.focus_set()

    def run(self) -> None:
        self.input_entry.focus_set()
        self.root.mainloop()


def main() -> None:
    app = ChatWindow()
    app.run()


if __name__ == "__main__":
    main()
