from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

from app.chains import build_qa_chain
from app.chains import get_session_history
from app.config import get_settings
from app.read_only_memory import load_memory_chunks, retrieve_memory_context
from app.sqlite_memory import (
    retrieve_memory_context_hybrid_from_sqlite,
    write_facts_to_sqlite,
)
from app.write_memory import (
    append_memory_facts,
    extract_candidate_facts,
    extract_candidate_facts_from_dialogue,
)


class ChatWindow:
    """简单桌面对话窗口，默认启用长短期记忆。"""

    def __init__(self) -> None:
        self.session_id = "gui-default"
        self.use_rag = True
        self.write_memory = True
        self.top_k = 3
        self.turn_count = 0  # 用于追踪轮次编号

        # 默认使用 SQLite 后端，保留 txt 兼容路径。
        self.memory_backend = "sqlite"
        self.memory_db = Path("memory/long_term_memory.db")
        self.memory_file = Path("memory/long_term_memory.txt")
        self.memory_chunks = load_memory_chunks(self.memory_file)

        settings = get_settings()
        self.chain = build_qa_chain(settings)

        self.root = tk.Tk()
        self.root.title("Agent Chat Demo")
        self.root.geometry("1400x620")

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

        # 主布局：左侧对话窗口，右侧侧边栏
        main_paned = ttk.PanedWindow(container, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # 左侧：对话框
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=2)

        self.chat_box = scrolledtext.ScrolledText(
            left_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Menlo", 11),
        )
        self.chat_box.pack(fill=tk.BOTH, expand=True)

        self.chat_box.tag_configure("you", foreground="#1f6feb")
        self.chat_box.tag_configure("assistant", foreground="#0f5132")
        self.chat_box.tag_configure("memory", foreground="#6a737d")
        self.chat_box.tag_configure("error", foreground="#b42318")

        # 右侧：侧边栏（Prompt 历史）
        sidebar_frame = ttk.LabelFrame(main_paned, text="Prompt History", padding=8)
        main_paned.add(sidebar_frame, weight=1)

        self.sidebar_box = scrolledtext.ScrolledText(
            sidebar_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Menlo", 9),
            bg="#f6f8fa",
        )
        self.sidebar_box.pack(fill=tk.BOTH, expand=True)

        self.sidebar_box.tag_configure("turn_header", foreground="#0969da", font=("Menlo", 9, "bold"))
        self.sidebar_box.tag_configure("user_query", foreground="#1f6feb")
        self.sidebar_box.tag_configure("context_header", foreground="#6f42c1", font=("Menlo", 8, "bold"))
        self.sidebar_box.tag_configure("context", foreground="#5a6268")

        # 输入区域
        input_row = ttk.Frame(container)
        input_row.pack(fill=tk.X, pady=(0, 0))

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

    def _append_to_sidebar(
        self,
        turn_num: int,
        question: str,
        context: str,
        extracted_facts: list[str],
        written_facts: list[str],
    ) -> None:
        """在侧边栏中记录完整的prompt信息"""
        self.sidebar_box.configure(state=tk.NORMAL)
        
        # 轮次标题
        self.sidebar_box.insert(tk.END, f"━━━ Turn {turn_num} ━━━\n", "turn_header")
        
        # 用户查询
        self.sidebar_box.insert(tk.END, "Query:\n", "context_header")
        self.sidebar_box.insert(tk.END, f"{question}\n\n", "user_query")
        
        # 检索上下文（如果有）
        if context.strip():
            self.sidebar_box.insert(tk.END, "Retrieved Context:\n", "context_header")
            self.sidebar_box.insert(tk.END, f"{context}\n\n", "context")
        else:
            self.sidebar_box.insert(tk.END, "[No context retrieved]\n\n", "context")

        # 本轮提取结果
        self.sidebar_box.insert(tk.END, "Extracted Facts:\n", "context_header")
        if extracted_facts:
            for fact in extracted_facts:
                self.sidebar_box.insert(tk.END, f"- {fact}\n", "context")
            self.sidebar_box.insert(tk.END, "\n", "context")
        else:
            self.sidebar_box.insert(tk.END, "[No facts extracted]\n\n", "context")

        # 实际写入结果
        self.sidebar_box.insert(tk.END, "Written Facts:\n", "context_header")
        if written_facts:
            for fact in written_facts:
                self.sidebar_box.insert(tk.END, f"- {fact}\n", "context")
            self.sidebar_box.insert(tk.END, "\n", "context")
        else:
            self.sidebar_box.insert(tk.END, "[No new facts written]\n\n", "context")
        
        self.sidebar_box.configure(state=tk.DISABLED)
        self.sidebar_box.see(tk.END)

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

    def _write_long_term_memory(self, question: str) -> tuple[list[str], list[str]]:
        if not self.write_memory:
            return [], []

        history = get_session_history(self.session_id)
        candidate_facts = extract_candidate_facts_from_dialogue(history.messages, max_turns=6)
        if not candidate_facts:
            candidate_facts = extract_candidate_facts(question)

        extracted_texts = [fact.text for fact in candidate_facts]
        if self.memory_backend == "sqlite":
            written = write_facts_to_sqlite(self.memory_db, candidate_facts)
            return extracted_texts, written

        written = append_memory_facts(self.memory_file, candidate_facts)
        self.memory_chunks = load_memory_chunks(self.memory_file)
        return extracted_texts, written

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
            self.turn_count += 1
            retrieved_context = self._build_retrieved_context(user_input)
            response = self.chain.invoke(
                {"question": user_input, "retrieved_context": retrieved_context},
                config={"configurable": {"session_id": self.session_id}},
            )
            extracted_facts, written_facts = self._write_long_term_memory(user_input)
            self.root.after(
                0,
                self._on_turn_finished,
                str(response),
                extracted_facts,
                written_facts,
                None,
                user_input,
                retrieved_context,
            )
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, self._on_turn_finished, "", [], [], str(exc), user_input, "")

    def _on_turn_finished(
        self,
        response: str,
        extracted_facts: list[str],
        written_facts: list[str],
        error: str | None,
        user_input: str = "",
        retrieved_context: str = "",
    ) -> None:
        # 记录到侧边栏
        self._append_to_sidebar(
            self.turn_count,
            user_input,
            retrieved_context,
            extracted_facts,
            written_facts,
        )
        
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
