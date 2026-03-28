from __future__ import annotations

import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

from dotenv import load_dotenv

from app.agents.chains import build_qa_chain
from app.agents.chains import get_session_history
from app.agents.stream_messages import StreamMessage, MessageType
from app.core.config import get_settings
from app.memory.read_only_memory import load_memory_chunks, retrieve_memory_context
from app.memory.sqlite_memory import (
    retrieve_memory_context_hybrid_from_sqlite,
    write_facts_to_sqlite,
)
from app.agents.tools import (
    clear_knowledge_retrieval_log,
    clear_python_exec_log,
    clear_search_log,
    get_knowledge_retrieval_log,
    get_python_exec_log,
    get_search_log,
)
from app.memory.write_memory import (
    append_memory_facts,
    extract_candidate_facts,
    extract_candidate_facts_from_dialogue,
)


class ChatWindow:
    """简单桌面对话窗口，默认启用长短期记忆。"""

    def __init__(self) -> None:
        # Ensure .env values are available before reading GUI runtime flags.
        load_dotenv()

        self.session_id = "gui-default"
        self.use_rag = True
        self.write_memory = True
        self.top_k = 3
        self.turn_count = 0  # 用于追踪轮次编号
        self.chat_font = ("Menlo", 10)
        self.sidebar_font = ("Menlo", 8)
        self.header_font = ("Menlo", 8, "bold")

        # 默认使用 SQLite 后端，保留 txt 兼容路径。
        self.memory_backend = "sqlite"
        self.memory_db = Path("memory/long_term_memory.db")
        self.memory_file = Path("memory/long_term_memory.txt")
        self.memory_chunks = load_memory_chunks(self.memory_file)
        self.orchestrator = os.getenv("AGENT_ORCHESTRATOR", "agent").strip().lower() or "agent"

        settings = get_settings()
        self.chain = build_qa_chain(settings, orchestrator=self.orchestrator)
        
        # 流式消息队列：后台线程put消息，主线程轮询处理
        self.message_queue: queue.Queue[StreamMessage | None] = queue.Queue(maxsize=10)
        self.is_processing = False  # 标记是否正在处理
        self._current_turn_data = {}  # 存储当前轮次的数据（用于侧边栏显示）

        self.root = tk.Tk()
        self.root.title("Agent Chat Demo")
        self.root.geometry("1240x680")
        self.root.minsize(980, 520)
        self.root.configure(bg="#eef2f6")

        self._build_layout()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        status_frame = ttk.Frame(container)
        status_frame.pack(fill=tk.X, pady=(0, 8))

        status_text = (
            f"session_id={self.session_id} | backend={self.memory_backend} "
            f"| orchestrator={self.orchestrator} | rag=on | write_memory=on"
        )
        status = ttk.Label(status_frame, text=status_text)
        status.pack(side=tk.LEFT, anchor=tk.W)

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
            font=self.chat_font,
            bg="#fbfdff",
            fg="#1f2328",
            relief=tk.SOLID,
            borderwidth=1,
            padx=10,
            pady=10,
            spacing1=2,
            spacing3=4,
            insertbackground="#1f2328",
        )
        self.chat_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.chat_box.tag_configure("you", foreground="#0a63cc")
        self.chat_box.tag_configure("assistant", foreground="#0e5a2f")
        self.chat_box.tag_configure("memory", foreground="#57606a")
        self.chat_box.tag_configure("error", foreground="#b42318")

        # 右侧：侧边栏（Prompt 历史）
        sidebar_frame = ttk.LabelFrame(main_paned, text="Prompt History", padding=8)
        main_paned.add(sidebar_frame, weight=1)

        self.sidebar_box = scrolledtext.ScrolledText(
            sidebar_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=self.sidebar_font,
            bg="#f6f8fa",
            fg="#24292f",
            relief=tk.FLAT,
            borderwidth=0,
            padx=8,
            pady=8,
            spacing1=1,
            spacing3=3,
        )
        self.sidebar_box.pack(fill=tk.BOTH, expand=True)

        self.sidebar_box.tag_configure("turn_header", foreground="#0969da", font=("Menlo", 8, "bold"))
        self.sidebar_box.tag_configure("user_query", foreground="#1f6feb")
        self.sidebar_box.tag_configure("context_header", foreground="#6f42c1", font=self.header_font)
        self.sidebar_box.tag_configure("context", foreground="#5a6268")

        # 输入区域
        input_row = ttk.Frame(container)
        input_row.pack(fill=tk.X, pady=(0, 0))

        style = ttk.Style(self.root)
        style.configure("Chat.TEntry", font=("Menlo", 10))
        style.configure("Chat.TButton", font=("Menlo", 9))

        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(input_row, textvariable=self.input_var, style="Chat.TEntry")
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind("<Return>", self._on_send)

        self.send_btn = ttk.Button(input_row, text="发送", command=self._on_send, style="Chat.TButton")
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
        searches: list[dict] | None = None,
        knowledge_hits: list[dict] | None = None,
        python_execs: list[dict] | None = None,
    ) -> None:
        """在侧边栏中记录完整的 prompt 信息和搜索结果。"""
        if searches is None:
            searches = []
        if knowledge_hits is None:
            knowledge_hits = []
        if python_execs is None:
            python_execs = []
            
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

        # 网络搜索结果（新增）
        if searches:
            self.sidebar_box.insert(tk.END, "Web Search:\n", "context_header")
            for search in searches:
                query = search.get("query", "")
                results = search.get("results", "")
                self.sidebar_box.insert(tk.END, f"Query: {query}\n", "user_query")
                # 限制搜索结果显示长度
                result_preview = results[:300] + "..." if len(results) > 300 else results
                self.sidebar_box.insert(tk.END, f"{result_preview}\n\n", "context")

        if knowledge_hits:
            self.sidebar_box.insert(tk.END, "Knowledge Retrieval:\n", "context_header")
            for item in knowledge_hits:
                query = str(item.get("query", "")).strip()
                result = str(item.get("result", "")).strip()
                auxiliary = item.get("auxiliary", {}) if isinstance(item, dict) else {}
                result_preview = result[:300] + "..." if len(result) > 300 else result
                self.sidebar_box.insert(tk.END, f"Query: {query}\n", "user_query")
                self.sidebar_box.insert(tk.END, f"{result_preview}\n\n", "context")

                if isinstance(auxiliary, dict) and auxiliary:
                    used = bool(auxiliary.get("rerank_used", False))
                    fallback = str(auxiliary.get("rerank_fallback", "")).strip()
                    prompt = str(auxiliary.get("rerank_prompt", "")).strip()
                    response = str(auxiliary.get("rerank_response", "")).strip()
                    prompt_preview = prompt[:350] + "..." if len(prompt) > 350 else prompt
                    response_preview = response[:350] + "..." if len(response) > 350 else response

                    self.sidebar_box.insert(
                        tk.END,
                        f"Aux rerank used: {used}"
                        + (f" (fallback={fallback})" if fallback else "")
                        + "\n",
                        "context",
                    )
                    if prompt_preview:
                        self.sidebar_box.insert(tk.END, "Aux rerank prompt:\n", "context_header")
                        self.sidebar_box.insert(tk.END, f"{prompt_preview}\n", "context")
                    if response_preview:
                        self.sidebar_box.insert(tk.END, "Aux rerank response:\n", "context_header")
                        self.sidebar_box.insert(tk.END, f"{response_preview}\n\n", "context")

        if python_execs:
            self.sidebar_box.insert(tk.END, "Python Execution:\n", "context_header")
            for idx, item in enumerate(python_execs, start=1):
                code = str(item.get("code", "")).strip()
                result = str(item.get("result", "")).strip()
                code_preview = code[:300] + "..." if len(code) > 300 else code
                result_preview = result[:300] + "..." if len(result) > 300 else result
                self.sidebar_box.insert(tk.END, f"Run {idx} code:\n", "user_query")
                self.sidebar_box.insert(tk.END, f"{code_preview}\n", "context")
                self.sidebar_box.insert(tk.END, f"Run {idx} result:\n{result_preview}\n\n", "context")

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
        self.is_processing = True
        
        # 启动后台线程处理流式消息
        threading.Thread(target=self._process_turn, args=(user_input,), daemon=True).start()
        
        # 启动主线程轮询队列（非阻塞）
        self._poll_message_queue()

    def _process_turn(self, user_input: str) -> None:
        """后台线程：流式处理一轮对话，将消息放入队列。"""
        try:
            self.turn_count += 1
            # 清除上一轮的搜索记录
            clear_search_log()
            clear_knowledge_retrieval_log()
            clear_python_exec_log()
            
            self.message_queue.put(StreamMessage.progress("正在检索长期记忆"), block=False)
            retrieved_context = self._build_retrieved_context(user_input)
            if retrieved_context.strip():
                self.message_queue.put(StreamMessage.progress("已加载相关长期记忆"), block=False)
            
            # 使用chain.stream()获取中间消息
            # 如果chain没有stream()方法（agent模式），fallback到invoke
            if hasattr(self.chain, 'stream'):
                # 流式模式
                for stream_msg in self.chain.stream(
                    {"question": user_input, "retrieved_context": retrieved_context},
                    session_id=self.session_id,
                    config={"configurable": {"session_id": self.session_id}},
                ):
                    self.message_queue.put(stream_msg, block=False)
            else:
                # Fallback：同步模式（不应该发生，因为build_qa_chain总是返回支持stream()的对象）
                response = self.chain.invoke(
                    {"question": user_input, "retrieved_context": retrieved_context},
                    config={"configurable": {"session_id": self.session_id}},
                )
                self.message_queue.put(StreamMessage.final_answer(response), block=False)
            
            # 准备侧边栏数据
            extracted_facts, written_facts = self._write_long_term_memory(user_input)
            searches = get_search_log()
            knowledge_hits = get_knowledge_retrieval_log()
            python_execs = get_python_exec_log()
            
            # 保存当前轮次数据，供主线程调用_append_to_sidebar
            self._current_turn_data = {
                "user_input": user_input,
                "retrieved_context": retrieved_context,
                "extracted_facts": extracted_facts,
                "written_facts": written_facts,
                "searches": searches,
                "knowledge_hits": knowledge_hits,
                "python_execs": python_execs,
            }
            
            # 放入一个特殊的信号，标记处理完成
            self.message_queue.put(None)  # None表示处理完成
            
        except Exception as exc:  # noqa: BLE001
            self.message_queue.put(StreamMessage.error(f"处理出错: {exc}"), block=False)
            self.message_queue.put(None)

    def _poll_message_queue(self) -> None:
        """主线程：定期轮询消息队列，处理流式消息。"""
        if not self.is_processing:
            return
        
        try:
            # 非阻塞读取队列（timeout=0意为立即返回，如果队列为空则抛出Empty异常）
            while True:
                try:
                    msg = self.message_queue.get_nowait()
                    
                    if msg is None:
                        # None表示处理完成，更新侧边栏
                        self.is_processing = False
                        
                        # 调用_append_to_sidebar记录本轮对话
                        written_facts: list[str] = []
                        if self._current_turn_data:
                            written_facts = self._current_turn_data.get("written_facts", [])
                            self._append_to_sidebar(
                                self.turn_count,
                                self._current_turn_data.get("user_input", ""),
                                self._current_turn_data.get("retrieved_context", ""),
                                self._current_turn_data.get("extracted_facts", []),
                                self._current_turn_data.get("written_facts", []),
                                searches=self._current_turn_data.get("searches", []),
                                knowledge_hits=self._current_turn_data.get("knowledge_hits", []),
                                python_execs=self._current_turn_data.get("python_execs", []),
                            )
                            self._current_turn_data = {}
                        
                        # 显示写入的记忆
                        if written_facts:
                            self._append_message("memory", "已记录:")
                            for fact in written_facts:
                                self._append_message("memory", f"- {fact}")
                        
                        self.send_btn.configure(state=tk.NORMAL)
                        self.input_entry.focus_set()
                        return
                    
                    # 处理流式消息
                    self._handle_stream_message(msg)
                    
                except queue.Empty:
                    # 队列为空，继续轮询
                    break
            
            # 继续定期轮询（100ms）
            if self.is_processing:
                self.root.after(100, self._poll_message_queue)
                
        except Exception as exc:  # noqa: BLE001
            self._append_message("error", f"处理消息队列出错: {exc}")
            self.is_processing = False
            self.send_btn.configure(state=tk.NORMAL)
    
    def _handle_stream_message(self, msg: StreamMessage) -> None:
        """处理单个流式消息，根据类型更新UI。"""
        if msg.message_type == MessageType.PROGRESS:
            self._append_message("memory", f"[执行中] {msg.content}")
        
        elif msg.message_type == MessageType.THINKING:
            # 思考消息：可选显示或忽略
            # self._append_message("memory", f"[思考] {msg.content}")
            pass
        
        elif msg.message_type == MessageType.TOOL_START:
            tool_name = str(msg.content or "")
            display_name = str(msg.metadata.get("display_name", tool_name) or tool_name)
            args = msg.metadata.get("args", {})
            if tool_name == "run_python_code":
                self._append_message("memory", "[执行中] 正在用Python工具做精确计算")
            elif tool_name == "web_search":
                query = str(args.get("query", "")).strip() if isinstance(args, dict) else ""
                suffix = f"：{query}" if query else ""
                self._append_message("memory", f"[执行中] 正在联网搜索{suffix}")
            elif tool_name == "retrieve_pg_knowledge":
                query = str(args.get("query", "")).strip() if isinstance(args, dict) else ""
                suffix = f"：{query}" if query else ""
                self._append_message("memory", f"[执行中] 正在检索知识库{suffix}")
            else:
                self._append_message("memory", f"[执行中] 正在调用工具：{display_name}")
        
        elif msg.message_type == MessageType.TOOL_RESULT:
            tool_name = msg.metadata.get("tool_name", "Unknown")
            result = msg.content
            result_preview = result[:200] + "..." if len(result) > 200 else result
            self._append_message("memory", f"[执行结果] {tool_name}: {result_preview}")
        
        elif msg.message_type == MessageType.FINAL_ANSWER:
            # 最终答案：递增显示到聊天框
            self._append_message("assistant", msg.content)
        
        elif msg.message_type == MessageType.ERROR:
            # 错误：显示为error标签
            self._append_message("error", msg.content)

    def run(self) -> None:
        self.input_entry.focus_set()
        self.root.mainloop()


def main() -> None:
    app = ChatWindow()
    app.run()


if __name__ == "__main__":
    main()
