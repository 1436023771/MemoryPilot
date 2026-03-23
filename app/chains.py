import json

from langchain.agents import create_agent
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.prompts import DEFAULT_QA_PROMPT


# 进程内会话仓库：key 是 session_id，value 是该会话的消息历史。
_SESSION_STORE: dict[str, InMemoryChatMessageHistory] = {}


def _strip_markdown_code_fence(raw_text: str) -> str:
    """移除可能的 ```json 代码块外壳。"""
    text = raw_text.strip()

    if text.startswith("```json") or text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def _parse_brief_steps(raw_text: str) -> list[str]:
    """从模型输出中解析 brief_steps 列表，失败时给空列表。"""
    text = _strip_markdown_code_fence(raw_text)

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(payload, dict):
        return []

    steps = payload.get("brief_steps", [])
    if not isinstance(steps, list):
        return []

    return [str(item).strip() for item in steps if str(item).strip()][:5]


def _steps_to_text(steps: list[str]) -> str:
    """将步骤列表转成给第二阶段模型可读的编号文本。"""
    if not steps:
        return ""
    return "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(steps))


def _build_brief_payload(data: dict) -> str:
    """将二阶段 CoT 结果组装成稳定 JSON 文本。"""
    answer = str(data.get("answer", "")).strip()
    steps = data.get("cot_steps", [])
    if not isinstance(steps, list):
        steps = []

    payload = {
        "answer": answer,
        "brief_steps": [str(item).strip() for item in steps if str(item).strip()][:5],
    }
    return json.dumps(payload, ensure_ascii=False)


def _format_agent_user_input(question: str, retrieved_context: str) -> str:
    """将问题和检索上下文组装成给 Agent 的用户输入。"""
    context = retrieved_context.strip() if retrieved_context else ""
    if not context:
        context = "(none)"

    return (
        "User question:\n"
        f"{question}\n\n"
        "Retrieved memory context (use only if relevant):\n"
        f"{context}\n\n"
        "Please answer clearly in 3 bullet points or fewer."
    )


def _to_agent_messages(data: dict) -> list[BaseMessage]:
    """将 RunnableWithMessageHistory 注入的数据转换为 Agent 所需消息列表。"""
    messages: list[BaseMessage] = []
    history = data.get("history", [])

    if isinstance(history, list):
        for item in history:
            if isinstance(item, BaseMessage):
                messages.append(item)

    messages.append(
        HumanMessage(
            content=_format_agent_user_input(
                question=str(data.get("question", "")),
                retrieved_context=str(data.get("retrieved_context", "")),
            )
        )
    )
    return messages


def _extract_agent_answer(result: dict) -> str:
    """从 create_agent 的输出中提取最后一条助手消息。"""
    if not isinstance(result, dict):
        return str(result)

    messages = result.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return str(result)

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content", ""))

    return str(messages[-1])



def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """根据 session_id 获取（或创建）会话历史。"""
    # 首次访问某个 session_id 时，创建新的内存历史对象。
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]


def build_qa_chain(settings: Settings, cot_mode: str = "off"):
    """构建带短期记忆的问答链：Prompt -> 模型 -> 输出解析。"""
    # 初始化聊天模型，兼容 OpenAI 与 DeepSeek（OpenAI 兼容接口）。
    model = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )

    if cot_mode == "brief":
        # 第一阶段：仅生成思路步骤，不直接给最终答案。
        reasoning_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a reasoning planner.\n"
                    "Use retrieved memory only when relevant.\n"
                    "Retrieved memory:\n{retrieved_context}\n\n"
                    "Return ONLY valid JSON in this shape:\n"
                    "{{\"brief_steps\":[\"...\"]}}\n"
                    "Rules:\n"
                    "1) brief_steps must contain 2-5 short actionable steps.\n"
                    "2) Do not include final answer content in steps.\n"
                    "3) Do not output markdown code fences or extra keys.",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        # 第二阶段：严格依据第一阶段步骤生成最终答案。
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a concise assistant.\n"
                    "Use the provided reasoning steps to produce the final answer.\n"
                    "Do not ignore the steps.\n"
                    "Retrieved memory:\n{retrieved_context}\n\n"
                    "Reasoning steps:\n{cot_steps_text}\n\n"
                    "Return only the final answer in plain text.",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        steps_chain = reasoning_prompt | model | StrOutputParser() | RunnableLambda(_parse_brief_steps)

        # brief 模式：先产出 steps，再基于 steps 生成 answer，最后打包为 JSON。
        base_chain = (
            RunnablePassthrough.assign(cot_steps=steps_chain)
            .assign(cot_steps_text=RunnableLambda(lambda x: _steps_to_text(x.get("cot_steps", []))))
            .assign(answer=answer_prompt | model | StrOutputParser())
            | RunnableLambda(_build_brief_payload)
        )
    elif cot_mode == "agent":
        # LangChain v1 内置 Agent（create_agent）模式。
        built_in_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=(
                "You are a concise assistant. Think step-by-step internally, "
                "then provide a concise final answer."
            ),
        )

        base_chain = (
            RunnableLambda(lambda x: {"messages": _to_agent_messages(x)})
            | built_in_agent
            | RunnableLambda(_extract_agent_answer)
        )
    else:
        # 默认模式：保持原有行为不变。
        base_chain = DEFAULT_QA_PROMPT | model | StrOutputParser()

    # 用 RunnableWithMessageHistory 包装后，链会自动读写会话历史。
    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
