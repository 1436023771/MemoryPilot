from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.prompt_store import get_prompt_template


# Prompt 结构：系统指令 + 历史消息 + 当前用户问题。
DEFAULT_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        # 系统消息：定义输出风格和长度约束。
        (
            "system",
            get_prompt_template("core.default_qa_system"),
        ),
        # 历史占位：由 RunnableWithMessageHistory 在运行时注入。
        MessagesPlaceholder(variable_name="history"),
        # 当前轮用户输入。
        ("human", "{question}"),
    ]
)
