from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# Prompt 结构：系统指令 + 历史消息 + 当前用户问题。
DEFAULT_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        # 系统消息：定义输出风格和长度约束。
        (
            "system",
            "You are a concise assistant. Answer clearly in 3 bullet points or fewer.",
        ),
        # 历史占位：由 RunnableWithMessageHistory 在运行时注入。
        MessagesPlaceholder(variable_name="history"),
        # 当前轮用户输入。
        ("human", "{question}"),
    ]
)
