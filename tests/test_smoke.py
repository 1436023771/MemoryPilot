from app.prompts import DEFAULT_QA_PROMPT


def test_prompt_variables() -> None:
    # Prompt 需要同时接收当前问题、历史消息和检索记忆。
    assert "question" in DEFAULT_QA_PROMPT.input_variables
    assert "history" in DEFAULT_QA_PROMPT.input_variables
    assert "retrieved_context" in DEFAULT_QA_PROMPT.input_variables
