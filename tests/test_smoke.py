from app.prompts import DEFAULT_QA_PROMPT


def test_prompt_variables() -> None:
    # Prompt 需要同时接收当前问题和历史消息。
    assert "question" in DEFAULT_QA_PROMPT.input_variables
    assert "history" in DEFAULT_QA_PROMPT.input_variables
