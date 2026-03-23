from app.agents.chains import get_session_history


def test_session_history_is_isolated() -> None:
    # 不同 session_id 应该使用独立的历史对象，互不影响。
    session_a = get_session_history("session-a")
    session_b = get_session_history("session-b")

    session_a.add_user_message("My name is Li")

    assert session_a is not session_b
    assert len(session_a.messages) == 1
    assert len(session_b.messages) == 0


def test_session_history_reuse_same_id() -> None:
    # 相同 session_id 应复用同一个历史对象。
    first = get_session_history("same-session")
    second = get_session_history("same-session")

    assert first is second
