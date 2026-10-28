from __future__ import annotations

from auth.user_store import UserStore
from research.conversations import ConversationStore
from streamlit.testing.v1 import AppTest


def _authenticate_app(app, data_dir):
    store = UserStore(data_dir / "accounts.sqlite3")
    user = store.register("reviewer", "long-enough-test-password")
    token = store.create_login_session(user.id)
    app.session_state["researchos_user"] = {"id": user.id, "username": user.username}
    app.session_state["researchos_session_token"] = token


def test_authenticated_app_renders_chat_first_ui(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RESEARCHOS_DATA_DIR", str(tmp_path / "data"))
    app = AppTest.from_file("../app.py")
    _authenticate_app(app, tmp_path / "data")
    app.run()
    assert not app.exception
    assert any(item.label == "Generation provider" for item in app.selectbox)
    assert any(item.value == "Choose a generation model" for item in app.subheader)
    assert any(item.label == "Which API key should this account use?" for item in app.radio)
    assert not any(item.label == "Workspace" for item in app.radio)
    assert len(app.chat_input) == 1
    assert any(item.label == "＋  New chat" for item in app.button)


def test_repeated_new_chat_clicks_do_not_create_empty_history_entries(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("RESEARCHOS_DATA_DIR", str(data_dir))
    app = AppTest.from_file("../app.py")
    _authenticate_app(app, data_dir)
    user_id = app.session_state["researchos_user"]["id"]
    app.run()

    for _ in range(3):
        button = next(item for item in app.button if item.label == "＋  New chat")
        button.click().run()
        assert not app.exception

    conversation_path = data_dir / "users" / user_id / "conversations" / "conversations.json"
    assert not conversation_path.exists()


def test_login_starts_a_fresh_chat_while_preserving_prior_chat_history(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("RESEARCHOS_DATA_DIR", str(data_dir))
    app = AppTest.from_file("../app.py")
    _authenticate_app(app, data_dir)
    user_id = app.session_state["researchos_user"]["id"]
    conversations = ConversationStore(data_dir / "users" / user_id)
    previous = conversations.create("Prior chat")
    conversations.append_message(previous["conversation_id"], "user", "Prior question marker")
    conversations.append_message(previous["conversation_id"], "assistant", "Prior answer marker")
    app.session_state["researchos_fresh_chat_user_id"] = user_id

    app.run()

    assert not app.exception
    assert not any("Prior question marker" in item.value for item in app.markdown)
    assert not any("Prior answer marker" in item.value for item in app.markdown)
    assert any(item.label == "Prior chat" for item in app.button)


