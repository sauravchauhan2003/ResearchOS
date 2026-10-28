from __future__ import annotations

from research.conversations import ConversationStore
from ui.chat_view import DRAFT_CONVERSATION_PREFIX, _persist_user_prompt


def test_conversations_persist_messages_and_pending_work_is_recoverable(tmp_path) -> None:
    store = ConversationStore(tmp_path / "account")
    conversation = store.create()
    user_message = store.append_message(conversation["conversation_id"], "user", "Continue the analysis")

    reloaded = ConversationStore(tmp_path / "account")
    saved = reloaded.get(conversation["conversation_id"])
    assert saved is not None
    assert saved["messages"][-1]["message_id"] == user_message["message_id"]
    assert saved["messages"][-1]["role"] == "user"
    assert saved["title"] == "Continue the analysis"


def test_conversations_and_tool_metadata_are_account_scoped(tmp_path) -> None:
    first = ConversationStore(tmp_path / "account-a")
    second = ConversationStore(tmp_path / "account-b")
    conversation = first.create("Sandbox chat")
    proposal = first.append_message(
        conversation["conversation_id"],
        "assistant",
        "Review this code",
        metadata={"kind": "sandbox_proposal", "code": "print(1)", "status": "pending"},
    )

    assert first.update_message(
        conversation["conversation_id"], proposal["message_id"], metadata={"status": "completed"}
    )
    assert first.get(conversation["conversation_id"])["messages"][-1]["metadata"]["status"] == "completed"
    assert second.list_all() == []


def test_legacy_research_is_imported_into_chat_once(tmp_path) -> None:
    store = ConversationStore(tmp_path / "account")
    records = [{
        "research_id": "old-id",
        "question": "A previous question",
        "report_markdown": "## Previous answer",
        "sources": [{"source_id": "W-12345678", "kind": "web", "title": "A source", "text": "Excerpt"}],
    }]

    assert store.import_legacy_research(records) == 1
    assert store.import_legacy_research(records) == 0
    saved = store.list_all()[0]
    assert saved["messages"][0]["content"] == "A previous question"
    assert saved["messages"][1]["metadata"]["legacy_research_id"] == "old-id"


def test_first_prompt_creates_the_conversation_from_a_draft(tmp_path) -> None:
    store = ConversationStore(tmp_path / "account")
    assert store.list_all() == []

    conversation_id, message = _persist_user_prompt(
        store, f"{DRAFT_CONVERSATION_PREFIX}session", "A first question"
    )

    assert store.list_all()[0]["conversation_id"] == conversation_id
    assert store.get(conversation_id)["messages"] == [message]
