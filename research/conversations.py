from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock


CONVERSATION_SCHEMA_VERSION = 1
MAX_CONVERSATIONS = 200
MAX_MESSAGES_PER_CONVERSATION = 120


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationStore:
    """Atomic, account-private chat storage."""

    def __init__(self, account_dir: Path):
        self.path = Path(account_dir) / "conversations" / "conversations.json"
        self.lock = FileLock(str(self.path) + ".lock", timeout=15)

    def _read_unlocked(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != CONVERSATION_SCHEMA_VERSION or not isinstance(payload.get("conversations"), list):
                raise ValueError("Unsupported conversation schema.")
            return payload["conversations"]
        except Exception as exc:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            quarantine = self.path.with_name(f"conversations.corrupt-{stamp}.json")
            try:
                os.replace(self.path, quarantine)
            except OSError:
                pass
            raise ValueError(f"Conversation history was invalid and preserved as {quarantine.name}.") from exc

    def _write_unlocked(self, conversations: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(
            {"schema_version": CONVERSATION_SCHEMA_VERSION, "conversations": conversations},
            indent=2,
            ensure_ascii=False,
        )
        descriptor, temp_name = tempfile.mkstemp(prefix="conversations-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temp_name, self.path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def create(self, title: str = "New chat") -> dict[str, Any]:
        conversation = {
            "conversation_id": str(uuid.uuid4()),
            "title": " ".join(title.split())[:80] or "New chat",
            "created_at": _now(),
            "updated_at": _now(),
            "messages": [],
        }
        with self.lock:
            records = self._read_unlocked()
            records.append(conversation)
            self._write_unlocked(records[-MAX_CONVERSATIONS:])
        return conversation

    def list_all(self) -> list[dict[str, Any]]:
        with self.lock:
            records = self._read_unlocked()
        return sorted(records, key=lambda item: item.get("updated_at", ""), reverse=True)

    def import_legacy_research(self, records: list[dict[str, Any]], *, limit: int = 25) -> int:
        """Move older completed research into chat history once, without duplicating imports."""
        with self.lock:
            conversations = self._read_unlocked()
            if conversations:
                return 0
            imported: list[dict[str, Any]] = []
            for record in reversed(records[:max(0, limit)]):
                question = str(record.get("question", "")).strip()
                response = str(record.get("report_markdown", "")).strip()
                if not question or not response:
                    continue
                now = _now()
                sources = [{
                    "source_id": source.get("source_id", ""),
                    "kind": source.get("kind", "source"),
                    "title": source.get("title", "Source"),
                    "text": str(source.get("text", ""))[:500],
                    "url": source.get("url"),
                } for source in record.get("sources", [])[:12]]
                imported.append({
                    "conversation_id": str(uuid.uuid4()),
                    "title": " ".join(question.split())[:60] or "Previous research",
                    "created_at": now,
                    "updated_at": now,
                    "messages": [
                        {"message_id": str(uuid.uuid4()), "role": "user", "content": question[:2000],
                         "created_at": now, "metadata": {}},
                        {"message_id": str(uuid.uuid4()), "role": "assistant", "content": response[:20000],
                         "created_at": now, "metadata": {"kind": "research_report", "sources": sources,
                                                            "legacy_research_id": record.get("research_id")}},
                    ],
                })
            if imported:
                self._write_unlocked(imported[-MAX_CONVERSATIONS:])
            return len(imported)

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        with self.lock:
            return next(
                (item for item in self._read_unlocked() if item.get("conversation_id") == conversation_id),
                None,
            )

    def append_message(self, conversation_id: str, role: str, content: str,
                       metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("Conversation message role must be user or assistant.")
        message = {
            "message_id": str(uuid.uuid4()),
            "role": role,
            "content": str(content),
            "created_at": _now(),
            "metadata": metadata or {},
        }
        with self.lock:
            conversations = self._read_unlocked()
            for conversation in conversations:
                if conversation.get("conversation_id") != conversation_id:
                    continue
                conversation.setdefault("messages", []).append(message)
                conversation["messages"] = conversation["messages"][-MAX_MESSAGES_PER_CONVERSATION:]
                conversation["updated_at"] = _now()
                if role == "user" and conversation.get("title") == "New chat":
                    conversation["title"] = " ".join(content.split())[:60] or "New chat"
                self._write_unlocked(conversations)
                return message
        raise KeyError(conversation_id)

    def update_message(self, conversation_id: str, message_id: str, *,
                       content: str | None = None, metadata: dict[str, Any] | None = None) -> bool:
        with self.lock:
            conversations = self._read_unlocked()
            for conversation in conversations:
                if conversation.get("conversation_id") != conversation_id:
                    continue
                for message in conversation.get("messages", []):
                    if message.get("message_id") != message_id:
                        continue
                    if content is not None:
                        message["content"] = content
                    if metadata is not None:
                        message["metadata"] = {**message.get("metadata", {}), **metadata}
                    conversation["updated_at"] = _now()
                    self._write_unlocked(conversations)
                    return True
        return False
