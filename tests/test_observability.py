from __future__ import annotations

import json

from observability import configure_logging, log_event, request_context


def test_structured_logging_redacts_secret_like_values_and_tracks_requests(tmp_path) -> None:
    configure_logging(tmp_path)
    with request_context("account-1", "conversation-1"):
        log_event("test_event", status="complete", reason="api_key=do-not-log", input_chars=42)

    records = [json.loads(line) for line in (tmp_path / "logs" / "researchos.log").read_text().splitlines()]
    record = next(item for item in records if item["event"] == "test_event")
    assert record["request_id"]
    assert record["user_id"] == "account-1"
    assert record["conversation_id"] == "conversation-1"
    assert record["fields"]["reason"] == "api_key=[REDACTED]"
    assert "do-not-log" not in json.dumps(record)
