from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


_request_id = contextvars.ContextVar("researchos_request_id", default=None)
_user_id = contextvars.ContextVar("researchos_user_id", default=None)
_conversation_id = contextvars.ContextVar("researchos_conversation_id", default=None)
_SAFE_FIELDS = {
    "user_id", "conversation_id", "provider", "model", "tool", "status", "duration_ms",
    "error_type", "input_chars", "output_chars", "result_count", "reason", "attempt", "code_bytes", "operation",
}
_SECRET = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|password|secret|authorization)\s*[:=]\s*([^\s,;]+)"
)


def _redact(value: str) -> str:
    return _SECRET.sub(r"\1=[REDACTED]", value)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "researchos_fields", {})
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "researchos_event", _redact(record.getMessage())),
            "request_id": _request_id.get(),
            "user_id": _user_id.get(),
            "conversation_id": _conversation_id.get(),
            "fields": fields,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(data_dir: Path) -> None:
    log_dir = Path(data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("researchos")
    if getattr(logger, "_researchos_log_dir", None) == str(log_dir.resolve()):
        return
    for existing_handler in tuple(logger.handlers):
        logger.removeHandler(existing_handler)
        existing_handler.close()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = _JsonFormatter()
    for handler in (
        logging.handlers.RotatingFileHandler(
            log_dir / "researchos.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        ),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger._researchos_log_dir = str(log_dir.resolve())
    log_event("logging_configured", status="ready")


def log_event(event: str, *, level: int = logging.INFO, **fields: object) -> None:
    safe_fields = {
        key: _redact(str(value))[:300]
        for key, value in fields.items()
        if key in _SAFE_FIELDS and value is not None
    }
    logging.getLogger("researchos").log(
        level,
        event,
        extra={"researchos_event": _redact(event)[:100], "researchos_fields": safe_fields},
    )


@contextmanager
def request_context(user_id: str, conversation_id: str) -> Iterator[str]:
    request_id = str(uuid.uuid4())
    request_token = _request_id.set(request_id)
    user_token = _user_id.set(str(user_id))
    conversation_token = _conversation_id.set(str(conversation_id))
    try:
        yield request_id
    finally:
        _conversation_id.reset(conversation_token)
        _user_id.reset(user_token)
        _request_id.reset(request_token)
