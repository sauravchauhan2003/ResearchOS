from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from observability import log_event


T = TypeVar("T")
MAX_RETRIES = 8
INITIAL_DELAY_SECONDS = 1.0


def _http_status(exc: Exception) -> int | None:
    for candidate in (exc, getattr(exc, "response", None)):
        if candidate is None:
            continue
        for attribute in ("status_code", "code"):
            value = getattr(candidate, attribute, None)
            if isinstance(value, int):
                return value
    return None


def _is_retryable(exc: Exception) -> bool:
    status = _http_status(exc)
    if status is None:
        return True
    return status in {408, 409, 425, 429} or status >= 500


def retry_request(
    request: Callable[[], T],
    *,
    operation: str,
    max_retries: int = MAX_RETRIES,
    initial_delay_seconds: float = INITIAL_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry transient provider errors up to eight times with exponential delays."""
    retries = 0
    delay = initial_delay_seconds
    while True:
        try:
            return request()
        except Exception as exc:
            if retries >= max_retries or not _is_retryable(exc):
                raise
            retries += 1
            log_event("provider_retry", level=30, operation=operation, error_type=type(exc).__name__,
                      attempt=retries, status="retrying")
            sleep(delay)
            delay *= 2
