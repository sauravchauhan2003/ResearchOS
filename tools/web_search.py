from __future__ import annotations

import hashlib
import ipaddress
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from observability import log_event


class SearchBackend(Protocol):
    def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class WebSource:
    source_id: str
    title: str
    url: str
    snippet: str
    published_at: str | None
    retrieved_at: str


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query: str
    sources: list[WebSource]
    error: str | None = None


def canonical_url(url: str) -> str | None:
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        return None
    host = parts.hostname.casefold().rstrip(".")
    if host.endswith((".local", ".internal", ".localhost")) or host in {"localhost", "localhost.localdomain"}:
        return None
    try:
        if not ipaddress.ip_address(host).is_global:
            return None
    except ValueError:
        pass
    port = parts.port
    netloc = host if port is None or (parts.scheme == "https" and port == 443) or (parts.scheme == "http" and port == 80) else f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    query = urlencode([
        (key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.casefold() not in {"fbclid", "gclid", "dclid", "mc_cid", "mc_eid"}
        and not key.casefold().startswith("utm_")
    ])
    return urlunsplit((parts.scheme, netloc, path, query, ""))


class WebSearchTool:
    """No-key web search backed by the maintained `ddgs` package."""

    def __init__(self, backend: SearchBackend | None = None, *, max_results: int = 8):
        self._backend = backend
        self.max_results = max(1, min(max_results, 20))

    @property
    def backend(self) -> SearchBackend:
        if self._backend is not None:
            return self._backend
        from ddgs import DDGS

        # Build per-call clients: parallel research agents must not share a
        # mutable third-party session unless that backend promises thread safety.
        return DDGS(timeout=10)

    def search(self, query: str, *, max_results: int | None = None) -> SearchResponse:
        clean_query = " ".join(query.split())
        if not clean_query:
            return SearchResponse(query, [], "Enter a research query before searching.")
        if len(clean_query) > 500:
            return SearchResponse(clean_query[:500], [], "Search query must be 500 characters or shorter.")
        count = max(1, min(max_results or self.max_results, 20))
        retrieved_at = datetime.now(timezone.utc).isoformat()
        try:
            results = self.backend.text(clean_query, max_results=count)
        except Exception as exc:
            log_event("web_search_failed", level=40, tool="web_search", status="failed",
                      error_type=type(exc).__name__)
            return SearchResponse(clean_query, [], f"Web search failed ({type(exc).__name__}). Try again later.")
        sources: list[WebSource] = []
        seen: set[str] = set()
        for result in results:
            url = canonical_url(str(result.get("href") or result.get("url") or ""))
            if not url or url in seen:
                continue
            seen.add(url)
            title = " ".join(str(result.get("title") or "Untitled source").split())[:300]
            snippet = " ".join(str(result.get("body") or result.get("snippet") or "").split())[:1200]
            published = result.get("date") or result.get("published") or result.get("published_at")
            source_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
            sources.append(WebSource(source_id, title, url, snippet, str(published) if published else None, retrieved_at))
            if len(sources) >= count:
                break
        log_event("web_search_completed", tool="web_search", status="complete", result_count=len(sources))
        return SearchResponse(clean_query, sources)
