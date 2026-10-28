from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura
from lxml import html as lxml_html
from trafilatura.metadata import extract_metadata

from tools.web_search import canonical_url


MAX_PAGE_BYTES = 3 * 1024 * 1024
REDIRECT_STATUSES = {301, 302, 303, 307, 308}


@dataclass(frozen=True, slots=True)
class WebPage:
    title: str
    url: str
    text: str
    published_at: str | None
    retrieved_at: str


class UnsafeURL(ValueError):
    pass


def _resolve(host: str, port: int):
    return socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)


def _assert_public_url(url: str, resolver: Callable = _resolve) -> str:
    canonical = canonical_url(url)
    if canonical is None:
        raise UnsafeURL("Only public HTTP or HTTPS webpages can be retrieved.")
    parts = urlsplit(canonical)
    host = parts.hostname or ""
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            addresses = [ipaddress.ip_address(info[4][0]) for info in resolver(host, parts.port or (443 if parts.scheme == "https" else 80))]
        except Exception as exc:
            raise UnsafeURL("Could not safely resolve the webpage host.") from exc
    if not addresses or any(not address.is_global for address in addresses):
        raise UnsafeURL("The webpage host resolves to a private or reserved network address.")
    return canonical


class WebpageReader:
    def __init__(self, *, client: httpx.Client | None = None, resolver: Callable = _resolve,
                 max_bytes: int = MAX_PAGE_BYTES, timeout_seconds: float = 10.0, max_redirects: int = 4):
        self.client = client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            headers={"User-Agent": "ResearchOS/0.1 (research source reader)"},
        )
        self.resolver = resolver
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects

    def read(self, url: str) -> WebPage:
        current = url
        response = None
        for redirect_count in range(self.max_redirects + 1):
            safe_url = _assert_public_url(current, self.resolver)
            response = self.client.get(safe_url, follow_redirects=False)
            if response.status_code not in REDIRECT_STATUSES:
                break
            location = response.headers.get("location")
            if not location or redirect_count >= self.max_redirects:
                raise ValueError("Webpage redirect limit reached.")
            current = urljoin(safe_url, location)
        if response is None:
            raise ValueError("No response was received from the webpage.")
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").casefold()
        if content_type and not any(kind in content_type for kind in ("text/html", "application/xhtml+xml")):
            raise ValueError("The source URL did not return an HTML page.")
        raw = response.content
        if len(raw) > self.max_bytes:
            raise ValueError("Webpage is larger than the configured retrieval limit.")
        html = raw.decode(response.encoding or "utf-8", errors="replace")
        text = trafilatura.extract(html, include_comments=False, include_tables=True, favor_precision=True)
        if not text:
            raise ValueError("Could not extract readable text from this webpage.")
        metadata = extract_metadata(html)
        tree = lxml_html.fromstring(html)
        og_title = tree.xpath("string(//meta[@property='og:title']/@content)").strip()
        html_title = tree.xpath("string(//title)").strip()
        final_url = str(response.url)
        return WebPage(
            title=(og_title or html_title or metadata.title or final_url)[:300],
            url=final_url,
            text=text[:100_000],
            published_at=metadata.date,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        )
