from __future__ import annotations

from datetime import datetime

import httpx
import pytest

from tools.web_search import WebSearchTool, canonical_url
from tools.webpage_reader import UnsafeURL, WebpageReader


class FakeSearch:
    def text(self, query, **kwargs):
        return [
            {"title": "Primary report", "href": "https://example.org/report?utm_source=x", "body": "Battery recycling evidence.", "date": "2026-01-02"},
            {"title": "Duplicate", "href": "https://example.org/report?utm_source=x", "body": "duplicate"},
            {"title": "Private address", "href": "http://127.0.0.1/admin", "body": "not allowed"},
        ]


class FakeResponse:
    status_code = 200
    headers = {"content-type": "text/html; charset=utf-8"}
    content = b"<html><head><title>Energy report</title><meta name='date' content='2026-01-02'></head><body><main><h1>Battery recycling</h1><p>Battery recycling can recover valuable materials from end-of-life cells. The report describes a process and provides evidence about lithium recovery across several facility types.</p></main></body></html>"
    encoding = "utf-8"
    url = "https://example.org/report"

    def raise_for_status(self):
        return None


class FakeClient:
    def __init__(self, response=None):
        self.response = response or FakeResponse()
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return self.response


def public_resolver(host, port):
    return [(None, None, None, None, ("93.184.216.34", port))]


def test_search_preserves_source_metadata_and_deduplicates_urls() -> None:
    response = WebSearchTool(FakeSearch()).search("battery recycling")
    assert response.error is None
    assert len(response.sources) == 1
    source = response.sources[0]
    assert source.title == "Primary report"
    assert source.url == "https://example.org/report"
    assert source.published_at == "2026-01-02"
    assert source.retrieved_at
    assert canonical_url("file:///etc/passwd") is None


def test_webpage_reader_extracts_text_and_date() -> None:
    reader = WebpageReader(client=FakeClient(), resolver=public_resolver)
    page = reader.read("https://example.org/report")
    assert page.title == "Energy report"
    assert "valuable materials" in page.text
    assert page.published_at == "2026-01-02"
    assert page.retrieved_at


def test_webpage_reader_blocks_private_ip_and_dns_rebinding_targets() -> None:
    client = FakeClient()
    reader = WebpageReader(client=client, resolver=lambda host, port: [(None, None, None, None, ("10.0.0.2", port))])
    with pytest.raises(UnsafeURL):
        reader.read("http://127.0.0.1/private")
    with pytest.raises(UnsafeURL):
        reader.read("https://example.org/private")
    assert client.calls == []


def test_search_failures_are_reported_without_fabricating_sources() -> None:
    class BrokenSearch:
        def text(self, query, **kwargs):
            raise TimeoutError

    result = WebSearchTool(BrokenSearch()).search("question")
    assert result.sources == []
    assert "failed" in result.error
