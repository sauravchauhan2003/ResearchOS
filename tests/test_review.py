from __future__ import annotations

import asyncio
import hashlib

from agents.adversarial_reviewer import AdversarialReviewer
from agents.models import EvidenceSource
from agents.orchestrator import ResearchOrchestrator
from providers.base import GenerationResponse
from tools.web_search import WebSearchTool


URL = "https://example.org/rate"
SOURCE_ID = "W-" + hashlib.sha256(URL.encode()).hexdigest()[:8]


class SeededMismatchProvider:
    def __init__(self, *, disposition: str = "unsupported"):
        self.disposition = disposition
        self.review_calls = 0
        self.revision_calls = 0
        self.review_prompt_contained_evidence = False

    def generate_text(self, model_id, prompt, *, system_instruction=None, **kwargs):
        role = system_instruction or ""
        if "lead researcher" in role:
            text = '{"subquestions":["Reported recovery rate"]}'
        elif "research agent" in role:
            text = f"The source says the rate is 10% [{SOURCE_ID}]."
        elif "You are an evidence analyst." in role:
            text = "The collected page reports 10 percent."
        elif "You independently assess evidence claims." in role:
            text = '{"decisions":[{"issue_id":"R1","status":"' + self.disposition + '","reason":"Evidence does not establish that the draft citation supports 90 percent."}]}'
        elif "independent adversarial reviewer" in role:
            self.review_calls += 1
            self.review_prompt_contained_evidence = "10 percent" in prompt and "90 percent" in prompt
            if self.review_calls == 1:
                text = '{"summary":"Citation mismatch found.","issues":[{"concern":"The cited page says 10 percent, but the draft says 90 percent.","evidence_ids":["' + SOURCE_ID + '"],"severity":"high"}]}'
            else:
                text = '{"summary":"The revised report is cautious.","issues":[]}'
        elif "research report synthesizer" in role:
            text = f"## Findings\n\nThe recovery rate is 90 percent [{SOURCE_ID}]."
        else:
            self.revision_calls += 1
            text = f"## Findings\n\nThe cited page reports 10 percent [{SOURCE_ID}]."
        return GenerationResponse(text, model_id)


class RateSearch:
    def text(self, query, **kwargs):
        return [{"title": "Recovery study", "href": URL, "body": "The recovery rate in the study was 10 percent."}]


def test_reviewer_gets_evidence_and_flags_seeded_citation_mismatch() -> None:
    provider = SeededMismatchProvider()
    reviewer = AdversarialReviewer(provider, "test-model")
    source = EvidenceSource(SOURCE_ID, "web", "Recovery study", "The recovery rate in the study was 10 percent.", URL)
    result = reviewer.review(f"The recovery rate is 90 percent [{SOURCE_ID}].", [source], "What rate was observed?")
    assert provider.review_prompt_contained_evidence
    assert result.issues and "10 percent" in result.issues[0].concern
    assert result.issues[0].evidence_ids == [SOURCE_ID]


def test_unsupported_reviewer_objection_is_recorded_but_not_accepted() -> None:
    provider = SeededMismatchProvider(disposition="unsupported")
    result = asyncio.run(ResearchOrchestrator(
        provider, "test-model", web_search=WebSearchTool(RateSearch()), max_review_rounds=2,
    ).research("What recovery rate was observed?", enable_rag=False, enable_memory=False))
    assert provider.review_calls == 1
    assert provider.revision_calls == 0
    assert result.trace.issue_dispositions[0]["status"] == "unsupported"
    assert not result.trace.unresolved_issues


def test_review_loop_is_bounded_and_keeps_supported_objection_when_no_round_remains() -> None:
    provider = SeededMismatchProvider(disposition="supported")
    result = asyncio.run(ResearchOrchestrator(
        provider, "test-model", web_search=WebSearchTool(RateSearch()), max_review_rounds=1,
    ).research("What recovery rate was observed?", enable_rag=False, enable_memory=False))
    assert provider.review_calls == 1
    assert provider.revision_calls == 0
    assert len(result.trace.review_rounds) == 1
    assert result.trace.unresolved_issues
    assert "Unresolved review issues" in result.report_markdown
