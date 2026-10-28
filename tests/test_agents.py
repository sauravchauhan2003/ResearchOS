from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from types import SimpleNamespace

from agents.adversarial_reviewer import AdversarialReviewer
from agents.models import EvidenceSource
from agents.orchestrator import ResearchOrchestrator
from agents.planner import Planner
from agents.synthesizer import Synthesizer
from providers.base import GenerationResponse, ProviderName
from tools.web_search import WebSearchTool


class FakeProvider:
    def __init__(self):
        self.lock = threading.Lock()
        self.active_research = 0
        self.max_active_research = 0

    def generate_text(self, model_id, prompt, *, system_instruction=None, max_output_tokens=2048, temperature=0.2):
        role = system_instruction or ""
        if "lead researcher" in role:
            text = '{"subquestions":["Current recycling evidence","Conflicting evidence"]}'
        elif "research agent" in role:
            with self.lock:
                self.active_research += 1
                self.max_active_research = max(self.max_active_research, self.active_research)
            time.sleep(0.03)
            with self.lock:
                self.active_research -= 1
            text = "Collected evidence supports further study."
        elif "evidence analyst" in role:
            text = "Evidence is limited and no disagreement can be resolved from snippets alone."
        elif "independent adversarial reviewer" in role:
            text = '{"summary":"No unsupported statements found.","issues":[]}'
        elif "research report synthesizer" in role:
            text = "## Findings\n\nThe report is cautious and cites the collected page [W-PLACEHOLDER]."
        else:
            text = "Revised report with evidence limitations."
        return GenerationResponse(text, model_id, 4, 8)


class FakeSearch:
    def text(self, query, **kwargs):
        return [{"title": "Research article", "href": "https://example.org/research", "body": "This article discusses battery recycling.", "date": "2026-01-01"}]


def test_planner_uses_dynamic_subquestions_and_bounds_agent_count() -> None:
    provider = FakeProvider()
    tasks = Planner(provider, "test-model").plan("Battery recycling", max_agents=1)
    assert len(tasks) == 1
    assert tasks[0].question == "Current recycling evidence"


def test_reviewer_parses_structured_specific_objections() -> None:
    class ReviewerProvider:
        def generate_text(self, model_id, prompt, **kwargs):
            return GenerationResponse(
                '{"summary":"One claim lacks evidence.","issues":[{"concern":"The claimed rate is not supported.","evidence_ids":["W-deadbeef"],"severity":"high"}]}',
                model_id,
            )

    reviewer = AdversarialReviewer(ReviewerProvider(), "test")
    result = reviewer.review("The rate is 90% [W-deadbeef].", [
        EvidenceSource("W-deadbeef", "web", "Study", "The study does not report a rate.")
    ], "What is the rate?")
    assert len(result.issues) == 1
    assert result.issues[0].evidence_ids == ["W-deadbeef"]
    assert result.issues[0].severity == "high"


def test_synthesizer_removes_unverified_citations_and_links() -> None:
    class NoopProvider:
        def generate_text(self, model_id, prompt, **kwargs):
            return GenerationResponse("unused", model_id)

    source = EvidenceSource("W-cafebabe", "web", "Known source", "evidence", "https://example.org/known")
    report = Synthesizer(NoopProvider(), "test").finalize(
        "Fact [W-cafebabe] [W-badbad00] [2] https://example.org/unknown", [source]
    )
    assert "[W-cafebabe]" in report
    assert "W-badbad00" not in report
    assert "[2]" not in report
    assert "unverified link omitted" in report
    assert "https://example.org/known" in report


def test_orchestrator_runs_independent_agents_in_bounded_parallel_and_reviews() -> None:
    provider = FakeProvider()
    actions = []
    orchestrator = ResearchOrchestrator(
        provider, "test-model", web_search=WebSearchTool(FakeSearch()),
        max_agents=2, max_review_rounds=1, research_timeout_seconds=20,
        on_step=lambda action, status, detail: actions.append((action, status, detail)),
    )
    result = asyncio.run(orchestrator.research("What does the evidence say about battery recycling?"))
    expected = "W-" + hashlib.sha256(b"https://example.org/research").hexdigest()[:8]
    assert len(result.trace.assignments) == 2
    assert len(result.findings) == 2
    assert provider.max_active_research <= 2
    assert len(result.trace.review_rounds) == 1
    assert result.sources[0].source_id == expected
    assert "## Sources" in result.report_markdown
    assert "[unverified link omitted]" not in result.report_markdown
    assert any(action == "Planning the research" and status == "complete" and detail.startswith("2 task(s):")
               for action, status, detail in actions)
    assert any(action == "Drafting the research response" and status == "complete"
               for action, status, _ in actions)


def test_orchestrator_caps_research_and_review_steps() -> None:
    orchestrator = ResearchOrchestrator(FakeProvider(), "test-model", max_agents=99, max_review_rounds=99)
    assert orchestrator.max_agents == 3
    assert orchestrator.max_review_rounds == 1

