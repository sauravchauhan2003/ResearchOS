from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ResearchTask:
    task_id: str
    question: str


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    source_id: str
    kind: str
    title: str
    text: str
    url: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentFinding:
    role: str
    task_id: str
    assignment: str
    summary: str
    evidence_ids: list[str]
    uncertainties: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ReviewIssue:
    issue_id: str
    concern: str
    evidence_ids: list[str] = field(default_factory=list)
    severity: str = "medium"


@dataclass(frozen=True, slots=True)
class ReviewResult:
    issues: list[ReviewIssue]
    summary: str = ""
    error: str | None = None


@dataclass(slots=True)
class ResearchTrace:
    events: list[str] = field(default_factory=list)
    assignments: list[ResearchTask] = field(default_factory=list)
    findings: list[AgentFinding] = field(default_factory=list)
    review_rounds: list[ReviewResult] = field(default_factory=list)
    issue_dispositions: list[dict[str, str]] = field(default_factory=list)
    unresolved_issues: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ResearchResult:
    question: str
    report_markdown: str
    sources: list[EvidenceSource]
    findings: list[AgentFinding]
    trace: ResearchTrace
    memory_ids: list[str] = field(default_factory=list)
