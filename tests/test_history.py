from pathlib import Path

from agents.models import AgentFinding, EvidenceSource, ResearchResult, ResearchTask, ResearchTrace
from research.history import ResearchHistoryStore


def sample_result(question: str = "Private question") -> ResearchResult:
    trace = ResearchTrace(
        events=["planned", "reviewed"],
        assignments=[ResearchTask("research-1", question)],
        findings=[AgentFinding("research agent", "research-1", question, "Finding", ["W-aabbccdd"])],
        unresolved_issues=["Evidence gap"],
    )
    return ResearchResult(
        question,
        "# Report\n\nA cautious result.",
        [EvidenceSource("W-aabbccdd", "web", "Actual source", "Excerpt", "https://example.org/source")],
        trace.findings,
        trace,
    )


def test_history_persists_full_trace_and_can_delete_or_clear(tmp_path: Path) -> None:
    first = ResearchHistoryStore(tmp_path / "account-a")
    saved = first.add_result(sample_result())
    restarted = ResearchHistoryStore(tmp_path / "account-a")
    records = restarted.list_all()
    assert records[0]["research_id"] == saved["research_id"]
    assert records[0]["sources"][0]["url"] == "https://example.org/source"
    assert records[0]["trace"]["unresolved_issues"] == ["Evidence gap"]
    assert restarted.delete(saved["research_id"])
    assert restarted.list_all() == []

    restarted.add_result(sample_result("Second question"))
    assert restarted.clear() == 1


def test_history_storage_is_scoped_by_account(tmp_path: Path) -> None:
    first = ResearchHistoryStore(tmp_path / "account-a")
    second = ResearchHistoryStore(tmp_path / "account-b")
    first.add_result(sample_result())
    assert len(first.list_all()) == 1
    assert second.list_all() == []
