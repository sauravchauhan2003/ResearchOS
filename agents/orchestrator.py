from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from typing import Any

from agents.adversarial_reviewer import AdversarialReviewer
from agents.context import compact_blocks, compact_text
from agents.evidence_analyst import EvidenceAnalyst
from agents.models import AgentFinding, EvidenceSource, ResearchResult, ResearchTask, ResearchTrace, ReviewIssue
from agents.planner import Planner
from agents.researcher import ResearchAgent
from agents.synthesizer import Synthesizer
from memory.memory_manager import MemoryManager, memory_search_query
from memory.models import MemoryRecord
from guardrails import should_save_answer_to_memory
from providers.base import ModelProvider
from rag.retriever import RagPipeline
from tools.web_search import WebSearchTool
from tools.webpage_reader import WebpageReader
from tools.tool_registry import ToolRegistry


def _raise_if_streamlit_control(exc: Exception) -> None:
    if type(exc).__name__ in {"StopException", "RerunException"}:
        raise exc


class ResearchOrchestrator:
    """A deterministic plan → parallel research → analyze → synthesize → review workflow."""

    def __init__(self, provider: ModelProvider, model_id: str, *, web_search: WebSearchTool | None = None,
                 webpage_reader: WebpageReader | None = None, rag_pipeline: RagPipeline | None = None,
                 memory_manager: MemoryManager | None = None, max_agents: int = 2,
                 max_review_rounds: int = 1, research_timeout_seconds: float = 180.0,
                 max_output_tokens: int = 2600,
                 on_step: Callable[[str, str, str], None] | None = None):
        self.provider = provider
        self.model_id = model_id
        self.web_search = web_search
        self.webpage_reader = webpage_reader
        self.rag_pipeline = rag_pipeline
        self.memory_manager = memory_manager
        self.tools = ToolRegistry(
            web_search=web_search,
            webpage_reader=webpage_reader,
            rag=rag_pipeline,
            memory=memory_manager,
        )
        self.max_agents = max(1, min(int(max_agents), 3))
        self.max_review_rounds = max(0, min(int(max_review_rounds), 1))
        self.research_timeout_seconds = max(5.0, min(float(research_timeout_seconds), 900.0))
        self.max_output_tokens = max_output_tokens
        self.on_step = on_step

    def _emit(self, action: str, status: str = "complete", detail: str = "") -> None:
        if self.on_step is not None:
            self.on_step(action, status, detail)

    async def _call(self, function, *args, timeout: float | None = None, **kwargs):
        return await asyncio.wait_for(
            asyncio.to_thread(function, *args, **kwargs),
            timeout=timeout or self.research_timeout_seconds,
        )

    async def research(self, question: str, *, objectives: str = "", enable_memory: bool = True,
                       enable_rag: bool = True, enable_web_search: bool = True,
                       conversation_context: str = "") -> ResearchResult:
        clean_question = " ".join(question.split())
        if not clean_question:
            raise ValueError("Research question cannot be empty.")
        conversation_context = compact_text(conversation_context, 1200)
        trace = ResearchTrace()
        memories: list[MemoryRecord] = []
        memory_ids: list[str] = []
        if enable_memory and self.memory_manager is not None:
            self._emit("Looking up relevant account memory", "running")
            try:
                memories = await self._call(
                    self.memory_manager.search,
                    compact_text(memory_search_query(clean_question, conversation_context), 1200),
                    limit=2,
                )
                memory_ids = [item.memory_id for item in memories]
                trace.events.append(f"Retrieved {len(memories)} relevant account memories before planning.")
                self._emit("Looking up relevant account memory", "complete", f"{len(memories)} matches")
            except Exception as exc:
                _raise_if_streamlit_control(exc)
                trace.events.append(f"Memory retrieval failed: {type(exc).__name__}.")
                self._emit("Looking up relevant account memory", "failed", type(exc).__name__)
        else:
            trace.events.append("Persistent memory is disabled for this session.")

        memory_context = compact_blocks(
            (
                f"Previous query: {record.user_query}. Useful findings: "
                f"{record.conclusions or record.findings}. Open questions: "
                f"{'; '.join(record.unresolved_questions[:2])}"
                for record in memories
            ),
            max_chars=800,
            per_block_chars=450,
        )
        planning_objectives = objectives.strip()
        if conversation_context:
            planning_objectives += f"\nRecent conversation context: {conversation_context}"
        if memory_context:
            planning_objectives += f"\nRelevant account memory: {memory_context}"

        self._emit("Planning the research", "running")
        try:
            tasks = await self._call(Planner(self.provider, self.model_id).plan, clean_question,
                                     objectives=planning_objectives, max_agents=self.max_agents)
            plan_details = " | ".join(task.question[:120] for task in tasks[:self.max_agents])
            self._emit("Planning the research", "complete",
                       f"{min(len(tasks), self.max_agents)} task(s): {plan_details}")
        except Exception as exc:
            _raise_if_streamlit_control(exc)
            tasks = [ResearchTask("research-1", clean_question)]
            trace.events.append(f"Planner failed; used one bounded fallback task ({type(exc).__name__}).")
            self._emit("Planning the research", "failed", "Using one fallback task")
        tasks = tasks[:self.max_agents]
        trace.assignments = tasks
        trace.events.append(f"Planner assigned {len(tasks)} independent research task(s).")
        researcher_context = "\n".join(
            item for item in (
                f"Recent conversation: {conversation_context}" if conversation_context else "",
                f"Relevant prior account findings: {memory_context}" if memory_context else "",
            ) if item
        )
        research_question_context = clean_question
        if researcher_context:
            research_question_context = f"{clean_question}\n\n{researcher_context}"

        async def run_task(task: ResearchTask):
            agent = ResearchAgent(
                self.provider,
                self.model_id,
                self.tools.get("web_search") if enable_web_search and "web_search" in self.tools.names() else None,
                self.tools.get("webpage_reader") if enable_web_search and "webpage_reader" in self.tools.names() else None,
                self.tools.get("rag") if enable_rag and "rag" in self.tools.names() else None,
            )
            self._emit(f"Researching: {task.question[:90]}", "running")
            try:
                outcome = await self._call(agent.investigate, research_question_context, task, memories=memories)
                self._emit(
                    f"Researching: {task.question[:90]}",
                    "complete" if not outcome[0].error else "failed",
                    f"{len(outcome[1])} evidence item(s)" if not outcome[0].error else outcome[0].error[:120],
                )
                return outcome
            except Exception as exc:
                _raise_if_streamlit_control(exc)
                self._emit(f"Researching: {task.question[:90]}", "failed", type(exc).__name__)
                return AgentFinding("research agent", task.task_id, task.question,
                                    "The assigned research task did not complete.", [],
                                    error=f"{type(exc).__name__}: {str(exc)[:250]}"), []

        semaphore = asyncio.Semaphore(self.max_agents)

        async def bounded(task):
            async with semaphore:
                return await run_task(task)

        combined = await asyncio.gather(*(bounded(task) for task in tasks))
        findings: list[AgentFinding] = [item[0] for item in combined]
        source_map: dict[str, EvidenceSource] = {}
        for finding, sources in combined:
            for source in sources:
                source_map.setdefault(source.source_id, source)
            trace.findings.append(finding)
            trace.events.append(f"{finding.task_id} completed with {len(finding.evidence_ids)} evidence item(s).")
        sources = list(source_map.values())
        trace.events.append(f"Evidence analyst received {len(sources)} unique source(s).")

        analyst = EvidenceAnalyst(self.provider, self.model_id)
        self._emit("Comparing evidence", "running")
        try:
            analysis = await self._call(analyst.analyze, findings, sources)
            self._emit("Comparing evidence", "complete", f"{len(sources)} source(s)")
        except Exception as exc:
            _raise_if_streamlit_control(exc)
            analysis = "Automated evidence comparison failed; the report must preserve uncertainty."
            trace.events.append(f"Evidence analysis failed: {type(exc).__name__}.")
            self._emit("Comparing evidence", "failed", type(exc).__name__)
        synthesizer = Synthesizer(self.provider, self.model_id)
        self._emit("Drafting the research response", "running")
        try:
            draft = await self._call(synthesizer.draft, clean_question, findings, sources, analysis)
            self._emit("Drafting the research response", "complete")
        except Exception as exc:
            _raise_if_streamlit_control(exc)
            draft = (
                "## Research report\n\nThe selected model could not synthesize a report. "
                "Collected evidence is listed below; no conclusion is claimed.\n\n" + Synthesizer._source_list(sources)
            )
            trace.events.append(f"Initial synthesis failed: {type(exc).__name__}.")
            self._emit("Drafting the research response", "failed", type(exc).__name__)

        reviewer = AdversarialReviewer(self.provider, self.model_id)
        for round_number in range(1, self.max_review_rounds + 1):
            self._emit("Reviewing the response", "running", f"Pass {round_number}")
            review = await self._call(reviewer.review, draft, sources, clean_question)
            trace.review_rounds.append(review)
            trace.events.append(f"Adversarial review {round_number} returned {len(review.issues)} issue(s).")
            self._emit("Reviewing the response", "complete" if not review.error else "failed",
                       review.summary or f"{len(review.issues)} issue(s)")
            if review.error:
                trace.unresolved_issues.append(review.error)
                break
            if not review.issues:
                break
            self._emit("Checking reviewer objections", "running")
            dispositions = await self._call(analyst.evaluate_objections, review.issues, sources)
            self._emit("Checking reviewer objections", "complete", f"{len(dispositions)} decision(s)")
            disposition_by_id = {item["issue_id"]: item for item in dispositions}
            accepted: list[dict[str, str]] = []
            for issue in review.issues:
                decision = disposition_by_id.get(issue.issue_id, {"status": "uncertain", "reason": "No disposition."})
                trace.issue_dispositions.append({"issue_id": issue.issue_id, "status": decision["status"],
                                                 "reason": decision.get("reason", "")})
                if decision["status"] in {"supported", "uncertain"}:
                    accepted.append({**decision, "issue_id": issue.issue_id, "concern": issue.concern})
                else:
                    trace.events.append(f"Reviewer objection {issue.issue_id} was not supported by collected evidence.")
            if not accepted:
                break
            if round_number == self.max_review_rounds:
                trace.unresolved_issues.extend(
                    f"{item['issue_id']}: {item['concern']} ({item['status']})" for item in accepted
                )
                break
            self._emit("Revising the response", "running")
            try:
                revised = await self._call(synthesizer.revise, clean_question, draft, findings, sources, analysis, accepted)
                self._emit("Revising the response", "complete")
            except Exception as exc:
                _raise_if_streamlit_control(exc)
                trace.unresolved_issues.extend(f"{item['issue_id']}: {item['concern']}" for item in accepted)
                trace.events.append(f"Revision failed: {type(exc).__name__}.")
                self._emit("Revising the response", "failed", type(exc).__name__)
                break
            if revised == draft:
                trace.unresolved_issues.extend(f"{item['issue_id']}: {item['concern']}" for item in accepted)
                trace.events.append("Revision returned an unchanged report; review loop stopped.")
                break
            draft = revised

        if trace.unresolved_issues:
            unresolved = "\n".join(f"- {item}" for item in trace.unresolved_issues)
            if "## Sources" in draft:
                before, after = draft.split("## Sources", 1)
                draft = f"{before.rstrip()}\n\n## Unresolved review issues\n\n{unresolved}\n\n## Sources{after}"
            else:
                draft += f"\n\n## Unresolved review issues\n\n{unresolved}"

        if enable_memory and self.memory_manager is not None and should_save_answer_to_memory(draft):
            self._emit("Saving useful findings to account memory", "running")
            try:
                stored = await self._call(
                    self.memory_manager.add,
                    research_session_id=str(uuid.uuid4()),
                    user_query=clean_question,
                    findings="\n\n".join(item.summary for item in findings),
                    conclusions=draft[:12000],
                    unresolved_questions=trace.unresolved_issues,
                    relevant_context=f"Evidence sources collected: {len(sources)}.",
                )
                memory_ids.append(stored.memory_id)
                trace.events.append("Final research result was written to this account's persistent memory.")
                self._emit("Saving useful findings to account memory", "complete")
            except Exception as exc:
                _raise_if_streamlit_control(exc)
                trace.events.append(f"Final memory write failed: {type(exc).__name__}.")
                self._emit("Saving useful findings to account memory", "failed", type(exc).__name__)
        return ResearchResult(clean_question, draft, sources, findings, trace, memory_ids)
