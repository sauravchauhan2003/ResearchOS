from __future__ import annotations

import asyncio

import streamlit as st

from agents.orchestrator import ResearchOrchestrator
from auth.user_store import User
from memory.memory_manager import MemoryManager
from providers.base import ProviderName
from providers.model_registry import ModelRegistry
from research.history import ResearchHistoryStore
from rag.retriever import RagPipeline
from tools.web_search import WebSearchTool
from tools.webpage_reader import WebpageReader


def render_research_view(user: User, settings, store, saved_selection: dict[str, str]) -> None:
    account_dir = settings.user_data_dir(user.id)
    st.header("Research")
    st.caption("Use web evidence, your uploaded documents, and relevant research from this account.")

    upload = st.file_uploader("Add a document to this account's library", type=["pdf", "txt", "md", "markdown"], key=f"upload_{user.id}")
    if st.button("Ingest document", disabled=upload is None, key=f"ingest_{user.id}") and upload is not None:
        try:
            pipeline = RagPipeline(account_dir)
            with st.spinner("Cleaning, chunking, embedding, and indexing document…"):
                document = pipeline.ingest(upload.name, upload.getvalue())
            st.success(f"Indexed {document.chunk_count} chunks from {document.source_filename}.")
        except Exception as exc:
            st.error(f"Document ingestion failed: {exc}")

    with st.form(f"research_form_{user.id}"):
        question = st.text_area("Research question", height=110, max_chars=2000)
        objectives = st.text_area("Objectives and constraints (optional)", height=80, max_chars=2000)
        start = st.form_submit_button("Start research", type="primary")
    if start:
        provider_name = saved_selection.get("provider")
        model_id = saved_selection.get("model")
        if not provider_name or not model_id:
            st.error("Choose and save a generation model above the page navigation first.")
        elif not question.strip():
            st.error("Enter a research question.")
        else:
            registry = ModelRegistry.for_account(settings=settings, store=store, user_id=user.id)
            if not registry.configured(ProviderName(provider_name)):
                st.error("No API key is available for this provider. Choose the common key or save your own key on the main screen.")
                return
            if not registry.model_matches_provider(ProviderName(provider_name), model_id):
                st.error("The saved model does not match the selected provider. Fetch and choose a model on the main screen.")
                return
            provider = registry.get(ProviderName(provider_name))
            sidebar_settings = st.session_state.get(f"research_limits_{user.id}", {})
            memory_enabled = st.session_state.get(f"enable_memory_{user.id}", True)
            rag_enabled = st.session_state.get(f"enable_rag_{user.id}", True)
            web_enabled = st.session_state.get(f"enable_web_{user.id}", True)
            rag_pipeline = RagPipeline(
                account_dir,
                chunk_size=sidebar_settings.get("chunk_size", 1200),
                overlap=sidebar_settings.get("chunk_overlap", 150),
            ) if rag_enabled else None
            memory_manager = MemoryManager(account_dir, settings.data_dir / ".memory-signing-key")
            orchestrator = ResearchOrchestrator(
                provider,
                model_id,
                web_search=WebSearchTool() if web_enabled else None,
                webpage_reader=WebpageReader() if web_enabled else None,
                rag_pipeline=rag_pipeline,
                memory_manager=memory_manager,
                max_agents=sidebar_settings.get("max_agents", 3),
                max_review_rounds=sidebar_settings.get("max_review_rounds", 2),
                research_timeout_seconds=sidebar_settings.get("research_timeout", 180),
            )
            try:
                with st.status("ResearchOS is planning, gathering evidence, and reviewing the report…", expanded=True) as status:
                    result = asyncio.run(orchestrator.research(
                        question,
                        objectives=objectives,
                        enable_memory=memory_enabled,
                        enable_rag=rag_enabled,
                        enable_web_search=web_enabled,
                    ))
                    history = ResearchHistoryStore(account_dir)
                    history.add_result(result)
                    status.update(label="Research complete", state="complete", expanded=False)
                st.session_state[f"last_research_{user.id}"] = result
            except Exception as exc:
                st.error(f"Research did not complete: {type(exc).__name__}: {str(exc)[:500]}")

    result = st.session_state.get(f"last_research_{user.id}")
    if result is not None:
        st.subheader("Final report")
        st.markdown(result.report_markdown)
        st.download_button(
            "Download report as Markdown",
            data=result.report_markdown.encode("utf-8"),
            file_name="researchos-report.md",
            mime="text/markdown",
            key=f"download_report_{user.id}",
        )
        with st.expander("Research trace", expanded=False):
            st.markdown("**Task assignments**")
            for task in result.trace.assignments:
                st.write(f"{task.task_id}: {task.question}")
            st.markdown("**Agent findings**")
            for finding in result.findings:
                with st.expander(f"{finding.role}: {finding.assignment}"):
                    st.write(finding.summary)
                    st.caption("Evidence IDs: " + (", ".join(finding.evidence_ids) or "none"))
                    if finding.uncertainties:
                        st.warning("\n".join(finding.uncertainties))
                    if finding.error:
                        st.error(finding.error)
            st.markdown("**Adversarial review**")
            for index, review in enumerate(result.trace.review_rounds, start=1):
                st.write(f"Review {index}: {review.summary or f'{len(review.issues)} issue(s)'}")
                for issue in review.issues:
                    st.write(f"{issue.severity.upper()}: {issue.concern} · {', '.join(issue.evidence_ids)}")
                if review.error:
                    st.warning(review.error)
            for event in result.trace.events:
                st.caption(event)
        if result.sources:
            with st.expander(f"Evidence collected ({len(result.sources)})", expanded=False):
                for source in result.sources:
                    label = f"[{source.source_id}] {source.title} · {source.kind}"
                    if source.url:
                        st.markdown(f"{label} — [{source.url}]({source.url})")
                    else:
                        st.markdown(label)
                    st.caption(source.text[:1000])
