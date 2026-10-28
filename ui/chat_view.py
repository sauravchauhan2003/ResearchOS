from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

from agents.chat_agent import ChatAgent
from agents.context import compact_blocks, compact_text
from agents.orchestrator import ResearchOrchestrator
from agents.routing import extract_research_command, should_create_report, should_use_statistical_simulation
from auth.user_store import User
from guardrails import (
    GuardrailViolation,
    preflight_guardrail,
    preflight_refusal,
    request_rate_limiter,
    should_save_answer_to_memory,
    validate_user_request,
)
from memory.memory_manager import MemoryManager
from observability import log_event, request_context
from providers.base import ProviderName
from providers.model_registry import ModelRegistry
from rag.retriever import RagPipeline
from research.conversations import ConversationStore
from research.history import ResearchHistoryStore
from tools.statistical_simulation import (
    StatisticalSimulationTool,
    parse_simulation_plan,
    simulation_plan_hash,
)
from tools.web_search import WebSearchTool
from tools.webpage_reader import WebpageReader
from ui.action_log import ActionLog, render_live_actions, render_saved_actions


DRAFT_CONVERSATION_PREFIX = "draft:"


def _new_draft_conversation_id() -> str:
    return f"{DRAFT_CONVERSATION_PREFIX}{uuid.uuid4()}"


def _persist_user_prompt(conversations: ConversationStore, conversation_id: str, prompt: str) -> tuple[str, dict[str, Any]]:
    if conversation_id.startswith(DRAFT_CONVERSATION_PREFIX) or conversations.get(conversation_id) is None:
        conversation_id = conversations.create()["conversation_id"]
    message = conversations.append_message(conversation_id, "user", prompt)
    return conversation_id, message


def _source_payload(sources) -> list[dict[str, Any]]:
    return [{
        "source_id": source.source_id,
        "kind": source.kind,
        "title": source.title,
        "text": source.text,
        "url": source.url,
        "published_at": source.published_at,
        "retrieved_at": source.retrieved_at,
    } for source in sources]


def _render_sources(raw_sources: list[dict[str, Any]]) -> None:
    if not raw_sources:
        return
    with st.expander(f"Sources ({len(raw_sources)})", expanded=False):
        for source in raw_sources:
            label = f"[{source.get('source_id', '?')}] {source.get('title', 'Source')} · {source.get('kind', 'source')}"
            url = source.get("url")
            if url:
                st.markdown(f"{label} — [{url}]({url})")
            else:
                st.markdown(label)
            excerpt = source.get("text", "")
            if excerpt:
                st.caption(compact_text(excerpt, 700))


def _recent_context(messages: list[dict[str, Any]]) -> str:
    return compact_blocks(
        (
            f"{message.get('role', 'assistant')}: {message.get('content', '')}"
            for message in messages[-6:]
            if message.get("role") in {"user", "assistant"}
        ),
        max_chars=1200,
        per_block_chars=400,
    )


def _memory_manager(account_dir: Path, settings) -> MemoryManager:
    return MemoryManager(account_dir, settings.data_dir / ".memory-signing-key")


def _provider(settings, store, user: User):
    selection = store.get_settings(user.id)
    if not selection.get("provider") or not selection.get("model"):
        raise ValueError("Choose a provider and model at the top of the page first.")
    registry = ModelRegistry.for_account(settings=settings, store=store, user_id=user.id)
    provider_name = ProviderName(selection["provider"])
    model_id = str(selection["model"])
    if not registry.model_matches_provider(provider_name, model_id):
        raise ValueError("The saved model does not match the selected provider. Fetch and choose a model on the main screen.")
    return registry.get(provider_name), model_id


def _process_request(
    user: User,
    settings,
    account_dir: Path,
    store,
    conversations: ConversationStore,
    conversation: dict[str, Any],
    user_message: dict[str, Any],
    actions: ActionLog,
) -> None:
    request_started = False

    def reply(content: str, metadata: dict[str, Any] | None = None) -> None:
        if request_started:
            actions.emit("Starting your request", "complete")
        conversations.append_message(
            conversation["conversation_id"],
            "assistant",
            content,
            metadata={**(metadata or {}), "actions": actions.snapshot()},
        )
        blocked = (metadata or {}).get("kind") == "guardrail"
        log_event("request_blocked" if blocked else "request_completed",
                  status="blocked" if blocked else "complete", output_chars=len(content))

    try:
        raw_request = validate_user_request(user_message.get("content", ""))
    except GuardrailViolation as exc:
        actions.emit("Checking request safety", "failed", "Request rejected by input limits")
        log_event("guardrail_triggered", status="blocked", reason="invalid_user_request",
                  input_chars=len(str(user_message.get("content", ""))))
        reply(str(exc), {"kind": "guardrail"})
        return
    block_reason = preflight_guardrail(raw_request)
    if block_reason:
        actions.emit("Checking request safety", "failed", "Blocked before memory, web, or document access")
        log_event("guardrail_triggered", status="blocked", reason=block_reason,
                  input_chars=len(raw_request))
        reply(preflight_refusal(block_reason), {"kind": "guardrail"})
        return
    if not request_rate_limiter.allow(user.id):
        actions.emit("Checking request safety", "failed", "Request rate limit reached")
        log_event("guardrail_triggered", status="blocked", reason="request_rate_limit")
        reply("You’ve sent requests too quickly. Wait a minute, then try again.", {"kind": "guardrail"})
        return

    actions.emit("Checking request safety", "complete", "No blocked request intent detected before retrieval")
    question, force_report = extract_research_command(raw_request)
    if not question:
        reply("Add a question after `/research` to start a detailed investigation.")
        return
    request_started = True
    actions.emit("Starting your request", "running")

    messages = conversation.get("messages", [])
    prior_turns = messages[:-1]
    prior_context = _recent_context(prior_turns)
    limits = st.session_state.get(f"research_limits_{user.id}", {})
    memory_enabled = st.session_state.get(f"enable_memory_{user.id}", True)
    rag_enabled = st.session_state.get(f"enable_rag_{user.id}", True)
    web_enabled = st.session_state.get(f"enable_web_{user.id}", True)
    try:
        log_event("request_started", status="running", input_chars=len(question))
        provider, model_id = _provider(settings, store, user)
        if should_use_statistical_simulation(question):
            actions.emit("Preparing a statistical simulation", "running")
            plan = StatisticalSimulationTool().build(provider, model_id, question)
            actions.emit("Statistical simulation ready for review", "complete")
            reply(plan.description, {"kind": "statistical_simulation_proposal", "plan": plan.to_dict(), "status": "pending"})
            return

        if should_create_report(question, force_report=force_report):
            memory = _memory_manager(account_dir, settings) if memory_enabled else None
            rag = RagPipeline(
                account_dir,
                chunk_size=limits.get("chunk_size", 1200),
                overlap=limits.get("chunk_overlap", 150),
            ) if rag_enabled else None
            orchestrator = ResearchOrchestrator(
                provider,
                model_id,
                web_search=WebSearchTool() if web_enabled else None,
                webpage_reader=WebpageReader() if web_enabled else None,
                rag_pipeline=rag,
                memory_manager=memory,
                max_agents=limits.get("max_agents", 2),
                max_review_rounds=limits.get("max_review_rounds", 1),
                research_timeout_seconds=limits.get("research_timeout", 180),
                on_step=actions.emit,
            )
            result = asyncio.run(orchestrator.research(
                question,
                enable_memory=memory_enabled,
                enable_rag=rag_enabled,
                enable_web_search=web_enabled,
                conversation_context=prior_context,
            ))
            ResearchHistoryStore(account_dir).add_result(result)
            actions.emit("Saving the completed research", "complete")
            reply(result.report_markdown, {"kind": "research_report", "sources": _source_payload(result.sources)})
            return

        memory = _memory_manager(account_dir, settings) if memory_enabled else None
        rag = RagPipeline(
            account_dir,
            chunk_size=limits.get("chunk_size", 1200),
            overlap=limits.get("chunk_overlap", 150),
        ) if rag_enabled else None
        answer = ChatAgent(
            provider,
            model_id,
            web_search=WebSearchTool() if web_enabled else None,
            webpage_reader=WebpageReader() if web_enabled else None,
            rag_pipeline=rag,
            memory_manager=memory,
        ).answer(
            question,
            prior_turns=prior_turns,
            use_memory=memory_enabled,
            use_documents=rag_enabled,
            use_web=web_enabled,
            on_step=actions.emit,
        )
        if memory is not None and len(question.split()) >= 3 and should_save_answer_to_memory(answer.text):
            actions.emit("Saving the answer to account memory", "running")
            try:
                memory.add(
                    user_query=question,
                    findings=answer.text[:1500],
                    conclusions=answer.text[:1500],
                    relevant_context="; ".join(source.title for source in answer.sources[:3]),
                )
                actions.emit("Saving the answer to account memory", "complete")
            except Exception as exc:
                actions.emit("Saving the answer to account memory", "failed", type(exc).__name__)
        elif memory is not None and len(question.split()) >= 3:
            log_event("answer_memory_write_skipped", status="skipped", reason="refusal_or_insufficient_answer")
        actions.emit("Finishing the response", "complete")
        reply(answer.text or "I couldn't produce an answer. Try rephrasing the question.",
              {"kind": "chat_answer", "sources": _source_payload(answer.sources),
               "memory_hits": answer.memory_hits})
    except Exception as exc:
        if type(exc).__name__ in {"StopException", "RerunException"}:
            raise
        log_event("request_failed", level=40, status="failed", error_type=type(exc).__name__)
        actions.emit("Completing the request", "failed", type(exc).__name__)
        reply(f"I couldn't complete that request ({type(exc).__name__}). Check the app logs for details, then try again.", {"kind": "error"})


def _render_simulation_proposal(user: User, conversations: ConversationStore,
                                conversation: dict[str, Any], message: dict[str, Any]) -> None:
    metadata = message.get("metadata", {})
    plan = metadata.get("plan", {})
    try:
        validated_plan = parse_simulation_plan(plan)
    except GuardrailViolation:
        st.error("This saved simulation plan is invalid. Ask the agent to build it again.")
        return
    st.markdown(message.get("content", "Statistical simulation proposal"))
    st.caption("The agent selected a bounded statistical method. Review these assumptions and parameters before approving this exact simulation.")
    st.write(f"Method: **{validated_plan.method}**")
    st.json(validated_plan.parameters)
    if metadata.get("status") != "pending":
        st.caption(f"Simulation status: {metadata.get('status')}")
        return
    plan_hash = simulation_plan_hash(validated_plan)[:12]
    approved = st.checkbox(
        "I reviewed these exact parameters and approve this simulation",
        key=f"simulation_approval_{user.id}_{message['message_id']}_{plan_hash}",
    )
    if st.button(
        "Run statistical simulation",
        type="primary",
        disabled=not approved,
        key=f"simulation_run_{user.id}_{message['message_id']}_{plan_hash}",
    ):
        try:
            result = StatisticalSimulationTool().run(validated_plan, approved=True)
        except GuardrailViolation as exc:
            st.warning(str(exc))
            return
        except Exception as exc:
            log_event("simulation_run_failed", level=40, user_id=user.id, status="failed",
                      tool=validated_plan.method, error_type=type(exc).__name__)
            st.error("The statistical simulation could not complete. Check the app logs and try again.")
            return
        log_event("simulation_run_completed", user_id=user.id, status=result.status,
                  tool=result.method, duration_ms=round(result.duration_seconds * 1000),
                  result_count=result.metrics.get("iterations"))
        conversations.update_message(
            conversation["conversation_id"],
            message["message_id"],
            metadata={"status": result.status, "run_id": result.run_id},
        )
        conversations.append_message(
            conversation["conversation_id"], "assistant", result.summary,
            metadata={"kind": "statistical_simulation_result", "details": result.to_dict()},
        )
        st.rerun()


def render_chat_view(user: User, settings, account_store) -> None:
    account_dir = settings.user_data_dir(user.id)
    account_dir.mkdir(parents=True, exist_ok=True)
    conversations = ConversationStore(account_dir)
    all_conversations = conversations.list_all()
    if not all_conversations:
        try:
            imported = conversations.import_legacy_research(ResearchHistoryStore(account_dir).list_all())
            if imported:
                all_conversations = conversations.list_all()
        except Exception:
            pass
    active_key = f"active_conversation_{user.id}"
    active_id = st.session_state.get(active_key)
    if st.session_state.pop("researchos_fresh_chat_user_id", None) == user.id:
        active_id = _new_draft_conversation_id()
        st.session_state[active_key] = active_id
    active_is_draft = isinstance(active_id, str) and active_id.startswith(DRAFT_CONVERSATION_PREFIX)
    if not active_is_draft and not any(item.get("conversation_id") == active_id for item in all_conversations):
        active = all_conversations[0] if all_conversations else {"conversation_id": _new_draft_conversation_id()}
        active_id = active["conversation_id"]
        st.session_state[active_key] = active_id

    with st.sidebar:
        st.title("ResearchOS")
        if st.button("＋  New chat", type="primary", use_container_width=True, key=f"new_chat_{user.id}"):
            st.session_state[active_key] = _new_draft_conversation_id()
            st.rerun()
        st.subheader("Chats")
        for item in all_conversations[:15]:
            label = item.get("title", "New chat")
            if len(label) > 42:
                label = label[:39] + "…"
            if st.button(label, key=f"open_chat_{user.id}_{item['conversation_id']}", use_container_width=True):
                st.session_state[active_key] = item["conversation_id"]
                st.rerun()
        with st.expander("Settings", expanded=False):
            if st.button("Sign out", key=f"sign_out_{user.id}"):
                log_event("sign_out", user_id=user.id, status="signed_out")
                account_store.revoke_login_session(st.session_state.get("researchos_session_token"))
                st.session_state.pop("researchos_user", None)
                st.session_state.pop("researchos_session_token", None)
                st.session_state["researchos_cookie_action"] = ("clear", "", False)
                st.rerun()
            st.caption("Choose the common provider key or your private account key at the top of the main screen.")
            max_agents = st.number_input("Research agents", min_value=1, max_value=3, value=2, step=1,
                                         key=f"max_agents_v2_{user.id}")
            max_reviews = st.number_input("Review passes", min_value=0, max_value=1, value=1, step=1,
                                          key=f"max_reviews_v2_{user.id}")
            research_timeout = st.number_input("Research timeout (seconds)", min_value=30, max_value=900,
                                               value=180, step=30, key=f"research_timeout_{user.id}")
            chunk_size = st.number_input("Document chunk size", min_value=300, max_value=4000,
                                         value=1200, step=100, key=f"chunk_size_{user.id}")
            chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=int(chunk_size) - 1,
                                            value=min(150, int(chunk_size) - 1), step=25,
                                            key=f"chunk_overlap_{user.id}")
            st.toggle("Use account memory", value=True, key=f"enable_memory_{user.id}")
            st.toggle("Use uploaded documents", value=True, key=f"enable_rag_{user.id}")
            st.toggle("Use web search", value=True, key=f"enable_web_{user.id}")
            st.session_state[f"research_limits_{user.id}"] = {
                "max_agents": int(max_agents), "max_review_rounds": int(max_reviews),
                "research_timeout": int(research_timeout), "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
            }
        with st.expander("Knowledge library", expanded=False):
            st.caption("Documents are private to your account and can inform chat answers.")
            upload = st.file_uploader("Add a PDF, TXT, or Markdown file",
                                      type=["pdf", "txt", "md", "markdown"], key=f"chat_upload_{user.id}")
            if st.button("Add to knowledge", disabled=upload is None, key=f"chat_ingest_{user.id}") and upload is not None:
                try:
                    with st.spinner("Indexing document…"):
                        indexed = RagPipeline(account_dir).ingest(upload.name, upload.getvalue())
                    log_event("document_ingested", user_id=user.id, status="complete",
                              tool="knowledge_library", input_chars=len(upload.getvalue()))
                    st.success(f"Added {indexed.chunk_count} chunks from {indexed.source_filename}.")
                except Exception as exc:
                    log_event("document_ingest_failed", level=40, user_id=user.id, status="failed",
                              tool="knowledge_library", error_type=type(exc).__name__)
                    st.error(f"Could not add document: {type(exc).__name__}: {str(exc)[:250]}")

    st.caption("Simple questions get concise answers. Use `/research` for a detailed report.")
    selected = account_store.get_settings(user.id)
    st.session_state["researchos_active_model_selection"] = selected
    provider_ready = False
    if selected.get("provider") and selected.get("model"):
        try:
            registry = ModelRegistry.for_account(settings=settings, store=account_store, user_id=user.id)
            provider_name = ProviderName(selected["provider"])
            provider_ready = (
                registry.configured(provider_name)
                and registry.model_matches_provider(provider_name, str(selected["model"]))
            )
        except ValueError:
            provider_ready = False
    if not selected.get("provider") or not selected.get("model"):
        st.info("Choose a provider and model above to start chatting.")
    elif not provider_ready:
        st.info("Choose a configured common key or save your own key at the top of the main screen to start chatting.")

    st.markdown(
        "<style>"
        "[data-testid='stMainBlockContainer']{max-width:900px;padding-top:1rem;}"
        "[data-testid='stChatMessage']{border-radius:14px;padding:0.6rem 0.9rem;}"
        "[data-testid='stExpander']{background:var(--secondary-background-color,rgba(128,128,128,.12));"
        "border:1px solid color-mix(in srgb,var(--text-color,#777) 18%,transparent);border-radius:8px;}"
        "[data-testid='stChatInput']{max-width:900px;margin:0 auto;}"
        "</style>",
        unsafe_allow_html=True,
    )
    conversation = conversations.get(active_id) or {
        "conversation_id": active_id,
        "title": "New chat",
        "messages": [],
    }
    messages = conversation.get("messages", [])
    if not messages:
        st.markdown("### What would you like to work on?")
        st.caption("Ask a question, continue earlier work, request a report, or ask for statistical modeling.")

    pending_user = bool(messages and messages[-1].get("role") == "user")
    for message in messages:
        role = message.get("role", "assistant")
        metadata = message.get("metadata", {})
        with st.chat_message(role):
            if metadata.get("kind") == "statistical_simulation_proposal":
                _render_simulation_proposal(user, conversations, conversation, message)
            elif metadata.get("kind") == "sandbox_proposal":
                st.markdown(message.get("content", "Legacy code proposal"))
                st.warning("This saved proposal used the retired Docker code runner. It cannot be executed now; ask the agent to rebuild it as a statistical simulation.")
            elif metadata.get("kind") == "statistical_simulation_result":
                st.markdown(message.get("content", "Simulation result"))
                details = metadata.get("details", {})
                if details.get("metrics"):
                    st.json(details["metrics"])
            elif metadata.get("kind") == "terminated":
                st.warning(message.get("content", "Request terminated before completion."))
            elif metadata.get("kind") == "sandbox_result":
                st.markdown(message.get("content", "Previous sandbox result"))
                st.caption("This result came from the retired Docker runner.")
            else:
                st.markdown(message.get("content", ""))
                _render_sources(metadata.get("sources", []))
            render_saved_actions(st, metadata.get("actions", []))

    if pending_user:
        process_key = f"processing_message_{user.id}"
        if st.session_state.get(process_key) == messages[-1].get("message_id"):
            action_area = None
            action_log = ActionLog(
                lambda entries: render_live_actions(action_area, entries) if action_area is not None else None
            )
            try:
                with st.chat_message("assistant"):
                    st.markdown("Working on your request…")
                    with st.expander("Agent actions", expanded=False):
                        action_area = st.empty()
                    action_log.emit("Checking request safety", "running")
                    with request_context(user.id, active_id):
                        _process_request(user, settings, account_dir, account_store,
                                         conversations, conversation, messages[-1], action_log)
            except BaseException as exc:
                if type(exc).__name__ != "StopException":
                    raise
                log_event("request_terminated", status="terminated", user_id=user.id,
                          conversation_id=active_id)
                action_log.terminate("Request stopped before completion.")
                conversations.append_message(
                    conversation["conversation_id"],
                    "assistant",
                    "Request terminated before completion. You can send a new message to continue.",
                    metadata={"kind": "terminated", "actions": action_log.snapshot()},
                )
                st.session_state.pop(process_key, None)
                st.rerun()
            st.session_state.pop(process_key, None)
            st.rerun()
        else:
            st.warning("This request has no saved response yet.")
            if st.button("Continue last request", type="primary", key=f"continue_{user.id}_{active_id}"):
                st.session_state[process_key] = messages[-1].get("message_id")
                st.rerun()
    prompt = st.chat_input(
        "Message ResearchOS…",
        disabled=pending_user or not provider_ready,
        max_chars=2000,
    )
    if prompt:
        active_id, new_message = _persist_user_prompt(conversations, active_id, prompt)
        st.session_state[active_key] = active_id
        st.session_state[f"processing_message_{user.id}"] = new_message["message_id"]
        st.rerun()
