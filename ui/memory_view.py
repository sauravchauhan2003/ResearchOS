from __future__ import annotations

import streamlit as st

from auth.user_store import User
from memory.memory_manager import MemoryManager
from rag.retriever import RagPipeline
from research.history import ResearchHistoryStore


def render_memory_view(user: User, settings) -> None:
    st.header("Memory & knowledge")
    account_dir = settings.user_data_dir(user.id)
    manager = MemoryManager(account_dir, settings.data_dir / ".memory-signing-key")
    memories = manager.search(st.text_input("Search account memories", key=f"memory_query_{user.id}"), limit=100)
    if manager.last_warning:
        st.warning(manager.last_warning)
    if memories:
        labels = {f"{record.timestamp[:10]} · {record.user_query[:100]} · {record.memory_id[:8]}": record.memory_id for record in memories}
        selected_labels = st.multiselect("Select memories to delete", list(labels), key=f"delete_memories_{user.id}")
        if st.button("Delete selected memories", disabled=not selected_labels, key=f"delete_memories_button_{user.id}"):
            count = sum(manager.delete(labels[label]) for label in selected_labels)
            st.success(f"Deleted {count} memory record(s).")
            st.rerun()
        for record in memories:
            with st.expander(f"{record.timestamp[:10]} · {record.user_query}"):
                if record.findings:
                    st.markdown("**Findings**")
                    st.write(record.findings)
                if record.conclusions:
                    st.markdown("**Conclusions**")
                    st.write(record.conclusions)
                if record.claims:
                    st.markdown("**Claims and supporting sources**")
                    for claim in record.claims:
                        st.write(f"• {claim.text} — {', '.join(claim.sources)}")
                if record.unresolved_questions:
                    st.markdown("**Unresolved questions**")
                    for question in record.unresolved_questions:
                        st.write(f"• {question}")
    else:
        st.info("No matching memory records.")

    st.divider()
    st.subheader("Research history")
    history = ResearchHistoryStore(account_dir)
    try:
        records = history.list_all()
    except ValueError as exc:
        records = []
        st.error(str(exc))
    if records:
        labels = {f"{item['created_at'][:10]} · {item['question'][:110]} · {item['research_id'][:8]}": item["research_id"] for item in records}
        selected = st.multiselect("Select history entries to delete", list(labels), key=f"delete_history_{user.id}")
        if st.button("Delete selected history", disabled=not selected, key=f"delete_history_button_{user.id}"):
            count = sum(history.delete(labels[label]) for label in selected)
            st.success(f"Deleted {count} history record(s).")
            st.rerun()
        confirm_clear = st.checkbox("Confirm clearing all research history", key=f"confirm_clear_history_{user.id}")
        if st.button("Clear all research history", disabled=not confirm_clear, key=f"clear_history_{user.id}"):
            count = history.clear()
            st.success(f"Cleared {count} research history record(s).")
            st.rerun()
        for item in records:
            with st.expander(f"{item['created_at'][:10]} · {item['question']}"):
                st.markdown(item["report_markdown"])
                st.caption(f"{len(item.get('sources', []))} source(s) · {len(item.get('findings', []))} agent result(s)")
    else:
        st.info("No saved research sessions.")

    st.divider()
    st.subheader("Uploaded document library")
    try:
        pipeline = RagPipeline(account_dir)
        documents = pipeline.list_documents()
    except Exception as exc:
        documents = []
        st.error(f"Could not open this account's vector library: {type(exc).__name__}: {str(exc)[:250]}")
    if documents:
        selected_docs = st.multiselect(
            "Select documents to delete",
            options=[item["document_id"] for item in documents],
            format_func=lambda document_id: next(
                (f"{item['source_filename']} ({item['chunk_count']} chunks)" for item in documents if item["document_id"] == document_id),
                document_id,
            ),
            key=f"delete_documents_{user.id}",
        )
        if st.button("Delete selected documents", disabled=not selected_docs, key=f"delete_documents_button_{user.id}"):
            count = sum(pipeline.delete_document(document_id) for document_id in selected_docs)
            st.success(f"Deleted {count} document(s) from this account's library.")
            st.rerun()
        for item in documents:
            st.write(f"• {item['source_filename']} · {item['chunk_count']} chunks")
    else:
        st.info("No documents have been indexed for this account.")
