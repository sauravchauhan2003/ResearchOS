from __future__ import annotations

from rag.retriever import RetrievedChunk


def format_document_citation(chunk: RetrievedChunk) -> str:
    """Return a clearly labeled citation for a retrieved uploaded document chunk."""
    return f"{chunk.source_filename} · chunk {chunk.metadata.get('chunk_index', '?')}"


def evidence_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No uploaded-document evidence was retrieved."
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        citation = format_document_citation(chunk)
        blocks.append(f"[Uploaded document evidence {index}: {citation}]\n{chunk.text}")
    return "\n\n".join(blocks)
