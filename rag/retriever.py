from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from rag.chunker import chunk_text
from rag.embeddings import FastEmbedder
from rag.hybrid_search import bm25_rank, reciprocal_rank_fusion
from rag.loaders import load_document
from rag.text_cleaner import clean_text
from rag.vector_store import ChromaVectorStore, StoredChunk


@dataclass(frozen=True, slots=True)
class IngestedDocument:
    document_id: str
    source_filename: str
    chunk_count: int
    content_hash: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    source_filename: str
    text: str
    distance: float | None
    metadata: dict


class RagPipeline:
    def __init__(self, account_dir: Path, *, embedder=None, chunk_size: int = 1200, overlap: int = 150):
        self.store = ChromaVectorStore(Path(account_dir) / "chroma_db")
        self.embedder = embedder or FastEmbedder()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest(self, filename: str, content: bytes) -> IngestedDocument:
        loaded = load_document(filename, content)
        cleaned = clean_text(loaded.text)
        pieces = chunk_text(cleaned, chunk_size=self.chunk_size, overlap=self.overlap)
        if not pieces:
            raise ValueError("Document contains no text after cleaning.")
        # Replacing a same-named source updates it; an identical re-upload remains idempotent.
        existing = next((doc for doc in self.store.documents() if doc["source_filename"].casefold() == loaded.source_filename.casefold()), None)
        if existing and existing["content_hash"] == loaded.content_hash:
            return IngestedDocument(existing["document_id"], loaded.source_filename, existing["chunk_count"], loaded.content_hash)
        if existing:
            self.store.delete_document(existing["document_id"])
        document_id = hashlib.sha256(f"{loaded.source_filename.casefold()}:{loaded.content_hash}".encode("utf-8")).hexdigest()
        vectors = self.embedder.embed(pieces)
        chunks: list[StoredChunk] = []
        chunk_count = len(pieces)
        for index, text in enumerate(pieces):
            chunk_id = hashlib.sha256(f"{document_id}:{index}:{text}".encode("utf-8")).hexdigest()
            chunks.append(StoredChunk(chunk_id, document_id, text, {
                "document_id": document_id,
                "source_filename": loaded.source_filename,
                "chunk_index": index,
                "chunk_count": chunk_count,
                "content_hash": loaded.content_hash,
            }))
        self.store.add_chunks(chunks, vectors)
        return IngestedDocument(document_id, loaded.source_filename, chunk_count, loaded.content_hash)

    def retrieve(self, query: str, *, limit: int = 5, document_ids: list[str] | None = None) -> list[RetrievedChunk]:
        if limit <= 0:
            return []
        corpus = self.store.chunks(document_ids=document_ids)
        if not corpus:
            return []

        candidate_limit = min(len(corpus), max(20, limit * 5))
        query_vector = self.embedder.embed([query])[0]
        vector_results = self.store.query(query_vector, limit=candidate_limit, document_ids=document_ids)
        lexical_results = bm25_rank(query, corpus, limit=candidate_limit)
        fused = reciprocal_rank_fusion((vector_results, lexical_results), limit=limit)

        vector_distances = {chunk.chunk_id: distance for chunk, distance in vector_results}
        lexical_scores = {chunk.chunk_id: score for chunk, score in lexical_results}
        output: list[RetrievedChunk] = []
        for chunk in fused:
            metadata = dict(chunk.metadata)
            metadata["retrieval_method"] = (
                "hybrid" if chunk.chunk_id in vector_distances and chunk.chunk_id in lexical_scores
                else "vector" if chunk.chunk_id in vector_distances else "bm25"
            )
            if chunk.chunk_id in lexical_scores:
                metadata["bm25_score"] = lexical_scores[chunk.chunk_id]
            output.append(RetrievedChunk(
                chunk.chunk_id, chunk.document_id, chunk.metadata["source_filename"], chunk.text,
                vector_distances.get(chunk.chunk_id), metadata,
            ))
        return output

    def delete_document(self, document_id: str) -> bool:
        return self.store.delete_document(document_id)

    def list_documents(self) -> list[dict]:
        return self.store.documents()
