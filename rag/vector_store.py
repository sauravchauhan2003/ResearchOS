from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb


@dataclass(frozen=True, slots=True)
class StoredChunk:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, *, collection_name: str = "research_documents", client=None):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "purpose": "account-scoped research documents"},
        )

    def document_metadata(self, document_id: str) -> dict[str, Any] | None:
        result = self.collection.get(where={"document_id": document_id}, limit=1, include=["metadatas"])
        if not result["ids"]:
            return None
        return dict(result["metadatas"][0])

    def documents(self) -> list[dict[str, Any]]:
        result = self.collection.get(include=["metadatas"])
        by_id: dict[str, dict[str, Any]] = {}
        for metadata in result["metadatas"]:
            doc_id = metadata.get("document_id")
            if doc_id:
                by_id[doc_id] = {
                    "document_id": doc_id,
                    "source_filename": metadata.get("source_filename", ""),
                    "content_hash": metadata.get("content_hash", ""),
                    "chunk_count": metadata.get("chunk_count", 0),
                }
        return sorted(by_id.values(), key=lambda item: item["source_filename"].casefold())

    def chunks(self, *, document_ids: list[str] | None = None) -> list[StoredChunk]:
        """Read stored text for lexical retrieval, preserving account/document scope."""
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        result = self.collection.get(where=where, include=["documents", "metadatas"])
        return [
            StoredChunk(chunk_id, metadata["document_id"], document, dict(metadata))
            for chunk_id, document, metadata in zip(result["ids"], result["documents"], result["metadatas"])
        ]

    def add_chunks(self, chunks: list[StoredChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("One embedding is required for each chunk.")
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            embeddings=embeddings,
        )

    def delete_document(self, document_id: str) -> bool:
        result = self.collection.get(where={"document_id": document_id}, include=[])
        ids = result["ids"]
        if not ids:
            return False
        self.collection.delete(ids=ids)
        return True

    def delete_source(self, source_filename: str) -> int:
        result = self.collection.get(where={"source_filename": source_filename}, include=[])
        ids = result["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def query(self, query_embedding: list[float], *, limit: int = 5,
              document_ids: list[str] | None = None) -> list[tuple[StoredChunk, float]]:
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        output: list[tuple[StoredChunk, float]] = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]
        for chunk_id, document, metadata, distance in zip(ids, docs, metas, distances):
            output.append((StoredChunk(chunk_id, metadata["document_id"], document, dict(metadata)), float(distance)))
        return output
