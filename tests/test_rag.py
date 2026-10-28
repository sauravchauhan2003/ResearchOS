from pathlib import Path

import pytest

from rag.chunker import chunk_text
from rag.loaders import load_document
from rag.retriever import RagPipeline
from rag.text_cleaner import clean_text


class FakeEmbedder:
    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.casefold()
            vectors.append([
                float(lowered.count("battery") + 1),
                float(lowered.count("recycling") + 1),
                float(lowered.count("garden") + 1),
            ])
        return vectors


def test_load_clean_and_chunk_text() -> None:
    loaded = load_document("notes.txt", b"Title\r\n\r\nBattery\x00 recycling works.\tUseful evidence.")
    cleaned = clean_text(loaded.text)
    assert "\x00" not in cleaned
    chunks = chunk_text(cleaned, chunk_size=100, overlap=15)
    assert chunks and all(len(chunk) <= 100 for chunk in chunks)
    with pytest.raises(ValueError, match="Supported document"):
        load_document("script.py", b"print('no')")


def test_ingestion_is_idempotent_updateable_and_persistent(tmp_path: Path) -> None:
    pipeline = RagPipeline(tmp_path / "account-a", embedder=FakeEmbedder(), chunk_size=150, overlap=20)
    first = pipeline.ingest("research.md", b"Battery recycling recovers useful materials. Battery recycling reduces waste.")
    duplicate = pipeline.ingest("research.md", b"Battery recycling recovers useful materials. Battery recycling reduces waste.")
    assert first == duplicate
    assert len(pipeline.list_documents()) == 1
    assert pipeline.retrieve("battery recycling", limit=2)

    updated = pipeline.ingest("research.md", b"Garden compost supports soil health. Garden compost improves soil.")
    assert updated.document_id != first.document_id
    reopened = RagPipeline(tmp_path / "account-a", embedder=FakeEmbedder(), chunk_size=150, overlap=20)
    assert [item["document_id"] for item in reopened.list_documents()] == [updated.document_id]
    assert reopened.delete_document(updated.document_id)
    assert reopened.list_documents() == []


def test_documents_are_isolated_by_account(tmp_path: Path) -> None:
    first = RagPipeline(tmp_path / "account-a", embedder=FakeEmbedder())
    second = RagPipeline(tmp_path / "account-b", embedder=FakeEmbedder())
    first.ingest("private.md", b"Private battery research belongs to account alpha.")
    assert first.list_documents()
    assert second.list_documents() == []
