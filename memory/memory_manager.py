from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from memory.models import MEMORY_SCHEMA_VERSION, MemoryRecord
from memory.pickle_store import PickleStore


TOKEN_RE = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)
STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been", "before", "being",
    "between", "both", "but", "can", "could", "did", "does", "doing", "down", "each",
    "for", "from", "have", "having", "here", "how", "into", "its", "just", "more", "most",
    "not", "now", "off", "once", "only", "other", "our", "out", "over", "same", "should",
    "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "through", "too", "under", "until", "very", "was", "were", "what", "when",
    "where", "which", "while", "who", "why", "will", "with", "would", "you", "your", "give",
    "tell", "show", "find", "make", "write", "create", "summary", "summarize", "report", "please",
}
FOLLOW_UP_RE = re.compile(
    r"\b(?:it|they|them|that|those|this|same|continue|more|another|former|latter|previous|earlier|"
    r"option|choice|instead|what about|how about)\b",
    re.IGNORECASE,
)


def memory_search_query(question: str, conversation_context: str = "") -> str:
    """Use earlier turns only when the current question clearly depends on them."""
    context = conversation_context.strip()
    if context and FOLLOW_UP_RE.search(question):
        return f"{question} {context}"
    return question


class MemoryManager:
    def __init__(self, account_dir: Path, signing_key_path: Path, *, signing_key: bytes | None = None):
        self.store = PickleStore(account_dir / "memory" / "research_memory.pkl", signing_key_path, signing_key)
        self.last_warning: str | None = None

    def _read_locked(self) -> list[MemoryRecord]:
        payload, warning = self.store.load_unlocked()
        self.last_warning = warning
        return [MemoryRecord.model_validate(item) for item in payload["records"]]

    def _write_locked(self, records: list[MemoryRecord]) -> None:
        self.store.write_unlocked({
            "schema_version": MEMORY_SCHEMA_VERSION,
            "records": [record.model_dump(mode="json") for record in records],
        })

    def add(self, *, research_session_id: str | None = None, user_query: str, findings: str = "",
            claims: list[dict[str, Any]] | None = None, conclusions: str = "",
            unresolved_questions: list[str] | None = None, user_preferences: dict[str, Any] | None = None,
            relevant_context: str = "", supersedes: list[str] | None = None) -> MemoryRecord:
        record = MemoryRecord(
            research_session_id=research_session_id or str(uuid.uuid4()),
            user_query=user_query,
            findings=findings,
            claims=claims or [],
            conclusions=conclusions,
            unresolved_questions=unresolved_questions or [],
            user_preferences=user_preferences or {},
            relevant_context=relevant_context,
            supersedes=supersedes or [],
        )
        with self.store.lock:
            records = self._read_locked()
            records.append(record)
            self._write_locked(records)
        return record

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self.store.lock:
            return next((record for record in self._read_locked() if record.memory_id == memory_id), None)

    def update(self, memory_id: str, **changes: Any) -> MemoryRecord:
        allowed = set(MemoryRecord.model_fields) - {"memory_id", "timestamp"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unsupported memory fields: {', '.join(sorted(unknown))}")
        with self.store.lock:
            records = self._read_locked()
            for index, record in enumerate(records):
                if record.memory_id == memory_id:
                    updated = record.model_copy(update=changes)
                    records[index] = updated
                    self._write_locked(records)
                    return updated
        raise KeyError(memory_id)

    def search(self, query: str = "", *, limit: int = 20) -> list[MemoryRecord]:
        if limit < 1:
            return []
        records, self.last_warning = self.store.read()
        if not query.strip():
            return sorted(records, key=lambda item: item.timestamp, reverse=True)[:limit]
        terms = {term for term in TOKEN_RE.findall(query.casefold()) if term not in STOPWORDS}
        if not terms:
            return []
        field_weights = {
            "user_query": 3.0,
            "conclusions": 2.0,
            "findings": 1.5,
            "claims": 1.5,
            "unresolved_questions": 1.0,
            "relevant_context": 0.75,
        }
        indexed: list[tuple[MemoryRecord, dict[str, set[str]], set[str]]] = []
        document_frequency: Counter[str] = Counter()
        for record in records:
            fields = {
                "user_query": record.user_query,
                "conclusions": record.conclusions,
                "findings": record.findings,
                "claims": " ".join(claim.text for claim in record.claims),
                "unresolved_questions": " ".join(record.unresolved_questions),
                "relevant_context": record.relevant_context,
            }
            token_fields = {
                name: set(TOKEN_RE.findall(value.casefold())) - STOPWORDS
                for name, value in fields.items()
            }
            all_words = set().union(*token_fields.values())
            indexed.append((record, token_fields, all_words))
            document_frequency.update(all_words)

        total_records = len(indexed)
        idf = {
            term: math.log(1 + (total_records - document_frequency[term] + 0.5)
                           / (document_frequency[term] + 0.5))
            for term in terms
        }
        total_weight = sum(idf[term] * max(field_weights.values()) for term in terms) or 1.0
        minimum_matches = min(2, len(terms))
        scored: list[tuple[float, str, MemoryRecord]] = []
        for record, token_fields, words in indexed:
            matches = terms & words
            if len(matches) < minimum_matches:
                continue
            weighted_score = sum(
                idf[term] * max(
                    (field_weights[name] for name, field_terms in token_fields.items() if term in field_terms),
                    default=0.0,
                )
                for term in matches
            )
            relevance = weighted_score / total_weight
            if relevance >= 0.30:
                scored.append((relevance, record.timestamp, record))
        return [item[2] for item in sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[:limit]]

    def delete(self, memory_id: str) -> bool:
        with self.store.lock:
            records = self._read_locked()
            remaining = [record for record in records if record.memory_id != memory_id]
            if len(remaining) == len(records):
                return False
            self._write_locked(remaining)
            return True

    def list_all(self) -> list[MemoryRecord]:
        return self.search(limit=10000)
