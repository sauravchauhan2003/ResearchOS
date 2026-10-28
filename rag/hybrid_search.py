from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from rag.vector_store import StoredChunk


_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


def bm25_rank(query: str, chunks: Iterable[StoredChunk], *, limit: int) -> list[tuple[StoredChunk, float]]:
    """Rank chunks lexically with BM25; no external service or model is needed."""
    corpus = list(chunks)
    query_terms = set(tokenize(query))
    if not corpus or not query_terms or limit <= 0:
        return []

    tokenized = [tokenize(chunk.text) for chunk in corpus]
    document_frequency: Counter[str] = Counter()
    for terms in tokenized:
        document_frequency.update(set(terms))

    average_length = sum(map(len, tokenized)) / len(tokenized) or 1.0
    document_count = len(corpus)
    k1, b = 1.5, 0.75
    ranked: list[tuple[StoredChunk, float]] = []
    for chunk, terms in zip(corpus, tokenized):
        frequencies = Counter(terms)
        length_norm = k1 * (1.0 - b + b * len(terms) / average_length)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log1p((document_count - df + 0.5) / (df + 0.5))
            score += inverse_frequency * (frequency * (k1 + 1.0)) / (frequency + length_norm)
        if score > 0:
            ranked.append((chunk, score))

    ranked.sort(key=lambda item: (-item[1], item[0].chunk_id))
    return ranked[:limit]


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[tuple[StoredChunk, float]]], *, limit: int, rank_constant: int = 60
) -> list[StoredChunk]:
    """Combine ranked lists without comparing incomparable BM25 and cosine scores."""
    fused: dict[str, tuple[StoredChunk, float]] = {}
    for ranking in rankings:
        for rank, (chunk, _score) in enumerate(ranking, start=1):
            previous = fused.get(chunk.chunk_id, (chunk, 0.0))
            fused[chunk.chunk_id] = (chunk, previous[1] + 1.0 / (rank_constant + rank))
    ordered = sorted(fused.values(), key=lambda item: (-item[1], item[0].chunk_id))
    return [chunk for chunk, _score in ordered[:max(0, limit)]]
