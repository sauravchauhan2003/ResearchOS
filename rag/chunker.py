from __future__ import annotations

import re


def chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")
    normalized = text.strip()
    if not normalized:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", normalized) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        pieces = [sentence]
        if len(sentence) > chunk_size:
            words = sentence.split()
            pieces = []
            piece = ""
            for word in words:
                candidate = f"{piece} {word}".strip()
                if len(candidate) > chunk_size and piece:
                    pieces.append(piece)
                    piece = word
                else:
                    piece = candidate
            if piece:
                pieces.append(piece)
        for piece in pieces:
            candidate = f"{current} {piece}".strip()
            if current and len(candidate) > chunk_size:
                chunks.append(current)
                tail = current[-overlap:].lstrip() if overlap else ""
                current = f"{tail} {piece}".strip()
                if len(current) > chunk_size:
                    current = piece[-chunk_size:]
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks
