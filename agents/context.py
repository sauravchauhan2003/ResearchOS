from __future__ import annotations

import re
from collections.abc import Iterable


_WHITESPACE = re.compile(r"\s+")
_COMPACTED = " …[compacted]"


def compact_text(text: str, max_chars: int) -> str:
    """Normalize whitespace and cap a prompt field without changing its beginning."""
    normalized = _WHITESPACE.sub(" ", str(text)).strip()
    if max_chars <= 0:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= len(_COMPACTED):
        return _COMPACTED[:max_chars]
    prefix = normalized[:max_chars - len(_COMPACTED)].rsplit(" ", 1)[0]
    return (prefix or normalized[:max_chars - len(_COMPACTED)]) + _COMPACTED


def compact_blocks(blocks: Iterable[str], *, max_chars: int, per_block_chars: int) -> str:
    """Compact a list of labeled context blocks under one shared character budget."""
    cleaned = [_WHITESPACE.sub(" ", str(block)).strip() for block in blocks]
    cleaned = [block for block in cleaned if block]
    if not cleaned or max_chars <= 0:
        return ""

    separator_cost = 2 * (len(cleaned) - 1)
    if len(cleaned) * per_block_chars + separator_cost <= max_chars:
        block_budget = per_block_chars
    else:
        block_budget = max(1, (max_chars - separator_cost) // len(cleaned))

    compacted = [compact_text(block, min(per_block_chars, block_budget)) for block in cleaned]
    return "\n\n".join(compacted)
