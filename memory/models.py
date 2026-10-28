from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


MEMORY_SCHEMA_VERSION = 1


class Claim(BaseModel):
    text: str
    sources: list[str] = Field(default_factory=list)


class MemoryRecord(BaseModel):
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    research_session_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_query: str
    findings: str = ""
    claims: list[Claim] = Field(default_factory=list)
    conclusions: str = ""
    unresolved_questions: list[str] = Field(default_factory=list)
    user_preferences: dict[str, Any] = Field(default_factory=dict)
    relevant_context: str = ""
    supersedes: list[str] = Field(default_factory=list)
