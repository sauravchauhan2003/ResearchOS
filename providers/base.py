from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ProviderName(StrEnum):
    GROQ = "Groq"
    GEMINI = "Gemini"


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
    display_name: str
    provider: ProviderName
    context_window: int | None = None


@dataclass(frozen=True, slots=True)
class ProviderResult:
    ok: bool
    message: str


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    text: str
    model_id: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelProvider(Protocol):
    name: ProviderName

    def list_models(self) -> list[ModelInfo]: ...

    def test_connection(self, model_id: str) -> ProviderResult: ...

    def generate_text(self, model_id: str, prompt: str, *, system_instruction: str | None = None,
                      max_output_tokens: int = 2048, temperature: float = 0.2) -> GenerationResponse: ...
