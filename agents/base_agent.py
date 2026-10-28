from __future__ import annotations

import json
import re
from typing import Any

from providers.base import ModelProvider


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Provider did not return a JSON object.")
    value = json.loads(candidate[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Provider response must be a JSON object.")
    return value


class BaseAgent:
    def __init__(self, provider: ModelProvider, model_id: str):
        self.provider = provider
        self.model_id = model_id

    def complete(self, prompt: str, *, system_instruction: str, max_output_tokens: int = 2048) -> str:
        response = self.provider.generate_text(
            self.model_id,
            prompt,
            system_instruction=system_instruction,
            max_output_tokens=max_output_tokens,
            temperature=0.2,
        )
        if not response.text.strip():
            raise ValueError("The selected model returned an empty response.")
        return response.text.strip()
