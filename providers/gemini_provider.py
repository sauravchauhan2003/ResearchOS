from __future__ import annotations

import time

from google import genai
from google.genai import types

from guardrails import prepare_generation, validate_model_response
from observability import log_event
from providers.base import GenerationResponse, ModelInfo, ProviderName, ProviderResult
from providers.retry import retry_request


class GeminiProvider:
    name = ProviderName.GEMINI

    def __init__(self, api_key: str | None, *, timeout_seconds: float = 30.0, client=None):
        if not api_key and client is None:
            raise ValueError("GEMINI_API_KEY is not configured.")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(
                    timeout=int(self._timeout_seconds * 1000),
                    # One SDK attempt; retries are handled consistently below.
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        return self._client

    def list_models(self) -> list[ModelInfo]:
        models: list[ModelInfo] = []
        items = retry_request(
            lambda: list(self.client.models.list()),
            operation="Gemini model listing",
        )
        for item in items:
            actions = getattr(item, "supported_actions", None) or []
            if "generateContent" not in actions:
                continue
            raw_name = str(getattr(item, "name", ""))
            model_id = raw_name.removeprefix("models/")
            if not model_id:
                continue
            context = getattr(item, "input_token_limit", None)
            display = str(getattr(item, "display_name", None) or model_id)
            models.append(ModelInfo(model_id, display, self.name, context if isinstance(context, int) else None))
        return sorted(models, key=lambda model: model.display_name.casefold())

    def test_connection(self, model_id: str) -> ProviderResult:
        try:
            retry_request(
                lambda: self.client.models.generate_content(model=model_id, contents="Reply with OK."),
                operation="Gemini connection test",
            )
            return ProviderResult(True, f"Connected to Gemini model {model_id}.")
        except Exception as exc:
            message = str(exc).replace(self._api_key or "\0", "[redacted]")
            return ProviderResult(False, f"Gemini connection test failed: {message[:400]}")

    def generate_text(self, model_id: str, prompt: str, *, system_instruction: str | None = None,
                      max_output_tokens: int = 2048, temperature: float = 0.2) -> GenerationResponse:
        try:
            system_instruction = prepare_generation(prompt, system_instruction, max_output_tokens, temperature)
        except Exception as exc:
            log_event("model_call_rejected", level=40, provider=self.name.value, model=model_id,
                      status="blocked", error_type=type(exc).__name__, input_chars=len(prompt) if isinstance(prompt, str) else 0)
            raise
        started = time.perf_counter()
        try:
            response = retry_request(
                lambda: self.client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                    ),
                ),
                operation="Gemini text generation",
            )
            text = validate_model_response(response.text or "")
        except Exception as exc:
            log_event("model_call_failed", level=40, provider=self.name.value, model=model_id,
                      status="failed", error_type=type(exc).__name__, input_chars=len(prompt),
                      duration_ms=round((time.perf_counter() - started) * 1000))
            raise
        log_event("model_call_completed", provider=self.name.value, model=model_id, status="complete",
                  input_chars=len(prompt), output_chars=len(text),
                  duration_ms=round((time.perf_counter() - started) * 1000))
        usage = getattr(response, "usage_metadata", None)
        return GenerationResponse(
            text=text,
            model_id=model_id,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )
