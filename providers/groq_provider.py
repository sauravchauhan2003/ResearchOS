from __future__ import annotations

import time

from groq import Groq

from guardrails import prepare_generation, validate_model_response
from observability import log_event
from providers.base import GenerationResponse, ModelInfo, ProviderName, ProviderResult
from providers.retry import retry_request


class GroqProvider:
    name = ProviderName.GROQ

    def __init__(self, api_key: str | None, *, timeout_seconds: float = 30.0, client=None):
        if not api_key and client is None:
            raise ValueError("GROQ_API_KEY is not configured.")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._client = client

    @property
    def client(self):
        if self._client is None:
            # Disable SDK retries so the shared policy below is the only retry loop.
            self._client = Groq(api_key=self._api_key, timeout=self._timeout_seconds, max_retries=0)
        return self._client

    def list_models(self) -> list[ModelInfo]:
        """List active text generation models using Groq's model catalog metadata."""
        response = retry_request(self.client.models.list, operation="Groq model listing")
        models: list[ModelInfo] = []
        for item in response.data:
            active = getattr(item, "active", True)
            context = getattr(item, "context_window", None)
            completion = getattr(item, "max_completion_tokens", None)
            # Groq's catalog includes non-generation models. Positive context and
            # completion limits identify chat-capable entries without a static list.
            if not active or not isinstance(context, int) or context <= 0:
                continue
            if not isinstance(completion, int) or completion <= 0:
                continue
            model_id = str(item.id)
            models.append(ModelInfo(model_id, model_id, self.name, context))
        return sorted(models, key=lambda model: model.id.casefold())

    def test_connection(self, model_id: str) -> ProviderResult:
        try:
            retry_request(
                lambda: self.client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": "Reply with OK."}],
                    max_tokens=2,
                ),
                operation="Groq connection test",
            )
            return ProviderResult(True, f"Connected to Groq model {model_id}.")
        except Exception as exc:  # SDK errors vary across authentication, quota, and model failures.
            message = str(exc).replace(self._api_key or "\0", "[redacted]")
            return ProviderResult(False, f"Groq connection test failed: {message[:400]}")

    def generate_text(self, model_id: str, prompt: str, *, system_instruction: str | None = None,
                      max_output_tokens: int = 2048, temperature: float = 0.2) -> GenerationResponse:
        try:
            system_instruction = prepare_generation(prompt, system_instruction, max_output_tokens, temperature)
        except Exception as exc:
            log_event("model_call_rejected", level=40, provider=self.name.value, model=model_id,
                      status="blocked", error_type=type(exc).__name__, input_chars=len(prompt) if isinstance(prompt, str) else 0)
            raise
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        started = time.perf_counter()
        try:
            response = retry_request(
                lambda: self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_output_tokens,
                    temperature=temperature,
                ),
                operation="Groq text generation",
            )
            text = validate_model_response(response.choices[0].message.content or "")
        except Exception as exc:
            log_event("model_call_failed", level=40, provider=self.name.value, model=model_id,
                      status="failed", error_type=type(exc).__name__, input_chars=len(prompt),
                      duration_ms=round((time.perf_counter() - started) * 1000))
            raise
        log_event("model_call_completed", provider=self.name.value, model=model_id, status="complete",
                  input_chars=len(prompt), output_chars=len(text),
                  duration_ms=round((time.perf_counter() - started) * 1000))
        usage = getattr(response, "usage", None)
        return GenerationResponse(
            text=text,
            model_id=model_id,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )
