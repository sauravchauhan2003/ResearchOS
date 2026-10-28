from __future__ import annotations

from collections.abc import Callable

import pytest

import providers.gemini_provider as gemini_module
import providers.groq_provider as groq_module
from providers.retry import MAX_RETRIES, retry_request


def test_retry_request_doubles_delay_for_rate_limits_and_other_errors() -> None:
    failures = iter([RuntimeError("429 rate limit"), ValueError("temporary error")])
    delays: list[float] = []
    calls = 0

    def request() -> str:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise next(failures)
        return "success"

    assert retry_request(request, operation="test", sleep=delays.append) == "success"
    assert calls == 3
    assert delays == [1.0, 2.0]


def test_retry_request_makes_at_most_eight_retries_then_reraises() -> None:
    delays: list[float] = []
    calls = 0

    def request() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("temporary provider error")

    with pytest.raises(RuntimeError, match="temporary provider error"):
        retry_request(request, operation="test", sleep=delays.append)

    assert calls == MAX_RETRIES + 1
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]


def test_groq_sdk_retries_are_disabled_in_favor_of_shared_policy(monkeypatch) -> None:
    from types import SimpleNamespace

    captured = {}

    def make_client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(groq_module, "Groq", make_client)
    groq_module.GroqProvider("key").client
    assert captured["max_retries"] == 0


def test_gemini_sdk_is_configured_for_one_attempt(monkeypatch) -> None:
    from types import SimpleNamespace

    captured = {}

    def make_client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(gemini_module.genai, "Client", make_client)
    gemini_module.GeminiProvider("key").client
    assert captured["http_options"].retry_options.attempts == 1


@pytest.mark.parametrize("provider_module", [groq_module, gemini_module])
def test_provider_generation_uses_retry_policy(monkeypatch, provider_module) -> None:
    calls = 0
    delays: list[float] = []
    original_retry: Callable = retry_request

    def fast_retry(request, **kwargs):
        return original_retry(request, sleep=delays.append, **kwargs)

    monkeypatch.setattr(provider_module, "retry_request", fast_retry)

    class Completion:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("provider temporarily unavailable")
            if provider_module is groq_module:
                from types import SimpleNamespace
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
                    usage=None,
                )
            from types import SimpleNamespace
            return SimpleNamespace(text="answer", usage_metadata=None)

    if provider_module is groq_module:
        from types import SimpleNamespace
        client = SimpleNamespace(chat=SimpleNamespace(completions=Completion()))
        provider = groq_module.GroqProvider("key", client=client)
    else:
        from types import SimpleNamespace
        client = SimpleNamespace(models=SimpleNamespace(generate_content=Completion().create))
        provider = gemini_module.GeminiProvider("key", client=client)

    assert provider.generate_text("model", "prompt").text == "answer"
    assert calls == 3
    assert delays == [1.0, 2.0]
