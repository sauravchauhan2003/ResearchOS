from types import SimpleNamespace

from providers.base import ProviderName
from providers.gemini_provider import GeminiProvider
from providers.groq_provider import GroqProvider
from providers.model_registry import ModelRegistry


def test_groq_model_list_filters_non_generation_and_inactive_entries() -> None:
    client = SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=[
        SimpleNamespace(id="text-model", active=True, context_window=8192, max_completion_tokens=2048),
        SimpleNamespace(id="inactive", active=False, context_window=8192, max_completion_tokens=2048),
        SimpleNamespace(id="audio", active=True, context_window=None, max_completion_tokens=None),
    ])))
    models = GroqProvider("test-secret", client=client).list_models()
    assert [model.id for model in models] == ["text-model"]
    assert models[0].provider is ProviderName.GROQ


def test_gemini_list_uses_provider_reported_generation_actions() -> None:
    client = SimpleNamespace(models=SimpleNamespace(list=lambda: iter([
        SimpleNamespace(name="models/gemini-test", display_name="Gemini Test", supported_actions=["generateContent"], input_token_limit=1000),
        SimpleNamespace(name="models/embed-test", display_name="Embed Test", supported_actions=["embedContent"], input_token_limit=1000),
    ])))
    models = GeminiProvider("test-secret", client=client).list_models()
    assert [model.id for model in models] == ["gemini-test"]
    assert models[0].display_name == "Gemini Test"


def test_registry_keeps_other_provider_available_when_one_key_is_missing() -> None:
    registry = ModelRegistry(groq_key="groq-test", gemini_key=None)
    assert registry.configured(ProviderName.GROQ)
    assert not registry.configured(ProviderName.GEMINI)
    assert isinstance(registry.get(ProviderName.GROQ), GroqProvider)
