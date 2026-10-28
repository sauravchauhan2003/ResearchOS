from __future__ import annotations

from providers.base import ModelProvider, ProviderName
from providers.gemini_provider import GeminiProvider
from providers.groq_provider import GroqProvider


class ModelRegistry:
    def __init__(self, *, groq_key: str | None, gemini_key: str | None, timeout_seconds: float = 30.0):
        self._keys = {ProviderName.GROQ: groq_key, ProviderName.GEMINI: gemini_key}
        self._timeout_seconds = timeout_seconds

    @classmethod
    def for_account(cls, *, settings, store, user_id: str) -> "ModelRegistry":
        """Resolve only the key source this account selected for each provider."""
        account_settings = store.get_settings(user_id)
        key_modes = account_settings.get("key_modes", {})

        def selected_key(provider: ProviderName, shared_key: str | None) -> str | None:
            mode = key_modes.get(provider.value) if isinstance(key_modes, dict) else None
            if mode == "shared":
                return shared_key
            personal_key = store.get_api_key(user_id, provider.value)
            if mode == "personal":
                return personal_key
            if mode not in {"shared", "personal"}:
                mode = "personal" if personal_key else "shared"
            return shared_key if mode == "shared" else personal_key

        return cls(
            groq_key=selected_key(ProviderName.GROQ, settings.groq_api_key),
            gemini_key=selected_key(ProviderName.GEMINI, settings.gemini_api_key),
            timeout_seconds=settings.provider_timeout_seconds,
        )

    def configured(self, provider: ProviderName) -> bool:
        return bool(self._keys[provider])

    @staticmethod
    def model_matches_provider(provider: ProviderName, model_id: str) -> bool:
        normalized = model_id.strip().casefold().removeprefix("models/")
        if provider is ProviderName.GEMINI:
            return normalized.startswith("gemini-")
        return bool(normalized) and not normalized.startswith("gemini-")

    def get(self, provider: ProviderName) -> ModelProvider:
        key = self._keys[provider]
        if not key:
            raise ValueError(
                f"No API key is available for {provider.value}. Choose the common key or save your own key on the main screen."
            )
        adapter = GroqProvider if provider is ProviderName.GROQ else GeminiProvider
        return adapter(key, timeout_seconds=self._timeout_seconds)
