"""Adapter factory."""

from __future__ import annotations

from .anthropic import AnthropicAdapter
from .base import ProviderAdapter
from .google import GoogleAdapter
from .ollama import OllamaAdapter
from .nan import NaNAdapter
from .openai_compatible import OpenAICompatibleAdapter
from .openai import OpenAIAdapter
from .openrouter import OpenRouterAdapter
from crupier.config import CrupierConfig


def build_default_adapters(config: CrupierConfig) -> dict[str, ProviderAdapter]:
    adapters: dict[str, ProviderAdapter] = {}
    for provider, settings in config.providers.items():
        if not settings.enabled:
            continue
        if provider == "openai":
            adapters[provider] = OpenAIAdapter(settings)
        elif provider == "anthropic":
            adapters[provider] = AnthropicAdapter(settings)
        elif provider == "google":
            adapters[provider] = GoogleAdapter(settings)
        elif provider == "ollama":
            adapters[provider] = OllamaAdapter(settings)
        elif provider == "openrouter":
            adapters[provider] = OpenRouterAdapter(settings)
        elif provider == "nan":
            adapters[provider] = NaNAdapter(settings)
        elif provider == "inference" or settings.mode == "openai_compatible":
            adapters[provider] = OpenAICompatibleAdapter(settings, provider=provider)
    return adapters
