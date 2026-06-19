"""OpenRouter BYOK adapter using OpenAI-compatible SDK calls."""

from __future__ import annotations

from typing import Any

from crupier.config import OPENROUTER_DEFAULT_HOST, ProviderSettings
from crupier.errors import (
    CrupierProviderAuthError,
    CrupierProviderRateLimitError,
    CrupierProviderUnavailableError,
)

from .common import provider_timeout_seconds, require_api_key
from .openai import OpenAIAdapter


class OpenRouterAdapter(OpenAIAdapter):
    provider = "openrouter"

    def _build_client(self) -> Any:
        api_key = require_api_key(self.settings, "OPENROUTER_API_KEY", provider=self.provider)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise CrupierProviderUnavailableError(
                "OpenRouter adapter requires the optional dependency: pip install 'crupier[openrouter]'.",
                retryable=False,
            ) from exc

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": self.settings.host or OPENROUTER_DEFAULT_HOST,
        }
        default_headers = _openrouter_headers(self.settings)
        if default_headers:
            kwargs["default_headers"] = default_headers
        timeout = provider_timeout_seconds(self.settings)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return OpenAI(**kwargs)

    def _raise_mapped_error(self, exc: Exception) -> None:
        name = exc.__class__.__name__.lower()
        if "auth" in name or "permission" in name:
            raise CrupierProviderAuthError(str(exc), provider=self.provider, env_key=self.settings.env_key) from exc
        if "ratelimit" in name or "rate_limit" in name:
            raise CrupierProviderRateLimitError(str(exc)) from exc
        raise CrupierProviderUnavailableError(f"OpenRouter request failed: {exc}") from exc


def _openrouter_headers(settings: ProviderSettings) -> dict[str, str]:
    headers: dict[str, str] = {}
    referer = settings.options.get("http_referer") or settings.options.get("referer")
    title = settings.options.get("title") or settings.options.get("app_title")
    if referer:
        headers["HTTP-Referer"] = str(referer)
    if title:
        headers["X-OpenRouter-Title"] = str(title)
    return headers
