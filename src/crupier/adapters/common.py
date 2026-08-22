"""Shared adapter helpers."""

from __future__ import annotations

import json
from typing import Any, Literal, NoReturn

from crupier.config import ProviderSettings
from crupier.errors import (
    CrupierProviderAuthError,
    CrupierProviderRateLimitError,
    CrupierProviderUnavailableError,
)
from crupier.models import RequestEnvelope

HttpErrorClass = Literal["auth", "rate_limit", "permanent", "transient"]

_AUTH_STATUS_CODES = frozenset({401, 403})
_TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_AUTH_ERROR_TOKENS = ("auth", "permission", "forbidden")
_RATE_LIMIT_ERROR_TOKENS = ("ratelimit", "rate_limit", "resourceexhausted")


def classify_http_error(exc: BaseException, *, status_code: object | None = None) -> HttpErrorClass:
    """Classify a provider failure, preferring HTTP status over known SDK class names.

    Auth and rate-limit statuses keep their dedicated types. Known transient
    statuses are retryable; every other HTTP 4xx/5xx is deterministic. When no
    HTTP status exists, conservative SDK class-name matching provides the
    fallback and unknown transport failures remain transient.
    """
    status = _status_code(status_code) or _status_code(getattr(exc, "status_code", None))
    status = status or _status_code(getattr(exc, "code", None))
    name = exc.__class__.__name__.lower()
    if status in _AUTH_STATUS_CODES:
        return "auth"
    if status == 429:
        return "rate_limit"
    if status in _TRANSIENT_STATUS_CODES:
        return "transient"
    if status is not None and 400 <= status <= 599:
        return "permanent"
    if any(token in name for token in _AUTH_ERROR_TOKENS):
        return "auth"
    if any(token in name for token in _RATE_LIMIT_ERROR_TOKENS):
        return "rate_limit"
    return "transient"


def raise_mapped_provider_error(
    exc: Exception,
    *,
    provider: str,
    env_key: str | None,
    message_prefix: str,
    status_code: object | None = None,
    detail: str | None = None,
) -> NoReturn:
    """Raise the shared Crupier error type for an HTTP or provider SDK failure."""
    classification = classify_http_error(exc, status_code=status_code)
    message = detail if detail is not None else str(exc)
    if classification == "auth":
        raise CrupierProviderAuthError(message, provider=provider, env_key=env_key) from exc
    if classification == "rate_limit":
        raise CrupierProviderRateLimitError(message) from exc
    raise CrupierProviderUnavailableError(
        f"{message_prefix}: {message}",
        retryable=classification == "transient",
    ) from exc


def _status_code(value: object | None) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        status = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
    return status if 100 <= status <= 599 else None


def env_value(settings: ProviderSettings, default_env_key: str, *, provider: str) -> str | None:
    env_key = settings.env_key or default_env_key
    value = settings.env_value(env_key)
    if not value:
        return None
    return value


def require_api_key(settings: ProviderSettings, default_env_key: str, *, provider: str) -> str:
    env_key = settings.env_key or default_env_key
    value = settings.env_value(env_key)
    if not value:
        raise CrupierProviderAuthError(
            f"Missing API key for provider {provider!r}.",
            provider=provider,
            env_key=env_key,
            hint=f"Set {env_key} or update [providers.{provider}].env_key in crupier.toml.",
        )
    return value


def provider_timeout_seconds(settings: ProviderSettings, *, default: float | None = None) -> float | None:
    value = settings.options.get("timeout_seconds", settings.options.get("timeout", default))
    return _positive_float(value)


def request_timeout_seconds(request: RequestEnvelope, *, default: float | None = None) -> float | None:
    value = request.constraints.get("timeout_seconds", request.constraints.get("timeout", default))
    return _positive_float(value)


def _positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def build_prompt(request: RequestEnvelope, *, extra: str | None = None) -> str:
    parts = [f"Task:\n{request.task}"]
    if request.messages:
        parts.append("Messages:\n" + _format_input(request.messages))
    if request.input is not None:
        parts.append("Input:\n" + _format_input(request.input))
    file_context = request.metadata.get("extracted_file_context") if request.metadata else None
    if isinstance(file_context, dict) and file_context.get("body"):
        parts.append("File context:\n" + str(file_context["body"]))
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


def _format_input(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except TypeError:
        return repr(value)


def extract_openai_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    if isinstance(response, dict):
        if response.get("output_text"):
            return str(response["output_text"])
        output = response.get("output", [])
    else:
        output = getattr(response, "output", [])
    chunks: list[str] = []
    for item in output or []:
        content = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
        for block in content or []:
            if isinstance(block, dict):
                text = block.get("text") or block.get("output_text")
            else:
                text = getattr(block, "text", None)
            if text:
                chunks.append(str(text))
    return "".join(chunks)


def extract_anthropic_text(message: Any) -> str:
    content = message.get("content", []) if isinstance(message, dict) else getattr(message, "content", [])
    chunks: list[str] = []
    for block in content or []:
        if isinstance(block, dict):
            if block.get("type") == "text" and block.get("text"):
                chunks.append(str(block["text"]))
        else:
            text = getattr(block, "text", None)
            if text:
                chunks.append(str(text))
    return "".join(chunks)


def object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    attrs = {}
    for name in ["input_tokens", "output_tokens", "total_tokens"]:
        if hasattr(value, name):
            attrs[name] = getattr(value, name)
    return attrs
