"""Shared adapter helpers."""

from __future__ import annotations

import json
import os
from typing import Any

from crupier.config import ProviderSettings
from crupier.errors import CrupierProviderAuthError
from crupier.models import RequestEnvelope


def env_value(settings: ProviderSettings, default_env_key: str, *, provider: str) -> str | None:
    env_key = settings.env_key or default_env_key
    value = os.environ.get(env_key)
    if not value:
        return None
    return value


def require_api_key(settings: ProviderSettings, default_env_key: str, *, provider: str) -> str:
    env_key = settings.env_key or default_env_key
    value = os.environ.get(env_key)
    if not value:
        raise CrupierProviderAuthError(
            f"Missing API key for provider {provider!r}.",
            provider=provider,
            env_key=env_key,
            hint=f"Set {env_key} or update [providers.{provider}].env_key in crupier.toml.",
        )
    return value


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
