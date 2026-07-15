"""Configurable adapter for OpenAI-compatible inference servers."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, NoReturn
from urllib.parse import urlparse

from crupier.config import INFERENCE_DEFAULT_HOST, ProviderSettings
from crupier.errors import (
    CrupierProviderAuthError,
    CrupierProviderRateLimitError,
    CrupierProviderUnavailableError,
)
from crupier.models import RequestEnvelope
from crupier.multimodal import native_file_payloads
from crupier.structured import schema_from_request

from .base import AdapterResponse, EmbeddingResponse, ProviderModel
from .common import object_to_dict, provider_timeout_seconds, request_timeout_seconds, require_api_key


class OpenAICompatibleAdapter:
    """Use a configurable OpenAI-compatible Chat Completions endpoint."""

    def __init__(self, settings: ProviderSettings, *, provider: str = "inference") -> None:
        self.settings = settings
        self.provider = provider
        self._client: Any = None

    @staticmethod
    def supports_file_kind(*, model: str, kind: str) -> bool:
        del model
        return kind == "image"

    def generate(self, *, model: str, prompt: str, request: RequestEnvelope) -> AdapterResponse:
        client = self._client or self._build_client()
        self._client = client
        messages, image_count = self._messages(prompt=prompt, request=request)
        payload: dict[str, Any] = {"model": model, "messages": messages}
        max_tokens = request.constraints.get("max_output_tokens", request.constraints.get("max_tokens"))
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        for key in ("temperature", "top_p"):
            if key in request.constraints:
                payload[key] = request.constraints[key]
        timeout = request_timeout_seconds(request)
        if timeout is not None:
            payload["timeout"] = timeout
        schema = schema_from_request(request)
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": str(request.constraints.get("response_schema_name") or "crupier_response"),
                    "strict": bool(request.constraints.get("strict_response_schema", True)),
                    "schema": schema,
                },
            }
        self._apply_compatibility_options(payload, request)
        try:
            response = client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)
        message = _first_message(response)
        return AdapterResponse(
            text=_message_text(message),
            raw=response,
            usage=object_to_dict(getattr(response, "usage", None)),
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "chat.completions.create",
                "multimodal_images": image_count,
                "response_format": "json_schema" if schema is not None else None,
            },
        )

    def list_models(self) -> list[ProviderModel]:
        client = self._client or self._build_client()
        self._client = client
        try:
            response = client.models.list()
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)
        data = getattr(response, "data", response.get("data", []) if isinstance(response, dict) else [])
        models: list[ProviderModel] = []
        for item in data or []:
            model_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
            if not model_id:
                continue
            metadata = item if isinstance(item, dict) else object_to_dict(item)
            models.append(ProviderModel(id=str(model_id), provider=self.provider, metadata=metadata))
        return sorted(models, key=lambda item: item.id)

    def embed(self, *, model: str, input: Any, dimensions: int | None = None) -> EmbeddingResponse:
        if dimensions is not None and dimensions <= 0:
            raise CrupierProviderUnavailableError("Embedding dimensions must be positive.", retryable=False)
        client = self._client or self._build_client()
        self._client = client
        payload: dict[str, Any] = {"model": model, "input": input}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        try:
            response = client.embeddings.create(**payload)
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)
        data = getattr(response, "data", response.get("data", []) if isinstance(response, dict) else [])
        embeddings = []
        for item in data or []:
            vector = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
            if vector is not None:
                embeddings.append([float(value) for value in vector])
        return EmbeddingResponse(
            embeddings=embeddings,
            raw=response,
            usage=object_to_dict(getattr(response, "usage", None)),
            metadata={"provider": self.provider, "model": model, "api": "embeddings.create"},
        )

    def probe_capability(self, *, model: str, probe: str, request: RequestEnvelope) -> AdapterResponse:
        if probe == "structured_output":
            schema = {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}, "probe": {"type": "string"}},
                "required": ["ok", "probe"],
                "additionalProperties": False,
            }
            probe_request = replace(
                request,
                constraints={
                    **request.constraints,
                    "response_schema": schema,
                    "response_schema_name": "crupier_probe",
                },
            )
            response = self.generate(
                model=model,
                prompt='Return {"ok": true, "probe": "crupier"}.',
                request=probe_request,
            )
            try:
                parsed = json.loads(response.text)
            except json.JSONDecodeError:
                parsed = {}
            ok = parsed.get("ok") is True and parsed.get("probe") == "crupier"
            response.metadata.update({"native_probe": True, "ok": ok, "probe_status": "verified" if ok else "failed"})
            return response
        if probe == "tool_call":
            return self._probe_tool_call(model=model, request=request)
        if probe == "streaming":
            return self._probe_streaming(model=model, request=request)
        raise NotImplementedError(f"OpenAI-compatible adapter has no native probe registered for {probe!r}.")

    def _probe_tool_call(self, *, model: str, request: RequestEnvelope) -> AdapterResponse:
        client = self._client or self._build_client()
        self._client = client
        tool_name = "crupier_probe_tool"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Call crupier_probe_tool with ok=true."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": "Report that the capability probe succeeded.",
                        "parameters": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}},
                            "required": ["ok"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
        }
        timeout = request_timeout_seconds(request)
        if timeout is not None:
            payload["timeout"] = timeout
        self._apply_compatibility_options(payload, request)
        try:
            response = client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)
        message = _first_message(response)
        tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else getattr(message, "tool_calls", [])
        names = []
        for call in tool_calls or []:
            function = call.get("function", {}) if isinstance(call, dict) else getattr(call, "function", None)
            name = function.get("name") if isinstance(function, dict) else getattr(function, "name", None)
            if name:
                names.append(str(name))
        ok = tool_name in names
        return AdapterResponse(
            text="",
            raw=response,
            usage=object_to_dict(getattr(response, "usage", None)),
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "chat.completions.create",
                "native_probe": True,
                "ok": ok,
                "probe_status": "verified" if ok else "failed",
                "tool_name": tool_name,
            },
        )

    def _probe_streaming(self, *, model: str, request: RequestEnvelope) -> AdapterResponse:
        client = self._client or self._build_client()
        self._client = client
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: stream-ok"}],
            "stream": True,
        }
        timeout = request_timeout_seconds(request)
        if timeout is not None:
            payload["timeout"] = timeout
        self._apply_compatibility_options(payload, request)
        try:
            stream = client.chat.completions.create(**payload)
            event_count = 0
            text_seen = False
            for event in stream:
                event_count += 1
                message = _first_delta(event)
                if _message_text(message):
                    text_seen = True
                if event_count >= 20 and text_seen:
                    break
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)
        ok = event_count > 0 and text_seen
        return AdapterResponse(
            text="",
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "chat.completions.create",
                "native_probe": True,
                "ok": ok,
                "probe_status": "verified" if ok else "failed",
                "event_count": event_count,
                "text_event_seen": text_seen,
            },
        )

    def _messages(self, *, prompt: str, request: RequestEnvelope) -> tuple[list[dict[str, Any]], int]:
        image_payloads = native_file_payloads(
            request.files,
            allowed_kinds={"image"},
            max_bytes=int(request.constraints.get("max_native_file_bytes", 20_000_000)),
        )
        if not image_payloads:
            return [{"role": "user", "content": prompt}], 0
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": item["data_url"]}}
            for item in image_payloads
        )
        return [{"role": "user", "content": content}], len(image_payloads)

    def _apply_compatibility_options(self, payload: dict[str, Any], request: RequestEnvelope) -> None:
        extra_body: dict[str, Any] = {}
        configured_extra = self.settings.options.get("extra_body")
        if isinstance(configured_extra, dict):
            extra_body.update(configured_extra)
        request_extra = request.constraints.get("extra_body")
        if isinstance(request_extra, dict):
            extra_body.update(request_extra)
        if self.settings.options.get("thinking_control") == "chat_template_kwargs":
            enabled = request.constraints.get("enable_thinking")
            if "disable_thinking" in request.constraints:
                enabled = not bool(request.constraints["disable_thinking"])
            if enabled is not None:
                template_options = dict(extra_body.get("chat_template_kwargs") or {})
                template_options["enable_thinking"] = bool(enabled)
                extra_body["chat_template_kwargs"] = template_options
        if extra_body:
            payload["extra_body"] = extra_body

    def _build_client(self) -> Any:
        host = self.settings.host or INFERENCE_DEFAULT_HOST
        if self.settings.env_key:
            api_key = require_api_key(self.settings, self.settings.env_key, provider=self.provider)
        elif str(self.settings.options.get("auth", "")).lower() == "none" and _is_loopback_host(host):
            api_key = "crupier-local"
        else:
            api_key = require_api_key(self.settings, "INFERENCE_API_KEY", provider=self.provider)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise CrupierProviderUnavailableError(
                "OpenAI-compatible inference adapter requires: pip install 'crupier[inference-server]'.",
                retryable=False,
            ) from exc
        kwargs: dict[str, Any] = {"api_key": api_key, "base_url": host}
        timeout = provider_timeout_seconds(self.settings)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return OpenAI(**kwargs)

    def _raise_mapped_error(self, exc: Exception) -> NoReturn:
        name = exc.__class__.__name__.lower()
        if "auth" in name or "permission" in name:
            raise CrupierProviderAuthError(str(exc), provider=self.provider, env_key=self.settings.env_key) from exc
        if "ratelimit" in name or "rate_limit" in name:
            raise CrupierProviderRateLimitError(str(exc)) from exc
        raise CrupierProviderUnavailableError(
            f"OpenAI-compatible inference request failed for {self.provider}: {exc}",
        ) from exc


def _is_loopback_host(host: str) -> bool:
    return (urlparse(host).hostname or "").lower() in {"localhost", "127.0.0.1", "::1"}


def _first_message(response: Any) -> Any:
    choices = response.get("choices", []) if isinstance(response, dict) else getattr(response, "choices", [])
    if not choices:
        return None
    choice = choices[0]
    return choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)


def _first_delta(response: Any) -> Any:
    choices = response.get("choices", []) if isinstance(response, dict) else getattr(response, "choices", [])
    if not choices:
        return None
    choice = choices[0]
    return choice.get("delta") if isinstance(choice, dict) else getattr(choice, "delta", None)


def _message_text(message: Any) -> str:
    if message is None:
        return ""
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return "".join(parts)
    return "" if content is None else str(content)
