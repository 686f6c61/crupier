"""Google Gemini adapter using the Google Gen AI SDK."""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from crupier.config import ProviderSettings
from crupier.errors import (
    CrupierProviderAuthError,
    CrupierProviderRateLimitError,
    CrupierProviderUnavailableError,
)
from crupier.models import RequestEnvelope
from crupier.multimodal import native_image_payloads

from .base import AdapterResponse, EmbeddingResponse, ProviderModel
from .common import build_prompt, object_to_dict, provider_timeout_seconds


GOOGLE_DEFAULT_ENV_KEYS = ("GOOGLE_API_KEY", "GEMINI_API_KEY")


class GoogleAdapter:
    provider = "google"

    def __init__(self, settings: ProviderSettings):
        self.settings = settings
        self._client: Any = None

    def generate(self, *, model: str, prompt: str, request: RequestEnvelope) -> AdapterResponse:
        client = self._client or self._build_client()
        self._client = client
        contents, image_count = self._contents(prompt or build_prompt(request), request)
        payload: dict[str, Any] = {"model": model, "contents": contents}
        config = self._generation_config(request)
        if config:
            payload["config"] = config
        try:
            response = client.models.generate_content(**payload)
        except Exception as exc:  # noqa: BLE001 - provider SDK exceptions vary by version
            self._raise_mapped_error(exc)

        return AdapterResponse(
            text=extract_google_text(response),
            raw=response,
            usage=_google_usage(response),
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "models.generate_content",
                "multimodal_images": image_count,
            },
        )

    def list_models(self) -> list[ProviderModel]:
        client = self._client or self._build_client()
        self._client = client
        try:
            response = client.models.list()
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)

        models: list[ProviderModel] = []
        for item in response:
            model_id = _model_id(item)
            if not model_id:
                continue
            metadata = object_to_dict(item)
            models.append(
                ProviderModel(
                    id=model_id,
                    provider=self.provider,
                    name=_display_name(item),
                    metadata=metadata,
                )
            )
        return sorted(models, key=lambda model: model.id)

    def embed(self, *, model: str, input: Any) -> EmbeddingResponse:
        client = self._client or self._build_client()
        self._client = client
        try:
            response = client.models.embed_content(model=model, contents=input)
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)

        embeddings = _google_embeddings(response)
        metadata = {
            "provider": self.provider,
            "model": model,
            "api": "models.embed_content",
            "embedding_dimensions": len(embeddings[0]) if embeddings else None,
        }
        return EmbeddingResponse(
            embeddings=embeddings,
            raw=response,
            usage=_google_usage(response),
            metadata={key: value for key, value in metadata.items() if value is not None},
        )

    def probe_capability(self, *, model: str, probe: str, request: RequestEnvelope) -> AdapterResponse:
        if probe == "structured_output":
            return self._probe_structured_output(model=model, request=request)
        if probe == "tool_call":
            return self._probe_tool_call(model=model, request=request)
        if probe == "streaming":
            return self._probe_streaming(model=model, request=request)
        raise NotImplementedError(f"Google adapter has no native probe registered for {probe!r}.")

    def _probe_structured_output(self, *, model: str, request: RequestEnvelope) -> AdapterResponse:
        client = self._client or self._build_client()
        self._client = client
        payload = {
            "model": model,
            "contents": 'Return {"ok": true, "probe": "crupier"} using the provided schema.',
            "config": {
                "response_mime_type": "application/json",
                "response_json_schema": _probe_schema(),
                "max_output_tokens": int(request.constraints.get("max_output_tokens", 128)),
                "temperature": request.constraints.get("temperature", 0),
            },
        }
        try:
            response = client.models.generate_content(**payload)
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)
        ok = _json_probe_ok(extract_google_text(response))
        return AdapterResponse(
            text="",
            raw=response,
            usage=_google_usage(response),
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "models.generate_content",
                "native_probe": True,
                "ok": ok,
                "probe_status": "verified" if ok else "failed",
                "response_format": "json_schema",
            },
        )

    def _probe_tool_call(self, *, model: str, request: RequestEnvelope) -> AdapterResponse:
        client = self._client or self._build_client()
        self._client = client
        tool_name = "crupier_probe_tool"

        def crupier_probe_tool(ok: bool, probe: str) -> dict[str, Any]:
            """Report that the capability probe succeeded."""

            return {"ok": ok, "probe": probe}

        config = _tool_probe_config(
            tools=[crupier_probe_tool],
            max_output_tokens=int(request.constraints.get("max_output_tokens", 128)),
            temperature=request.constraints.get("temperature", 0),
        )
        try:
            response = client.models.generate_content(
                model=model,
                contents="Call crupier_probe_tool with ok=true and probe='crupier'.",
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)
        ok = _google_has_tool_call(response, tool_name)
        return AdapterResponse(
            text="",
            raw=response,
            usage=_google_usage(response),
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "models.generate_content",
                "native_probe": True,
                "ok": ok,
                "probe_status": "verified" if ok else "failed",
                "tool_name": tool_name,
            },
        )

    def _probe_streaming(self, *, model: str, request: RequestEnvelope) -> AdapterResponse:
        client = self._client or self._build_client()
        self._client = client
        try:
            stream = client.models.generate_content_stream(
                model=model,
                contents='Reply with exactly: "stream-ok"',
                config={
                    "max_output_tokens": int(request.constraints.get("max_output_tokens", 16)),
                    "temperature": request.constraints.get("temperature", 0),
                },
            )
            event_count = 0
            text_seen = False
            for chunk in stream:
                event_count += 1
                if extract_google_text(chunk):
                    text_seen = True
                if event_count >= 20 and text_seen:
                    break
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)
        ok = event_count > 0
        return AdapterResponse(
            text="",
            raw=None,
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "models.generate_content_stream",
                "native_probe": True,
                "ok": ok,
                "probe_status": "verified" if ok else "failed",
                "event_count": event_count,
                "text_event_seen": text_seen,
            },
        )

    def _contents(self, text: str, request: RequestEnvelope) -> tuple[Any, int]:
        if not request.files:
            return text, 0
        image_payloads = native_image_payloads(request.files)
        parts: list[Any] = [_google_text_part(text)]
        parts.extend(_google_image_part(image) for image in image_payloads)
        return parts, len(image_payloads)

    def _generation_config(self, request: RequestEnvelope) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if "temperature" in request.constraints:
            config["temperature"] = request.constraints["temperature"]
        if "max_output_tokens" in request.constraints:
            config["max_output_tokens"] = request.constraints["max_output_tokens"]
        if request.response_schema:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = request.response_schema
        return config

    def _build_client(self) -> Any:
        api_key = google_api_key(self.settings)
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise CrupierProviderUnavailableError(
                "Google adapter requires the optional dependency: pip install 'crupier[google]'.",
                retryable=False,
            ) from exc
        kwargs: dict[str, Any] = {"api_key": api_key}
        timeout = provider_timeout_seconds(self.settings)
        if timeout is not None:
            kwargs["http_options"] = types.HttpOptions(timeout=int(timeout * 1000))
        return genai.Client(**kwargs)

    def _raise_mapped_error(self, exc: Exception) -> None:
        name = exc.__class__.__name__.lower()
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        text = str(exc).lower()
        if status in {401, 403} or any(token in name or token in text for token in ["auth", "permission", "forbidden"]):
            raise CrupierProviderAuthError(str(exc), provider=self.provider, env_key=_google_env_label(self.settings)) from exc
        if status == 429 or any(token in name or token in text for token in ["ratelimit", "rate_limit", "resourceexhausted"]):
            raise CrupierProviderRateLimitError(str(exc)) from exc
        raise CrupierProviderUnavailableError(f"Google request failed: {exc}") from exc


def google_api_key(settings: ProviderSettings) -> str:
    env_key = settings.env_key
    keys = [env_key] if env_key else list(GOOGLE_DEFAULT_ENV_KEYS)
    if env_key in GOOGLE_DEFAULT_ENV_KEYS:
        keys = list(dict.fromkeys([env_key, *GOOGLE_DEFAULT_ENV_KEYS]))
    for key in keys:
        if key and os.environ.get(key):
            return str(os.environ[key])
    raise CrupierProviderAuthError(
        "Missing API key for provider 'google'.",
        provider="google",
        env_key=_google_env_label(settings),
        hint="Set GOOGLE_API_KEY or GEMINI_API_KEY, or update [providers.google].env_key in crupier.toml.",
    )


def google_env_present(settings: ProviderSettings | None) -> bool:
    if settings is None:
        return False
    env_key = settings.env_key
    keys = [env_key] if env_key else list(GOOGLE_DEFAULT_ENV_KEYS)
    if env_key in GOOGLE_DEFAULT_ENV_KEYS:
        keys = list(dict.fromkeys([env_key, *GOOGLE_DEFAULT_ENV_KEYS]))
    return any(bool(key and os.environ.get(key)) for key in keys)


def google_env_label(settings: ProviderSettings | None) -> str:
    if settings is None:
        return "GOOGLE_API_KEY/GEMINI_API_KEY"
    return _google_env_label(settings)


def extract_google_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    if isinstance(response, dict) and response.get("text"):
        return str(response["text"])
    chunks: list[str] = []
    for part in _google_parts(response):
        part_text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        if part_text:
            chunks.append(str(part_text))
    return "".join(chunks)


def _google_env_label(settings: ProviderSettings) -> str:
    if settings.env_key and settings.env_key not in GOOGLE_DEFAULT_ENV_KEYS:
        return settings.env_key
    return "GOOGLE_API_KEY/GEMINI_API_KEY"


def _google_text_part(text: str) -> Any:
    try:
        from google.genai import types

        return types.Part.from_text(text=text)
    except ImportError:
        return {"text": text}


def _google_image_part(image: dict[str, str]) -> Any:
    try:
        from google.genai import types

        return types.Part.from_bytes(data=base64.b64decode(image["base64"]), mime_type=image["mime_type"])
    except ImportError:
        return {"inline_data": {"mime_type": image["mime_type"], "data": image["base64"]}}


def _tool_probe_config(*, tools: list[Any], max_output_tokens: int, temperature: Any) -> Any:
    try:
        from google.genai import types

        return types.GenerateContentConfig(
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    except ImportError:
        return {
            "tools": tools,
            "automatic_function_calling": {"disable": True},
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }


def _model_id(item: Any) -> str | None:
    value = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
    if not value:
        value = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
    if not value:
        return None
    text = str(value)
    return text.split("/", 1)[1] if text.startswith("models/") else text


def _display_name(item: Any) -> str | None:
    if isinstance(item, dict):
        return item.get("display_name") or item.get("displayName")
    return getattr(item, "display_name", None) or getattr(item, "displayName", None)


def _google_usage(response: Any) -> dict[str, Any]:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None and isinstance(response, dict):
        metadata = response.get("usage_metadata") or response.get("usageMetadata")
    return object_to_dict(metadata)


def _google_embeddings(response: Any) -> list[list[float]]:
    raw = response.get("embeddings") if isinstance(response, dict) else getattr(response, "embeddings", None)
    if raw is None:
        raw = response.get("embedding") if isinstance(response, dict) else getattr(response, "embedding", None)
    if raw is None:
        return []
    if _looks_like_vector(raw):
        raw = [raw]
    embeddings: list[list[float]] = []
    for item in raw or []:
        values = item.get("values") if isinstance(item, dict) else getattr(item, "values", None)
        if values is None and isinstance(item, list):
            values = item
        if values is not None:
            embeddings.append([float(value) for value in values])
    return embeddings


def _google_has_tool_call(response: Any, tool_name: str) -> bool:
    for part in _google_parts(response):
        function_call = part.get("function_call") if isinstance(part, dict) else getattr(part, "function_call", None)
        if function_call is None and isinstance(part, dict):
            function_call = part.get("functionCall")
        if not function_call:
            continue
        name = function_call.get("name") if isinstance(function_call, dict) else getattr(function_call, "name", None)
        if name == tool_name:
            return True
    return False


def _google_parts(response: Any) -> list[Any]:
    direct_parts = response.get("parts") if isinstance(response, dict) else getattr(response, "parts", None)
    if direct_parts:
        return list(direct_parts)
    data = response if isinstance(response, dict) else object_to_dict(response)
    parts: list[Any] = []
    for candidate in data.get("candidates", []) or []:
        content = candidate.get("content", {}) if isinstance(candidate, dict) else getattr(candidate, "content", {})
        candidate_parts = content.get("parts", []) if isinstance(content, dict) else getattr(content, "parts", [])
        parts.extend(candidate_parts or [])
    return parts


def _probe_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "probe": {"type": "string"},
        },
        "required": ["ok", "probe"],
        "additionalProperties": False,
    }


def _json_probe_ok(text: str) -> bool:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return False
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return False
    return data.get("ok") is True and data.get("probe") == "crupier"


def _looks_like_vector(value: Any) -> bool:
    return isinstance(value, list) and (not value or isinstance(value[0], int | float))
