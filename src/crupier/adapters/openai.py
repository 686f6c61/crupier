"""OpenAI Responses API adapter."""

from __future__ import annotations

import json
import re
from typing import Any, NoReturn

from crupier.config import ProviderSettings
from crupier.errors import (
    CrupierProviderAuthError,
    CrupierProviderRateLimitError,
    CrupierProviderUnavailableError,
)
from crupier.models import RequestEnvelope
from crupier.multimodal import native_file_payloads
from crupier.structured import schema_from_request

from .base import AdapterResponse, EmbeddingResponse, ProviderModel
from .common import (
    build_prompt,
    extract_openai_text,
    object_to_dict,
    provider_timeout_seconds,
    request_timeout_seconds,
    require_api_key,
)


class OpenAIAdapter:
    provider = "openai"

    def __init__(self, settings: ProviderSettings):
        self.settings = settings
        self._client: Any = None

    @staticmethod
    def supports_file_kind(*, model: str, kind: str) -> bool:
        del model
        return kind in {"image", "pdf"}

    def generate(self, *, model: str, prompt: str, request: RequestEnvelope) -> AdapterResponse:
        client = self._client or self._build_client()
        self._client = client
        input_payload: Any = prompt or build_prompt(request)
        native_payloads = []
        if request.files:
            native_payloads = native_file_payloads(
                request.files,
                allowed_kinds={"image", "pdf"},
                max_bytes=int(request.constraints.get("max_native_file_bytes", 20_000_000)),
            )
            content: list[dict[str, Any]] = [{"type": "input_text", "text": input_payload}]
            for item in native_payloads:
                if item["kind"] == "image":
                    content.append({"type": "input_image", "image_url": item["data_url"]})
                else:
                    content.append(
                        {
                            "type": "input_file",
                            "filename": item["name"],
                            "file_data": item["data_url"],
                        }
                    )
            input_payload = [{"role": "user", "content": content}]
        payload: dict[str, Any] = {
            "model": model,
            "input": input_payload,
        }
        if "temperature" in request.constraints:
            payload["temperature"] = request.constraints["temperature"]
        if "max_output_tokens" in request.constraints:
            payload["max_output_tokens"] = request.constraints["max_output_tokens"]
        timeout = request_timeout_seconds(request)
        if timeout is not None:
            payload["timeout"] = timeout
        response_schema = schema_from_request(request)
        if response_schema:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": _response_schema_name(request),
                    "schema": response_schema,
                    "strict": bool(request.constraints.get("strict_response_schema", True)),
                }
            }

        response, removed_params = self._responses_create_with_param_repair(client, payload)

        text = extract_openai_text(response)
        usage = object_to_dict(getattr(response, "usage", None))
        metadata = {
            "provider": self.provider,
            "model": model,
            "api": "responses.create",
            "multimodal_images": sum(item["kind"] == "image" for item in native_payloads),
            "native_files": sum(item["kind"] != "image" for item in native_payloads),
            "response_format": "json_schema" if response_schema else None,
            "removed_params": removed_params,
        }
        return AdapterResponse(
            text=text,
            raw=response,
            usage=usage,
            metadata={key: value for key, value in metadata.items() if value is not None},
        )

    def list_models(self) -> list[ProviderModel]:
        client = self._client or self._build_client()
        self._client = client
        try:
            response = client.models.list()
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)

        data = getattr(response, "data", response)
        models: list[ProviderModel] = []
        for item in data:
            model_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
            if not model_id:
                continue
            metadata = item if isinstance(item, dict) else object_to_dict(item)
            models.append(ProviderModel(id=str(model_id), provider=self.provider, metadata=metadata))
        return sorted(models, key=lambda model: model.id)

    def embed(self, *, model: str, input: Any, dimensions: int | None = None) -> EmbeddingResponse:
        client = self._client or self._build_client()
        self._client = client
        payload: dict[str, Any] = {"model": model, "input": input}
        if dimensions is not None:
            if dimensions <= 0:
                raise CrupierProviderUnavailableError("Embedding dimensions must be positive.", retryable=False)
            payload["dimensions"] = dimensions
        try:
            response = client.embeddings.create(**payload)
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)

        data = getattr(response, "data", response.get("data", []) if isinstance(response, dict) else [])
        embeddings: list[list[float]] = []
        for item in data or []:
            embedding = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
            if embedding is not None:
                embeddings.append([float(value) for value in embedding])
        return EmbeddingResponse(
            embeddings=embeddings,
            raw=response,
            usage=object_to_dict(getattr(response, "usage", None)),
            metadata={"provider": self.provider, "model": model, "api": "embeddings.create"},
        )

    def probe_capability(self, *, model: str, probe: str, request: RequestEnvelope) -> AdapterResponse:
        if probe == "structured_output":
            return self._probe_structured_output(model=model, request=request)
        if probe == "tool_call":
            return self._probe_tool_call(model=model, request=request)
        if probe == "streaming":
            return self._probe_streaming(model=model, request=request)
        raise NotImplementedError(f"OpenAI adapter has no native probe registered for {probe!r}.")

    def _probe_structured_output(self, *, model: str, request: RequestEnvelope) -> AdapterResponse:
        client = self._client or self._build_client()
        self._client = client
        schema = _probe_schema()
        payload = {
            "model": model,
            "input": 'Return {"ok": true, "probe": "crupier"} using the provided schema.',
            "max_output_tokens": int(request.constraints.get("max_output_tokens", 128)),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "crupier_probe",
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        timeout = request_timeout_seconds(request)
        if timeout is not None:
            payload["timeout"] = timeout
        try:
            response = client.responses.create(**payload)
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)
        text = extract_openai_text(response)
        ok = _json_probe_ok(text)
        return AdapterResponse(
            text="",
            raw=response,
            usage=object_to_dict(getattr(response, "usage", None)),
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "responses.create",
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
        payload = {
            "model": model,
            "input": "Call the crupier_probe_tool with ok=true.",
            "max_output_tokens": int(request.constraints.get("max_output_tokens", 128)),
            "tools": [
                {
                    "type": "function",
                    "name": tool_name,
                    "description": "Report that the capability probe succeeded.",
                    "parameters": _probe_schema(),
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "function", "name": tool_name},
        }
        timeout = request_timeout_seconds(request)
        if timeout is not None:
            payload["timeout"] = timeout
        try:
            response = client.responses.create(**payload)
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)
        ok = _openai_has_tool_call(response, tool_name)
        return AdapterResponse(
            text="",
            raw=response,
            usage=object_to_dict(getattr(response, "usage", None)),
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "responses.create",
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
            payload: dict[str, Any] = {
                "model": model,
                "input": 'Reply with exactly: "stream-ok"',
                "max_output_tokens": int(request.constraints.get("max_output_tokens", 256)),
                "stream": True,
            }
            timeout = request_timeout_seconds(request)
            if timeout is not None:
                payload["timeout"] = timeout
            stream = client.responses.create(**payload)
            event_count = 0
            text_seen = False
            for event in stream:
                event_count += 1
                if _event_has_text(event):
                    text_seen = True
                if event_count >= 20 and text_seen:
                    break
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)
        ok = event_count > 0 and text_seen
        return AdapterResponse(
            text="",
            raw=None,
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "responses.create",
                "native_probe": True,
                "ok": ok,
                "probe_status": "verified" if ok else "failed",
                "event_count": event_count,
                "text_event_seen": text_seen,
            },
        )

    def _build_client(self) -> Any:
        api_key = require_api_key(self.settings, "OPENAI_API_KEY", provider=self.provider)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise CrupierProviderUnavailableError(
                "OpenAI adapter requires the optional dependency: pip install 'crupier[openai]'.",
                retryable=False,
            ) from exc
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.settings.host:
            kwargs["base_url"] = self.settings.host
        timeout = provider_timeout_seconds(self.settings)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return OpenAI(**kwargs)

    def _responses_create_with_param_repair(self, client: Any, payload: dict[str, Any]) -> tuple[Any, list[str]]:
        try:
            return client.responses.create(**payload), []
        except Exception as exc:  # noqa: BLE001 - provider SDK exceptions vary by version
            unsupported = _unsupported_parameter(exc)
            if unsupported and unsupported in payload:
                repaired = dict(payload)
                repaired.pop(unsupported, None)
                try:
                    return client.responses.create(**repaired), [unsupported]
                except Exception as repaired_exc:  # noqa: BLE001
                    self._raise_mapped_error(repaired_exc)
            self._raise_mapped_error(exc)

    def _raise_mapped_error(self, exc: Exception) -> NoReturn:
        name = exc.__class__.__name__.lower()
        if "auth" in name or "permission" in name:
            raise CrupierProviderAuthError(str(exc), provider=self.provider, env_key=self.settings.env_key) from exc
        if "ratelimit" in name or "rate_limit" in name:
            raise CrupierProviderRateLimitError(str(exc)) from exc
        raise CrupierProviderUnavailableError(f"OpenAI request failed: {exc}") from exc


def _unsupported_parameter(exc: Exception) -> str | None:
    text = str(exc)
    lowered = text.lower()
    if "unsupported" not in lowered and "not supported" not in lowered:
        return None
    for pattern in [
        r"Unsupported parameter:\s*'([^']+)'",
        r'"param":\s*"([^"]+)"',
        r"'param':\s*'([^']+)'",
    ]:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


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


def _response_schema_name(request: RequestEnvelope) -> str:
    name = request.constraints.get("response_schema_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return "crupier_response"


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


def _openai_has_tool_call(response: Any, tool_name: str) -> bool:
    data = response if isinstance(response, dict) else object_to_dict(response)
    output = data.get("output", [])
    for item in output or []:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        name = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
        if item_type in {"function_call", "tool_call"} and name == tool_name:
            return True
    return False


def _event_has_text(event: Any) -> bool:
    if isinstance(event, dict):
        return bool(event.get("delta") or event.get("text") or event.get("output_text"))
    for attr in ["delta", "text", "output_text"]:
        if getattr(event, attr, None):
            return True
    return False
