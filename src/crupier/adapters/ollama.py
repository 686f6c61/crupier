"""Ollama Cloud and explicit local adapter using the native REST API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, NoReturn

from crupier.config import (
    OLLAMA_CLOUD_HOST,
    ProviderSettings,
    validate_provider_endpoint,
)
from crupier.errors import (
    CrupierModelUnsupportedError,
    CrupierProviderAuthError,
    CrupierProviderUnavailableError,
)
from crupier.models import RequestEnvelope
from crupier.multimodal import native_image_payloads
from crupier.structured import schema_from_request

from .base import AdapterResponse, EmbeddingResponse, ProviderModel
from .common import (
    build_prompt,
    env_value,
    provider_timeout_seconds,
    raise_mapped_provider_error,
    request_timeout_seconds,
)

OLLAMA_AUTH_HINT = (
    "Set OLLAMA_API_KEY for https://ollama.com/api, or configure host explicitly for a local Ollama daemon."
)


class OllamaAdapter:
    provider = "ollama"

    def __init__(self, settings: ProviderSettings):
        self.settings = settings

    @staticmethod
    def supports_file_kind(*, model: str, kind: str) -> bool:
        del model
        return kind == "image"

    def generate(self, *, model: str, prompt: str, request: RequestEnvelope) -> AdapterResponse:
        validate_provider_endpoint(self.provider, self.settings)
        url = self._chat_url()
        message: dict[str, Any] = {"role": "user", "content": prompt or build_prompt(request)}
        image_payloads = []
        if request.files:
            image_payloads = native_image_payloads(request.files)
            message["images"] = [image["base64"] for image in image_payloads]
        payload = {
            "model": model,
            "messages": [message],
            "stream": False,
        }
        if "temperature" in request.constraints:
            payload["options"] = {"temperature": request.constraints["temperature"]}
        response_schema = schema_from_request(request)
        if response_schema:
            payload["format"] = response_schema
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = env_value(self.settings, "OLLAMA_API_KEY", provider=self.provider)
        if self._requires_cloud_auth() and not api_key:
            raise CrupierProviderAuthError(
                "Ollama Cloud direct API requires OLLAMA_API_KEY or [providers.ollama].env_key.",
                provider=self.provider,
                env_key=self.settings.env_key or "OLLAMA_API_KEY",
                hint=OLLAMA_AUTH_HINT,
            )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        timeout = request_timeout_seconds(request, default=provider_timeout_seconds(self.settings, default=120))
        body = self._request_json(
            req,
            timeout=float(timeout or 120),
            operation="request",
            valid_shape=lambda value: isinstance(value.get("message"), dict),
        )

        message = body.get("message", {}) if isinstance(body, dict) else {}
        text = message.get("content", "")
        usage = {
            "prompt_eval_count": body.get("prompt_eval_count"),
            "eval_count": body.get("eval_count"),
        }
        metadata = {
            "provider": self.provider,
            "model": model,
            "api": "api/chat",
            "host": self._base_url(),
            "multimodal_images": len(image_payloads),
            "response_format": "json_schema" if response_schema else None,
        }
        return AdapterResponse(
            text=text,
            raw=body,
            usage={key: value for key, value in usage.items() if value is not None},
            metadata={key: value for key, value in metadata.items() if value is not None},
        )

    def list_models(self) -> list[ProviderModel]:
        validate_provider_endpoint(self.provider, self.settings)
        url = self._tags_url()
        headers = {}
        api_key = env_value(self.settings, "OLLAMA_API_KEY", provider=self.provider)
        if self._requires_cloud_auth() and not api_key:
            raise CrupierProviderAuthError(
                "Ollama Cloud direct API requires OLLAMA_API_KEY or [providers.ollama].env_key.",
                provider=self.provider,
                env_key=self.settings.env_key or "OLLAMA_API_KEY",
                hint=OLLAMA_AUTH_HINT,
            )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(url, headers=headers, method="GET")
        timeout = provider_timeout_seconds(self.settings, default=30)
        body = self._request_json(
            req,
            timeout=float(timeout or 30),
            operation="model listing",
            valid_shape=lambda value: isinstance(value.get("models"), list),
        )

        models: list[ProviderModel] = []
        for item in body.get("models", []):
            model_id = item.get("model") or item.get("name")
            if not model_id:
                continue
            models.append(
                ProviderModel(
                    id=str(model_id),
                    provider=self.provider,
                    name=item.get("name"),
                    metadata=item,
                )
            )
        return sorted(models, key=lambda model: model.id)

    def embed(self, *, model: str, input: Any, dimensions: int | None = None) -> EmbeddingResponse:
        if dimensions is not None:
            raise CrupierModelUnsupportedError(
                "Ollama embeddings do not expose provider-side output dimensions; choose a model with the required size."
            )
        payload = {"model": model, "input": input}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self._embed_url(), data=data, headers=self._headers(), method="POST")
        timeout = provider_timeout_seconds(self.settings, default=120)
        body = self._request_json(
            req,
            timeout=float(timeout or 120),
            operation="embedding request",
            valid_shape=lambda value: isinstance(
                value.get("embeddings", value.get("embedding")), list
            ),
        )

        embeddings = _ollama_embeddings(body)
        usage = {
            "prompt_eval_count": body.get("prompt_eval_count"),
            "eval_count": body.get("eval_count"),
            "total_duration": body.get("total_duration"),
            "load_duration": body.get("load_duration"),
        }
        metadata = {
            "provider": self.provider,
            "model": model,
            "api": "api/embed",
            "host": self._base_url(),
            "embedding_dimensions": len(embeddings[0]) if embeddings else None,
        }
        return EmbeddingResponse(
            embeddings=embeddings,
            raw=body,
            usage={key: value for key, value in usage.items() if value is not None},
            metadata={key: value for key, value in metadata.items() if value is not None},
        )

    def probe_capability(self, *, model: str, probe: str, request: RequestEnvelope) -> AdapterResponse:
        if probe == "structured_output":
            return self._probe_structured_output(model=model, request=request)
        if probe == "tool_call":
            return self._probe_tool_call(model=model, request=request)
        if probe == "streaming":
            return self._probe_streaming(model=model, request=request)
        raise NotImplementedError(f"Ollama adapter has no native probe registered for {probe!r}.")

    def _probe_structured_output(self, *, model: str, request: RequestEnvelope) -> AdapterResponse:
        body = self._chat_json(
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": 'Return exactly {"ok": true, "probe": "crupier"}.'}
                ],
                "format": _probe_schema(),
                "stream": False,
                "options": {"temperature": request.constraints.get("temperature", 0)},
            },
            timeout=float(
                request_timeout_seconds(request, default=provider_timeout_seconds(self.settings, default=60)) or 60
            ),
        )
        message = body.get("message", {}) if isinstance(body, dict) else {}
        text = message.get("content", "")
        ok = _json_probe_ok(text)
        return AdapterResponse(
            text="",
            raw=body,
            usage=_ollama_usage(body),
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "api/chat",
                "native_probe": True,
                "ok": ok,
                "probe_status": "verified" if ok else "failed",
                "response_format": "json_schema",
                "host": self._base_url(),
            },
        )

    def _probe_tool_call(self, *, model: str, request: RequestEnvelope) -> AdapterResponse:
        tool_name = "crupier_probe_tool"
        body = self._chat_json(
            {
                "model": model,
                "messages": [{"role": "user", "content": "Call crupier_probe_tool with ok=true."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": "Report that the capability probe succeeded.",
                            "parameters": _probe_schema(),
                        },
                    }
                ],
                "stream": False,
                "options": {"temperature": request.constraints.get("temperature", 0)},
            },
            timeout=float(
                request_timeout_seconds(request, default=provider_timeout_seconds(self.settings, default=60)) or 60
            ),
        )
        ok = _ollama_has_tool_call(body, tool_name)
        return AdapterResponse(
            text="",
            raw=body,
            usage=_ollama_usage(body),
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "api/chat",
                "native_probe": True,
                "ok": ok,
                "probe_status": "verified" if ok else "failed",
                "tool_name": tool_name,
                "host": self._base_url(),
            },
        )

    def _probe_streaming(self, *, model: str, request: RequestEnvelope) -> AdapterResponse:
        event_count = 0
        text_seen = False
        for body in self._chat_stream(
            {
                "model": model,
                "messages": [{"role": "user", "content": 'Reply with exactly: "stream-ok"'}],
                "stream": True,
                "options": {"temperature": request.constraints.get("temperature", 0)},
            },
            timeout=float(
                request_timeout_seconds(request, default=provider_timeout_seconds(self.settings, default=60)) or 60
            ),
        ):
            event_count += 1
            message = body.get("message", {}) if isinstance(body, dict) else {}
            if message.get("content"):
                text_seen = True
            if event_count >= 20 and text_seen:
                break
        ok = event_count > 0 and text_seen
        return AdapterResponse(
            text="",
            raw=None,
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "api/chat",
                "native_probe": True,
                "ok": ok,
                "probe_status": "verified" if ok else "failed",
                "event_count": event_count,
                "text_event_seen": text_seen,
                "host": self._base_url(),
            },
        )

    def _chat_json(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self._chat_url(), data=data, headers=self._headers(), method="POST")
        return self._request_json(
            req,
            timeout=timeout,
            operation="request",
            valid_shape=lambda value: isinstance(value.get("message"), dict),
        )

    def _chat_stream(self, payload: dict[str, Any], *, timeout: float):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self._chat_url(), data=data, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    body = json.loads(line)
                    if not isinstance(body, dict):
                        raise CrupierProviderUnavailableError(
                            "Ollama request failed: invalid response shape.",
                            retryable=True,
                        )
                    yield body
        except urllib.error.HTTPError as exc:
            self._raise_http_error(exc)
        except urllib.error.URLError as exc:
            self._raise_transport_error("request", exc.reason, exc)
        except TimeoutError as exc:
            self._raise_transport_error("request", "timed out", exc)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._raise_transport_error("request", "invalid JSON response", exc)

    def _request_json(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
        operation: str,
        valid_shape: Callable[[dict[str, Any]], bool],
    ) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            self._raise_http_error(exc)
        except urllib.error.URLError as exc:
            self._raise_transport_error(operation, exc.reason, exc)
        except TimeoutError as exc:
            self._raise_transport_error(operation, "timed out", exc)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._raise_transport_error(operation, "invalid JSON response", exc)

        if not isinstance(body, dict) or not valid_shape(body):
            raise CrupierProviderUnavailableError(
                f"Ollama {operation} failed: invalid response shape.",
                retryable=True,
            )
        return body

    @staticmethod
    def _raise_transport_error(operation: str, reason: object, exc: BaseException) -> NoReturn:
        raise CrupierProviderUnavailableError(
            f"Ollama {operation} failed: {reason}",
            retryable=True,
        ) from exc

    def _headers(self) -> dict[str, str]:
        validate_provider_endpoint(self.provider, self.settings)
        headers = {"Content-Type": "application/json"}
        api_key = env_value(self.settings, "OLLAMA_API_KEY", provider=self.provider)
        if self._requires_cloud_auth() and not api_key:
            raise CrupierProviderAuthError(
                "Ollama Cloud direct API requires OLLAMA_API_KEY or [providers.ollama].env_key.",
                provider=self.provider,
                env_key=self.settings.env_key or "OLLAMA_API_KEY",
                hint=OLLAMA_AUTH_HINT,
            )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _base_url(self) -> str:
        return (self.settings.host or OLLAMA_CLOUD_HOST).rstrip("/")

    def _chat_url(self) -> str:
        base = self._base_url()
        if base.endswith("/api"):
            return f"{base}/chat"
        return f"{base}/api/chat"

    def _tags_url(self) -> str:
        base = self._base_url()
        if base.endswith("/api"):
            return f"{base}/tags"
        return f"{base}/api/tags"

    def _embed_url(self) -> str:
        base = self._base_url()
        if base.endswith("/api"):
            return f"{base}/embed"
        return f"{base}/api/embed"

    def _requires_cloud_auth(self) -> bool:
        base = self._base_url()
        return (
            "ollama.com" in base
            and not base.startswith("http://localhost")
            and not base.startswith("http://127.0.0.1")
        )

    def _raise_http_error(self, exc: urllib.error.HTTPError) -> NoReturn:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        finally:
            exc.close()
        message = body or str(exc)
        raise_mapped_provider_error(
            exc,
            provider=self.provider,
            env_key=self.settings.env_key,
            message_prefix=f"Ollama HTTP {exc.code}",
            status_code=exc.code,
            detail=message,
        )


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


def _ollama_has_tool_call(body: dict[str, Any], tool_name: str) -> bool:
    message = body.get("message", {}) if isinstance(body, dict) else {}
    for call in message.get("tool_calls", []) or []:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        if function.get("name") == tool_name:
            return True
    return False


def _ollama_usage(body: dict[str, Any]) -> dict[str, Any]:
    usage = {
        "prompt_eval_count": body.get("prompt_eval_count"),
        "eval_count": body.get("eval_count"),
    }
    return {key: value for key, value in usage.items() if value is not None}


def _ollama_embeddings(body: dict[str, Any]) -> list[list[float]]:
    raw = body.get("embeddings")
    if raw is None and "embedding" in body:
        raw = [body["embedding"]]
    embeddings: list[list[float]] = []
    for item in raw or []:
        if isinstance(item, list):
            embeddings.append([float(value) for value in item])
    return embeddings
