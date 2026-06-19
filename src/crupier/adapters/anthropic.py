"""Anthropic Claude Messages API adapter."""

from __future__ import annotations

from typing import Any

from crupier.config import ProviderSettings
from crupier.errors import (
    CrupierProviderAuthError,
    CrupierProviderRateLimitError,
    CrupierProviderUnavailableError,
)
from crupier.models import RequestEnvelope
from crupier.multimodal import native_image_payloads

from .base import AdapterResponse, ProviderModel
from .common import build_prompt, extract_anthropic_text, object_to_dict, require_api_key


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(self, settings: ProviderSettings):
        self.settings = settings
        self._client: Any = None

    def generate(self, *, model: str, prompt: str, request: RequestEnvelope) -> AdapterResponse:
        client = self._client or self._build_client()
        self._client = client
        max_tokens = int(request.constraints.get("max_output_tokens", request.constraints.get("max_tokens", 1024)))
        content: Any = prompt or build_prompt(request)
        image_payloads = []
        if request.files:
            image_payloads = native_image_payloads(request.files)
            content = [{"type": "text", "text": content}]
            content.extend(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image["mime_type"],
                        "data": image["base64"],
                    },
                }
                for image in image_payloads
            )
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if "temperature" in request.constraints:
            payload["temperature"] = request.constraints["temperature"]

        message, removed_params = self._messages_create_with_param_repair(client, payload)

        text = extract_anthropic_text(message)
        usage = object_to_dict(getattr(message, "usage", None))
        return AdapterResponse(
            text=text,
            raw=message,
            usage=usage,
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "messages.create",
                "removed_params": removed_params,
                "multimodal_images": len(image_payloads),
            },
        )

    def list_models(self) -> list[ProviderModel]:
        client = self._client or self._build_client()
        self._client = client
        try:
            response = client.models.list()
        except AttributeError as exc:
            raise CrupierProviderUnavailableError(
                "Installed anthropic SDK does not expose client.models.list(); upgrade the optional dependency."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)

        data = getattr(response, "data", response)
        models: list[ProviderModel] = []
        for item in data:
            model_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
            if not model_id:
                continue
            name = item.get("display_name") if isinstance(item, dict) else getattr(item, "display_name", None)
            metadata = item if isinstance(item, dict) else object_to_dict(item)
            models.append(ProviderModel(id=str(model_id), provider=self.provider, name=name, metadata=metadata))
        return sorted(models, key=lambda model: model.id)

    def probe_capability(self, *, model: str, probe: str, request: RequestEnvelope) -> AdapterResponse:
        if probe == "structured_output":
            return self._probe_tool_schema(model=model, request=request, capability="structured_output")
        if probe == "tool_call":
            return self._probe_tool_schema(model=model, request=request, capability="tool_call")
        if probe == "streaming":
            return self._probe_streaming(model=model, request=request)
        raise NotImplementedError(f"Anthropic adapter has no native probe registered for {probe!r}.")

    def _probe_tool_schema(self, *, model: str, request: RequestEnvelope, capability: str) -> AdapterResponse:
        client = self._client or self._build_client()
        self._client = client
        tool_name = "crupier_probe_tool"
        payload = {
            "model": model,
            "max_tokens": int(request.constraints.get("max_output_tokens", 128)),
            "messages": [{"role": "user", "content": "Use the crupier_probe_tool with ok=true and probe='crupier'."}],
            "tools": [
                {
                    "name": tool_name,
                    "description": "Report that the capability probe succeeded.",
                    "input_schema": _probe_schema(),
                }
            ],
            "tool_choice": {"type": "tool", "name": tool_name},
        }
        try:
            message = client.messages.create(**payload)
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)
        ok = _anthropic_has_tool_use(message, tool_name)
        return AdapterResponse(
            text="",
            raw=message,
            usage=object_to_dict(getattr(message, "usage", None)),
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "messages.create",
                "native_probe": True,
                "capability": capability,
                "ok": ok,
                "probe_status": "verified" if ok else "failed",
                "tool_name": tool_name,
            },
        )

    def _probe_streaming(self, *, model: str, request: RequestEnvelope) -> AdapterResponse:
        client = self._client or self._build_client()
        self._client = client
        try:
            stream = client.messages.create(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": 'Reply with exactly: "stream-ok"'}],
                stream=True,
            )
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
        ok = event_count > 0
        return AdapterResponse(
            text="",
            raw=None,
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "messages.create",
                "native_probe": True,
                "ok": ok,
                "probe_status": "verified" if ok else "failed",
                "event_count": event_count,
                "text_event_seen": text_seen,
            },
        )

    def _build_client(self) -> Any:
        api_key = require_api_key(self.settings, "ANTHROPIC_API_KEY", provider=self.provider)
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise CrupierProviderUnavailableError(
                "Anthropic adapter requires the optional dependency: pip install 'crupier[anthropic]'."
            ) from exc
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.settings.host:
            kwargs["base_url"] = self.settings.host
        return Anthropic(**kwargs)

    def _messages_create_with_param_repair(self, client: Any, payload: dict[str, Any]) -> tuple[Any, list[str]]:
        try:
            return client.messages.create(**payload), []
        except Exception as exc:  # noqa: BLE001 - provider SDK exceptions vary by version
            if "temperature" in payload and _is_temperature_deprecated(exc):
                repaired = dict(payload)
                repaired.pop("temperature", None)
                try:
                    return client.messages.create(**repaired), ["temperature"]
                except Exception as repaired_exc:  # noqa: BLE001
                    self._raise_mapped_error(repaired_exc)
            self._raise_mapped_error(exc)

    def _raise_mapped_error(self, exc: Exception) -> None:
        name = exc.__class__.__name__.lower()
        if "auth" in name or "permission" in name:
            raise CrupierProviderAuthError(str(exc), provider=self.provider, env_key=self.settings.env_key) from exc
        if "ratelimit" in name or "rate_limit" in name:
            raise CrupierProviderRateLimitError(str(exc)) from exc
        raise CrupierProviderUnavailableError(f"Anthropic request failed: {exc}") from exc


def _is_temperature_deprecated(exc: Exception) -> bool:
    text = str(exc).lower()
    return "temperature" in text and "deprecated" in text


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


def _anthropic_has_tool_use(message: Any, tool_name: str) -> bool:
    content = message.get("content", []) if isinstance(message, dict) else getattr(message, "content", [])
    for block in content or []:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
        if block_type == "tool_use" and name == tool_name:
            return True
    return False


def _event_has_text(event: Any) -> bool:
    if isinstance(event, dict):
        return bool(event.get("delta") or event.get("text") or event.get("content_block"))
    for attr in ["delta", "text", "content_block"]:
        if getattr(event, attr, None):
            return True
    return False
