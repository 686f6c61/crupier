"""NaN Builders adapter using its OpenAI-compatible API."""

from __future__ import annotations

import json
import mimetypes
import re
from pathlib import Path
from typing import Any, NoReturn

from crupier.config import NAN_DEFAULT_HOST, ProviderSettings
from crupier.errors import (
    CrupierModelUnsupportedError,
    CrupierProviderAuthError,
    CrupierProviderRateLimitError,
    CrupierProviderUnavailableError,
)
from crupier.models import RequestEnvelope
from crupier.multimodal import native_file_payloads
from crupier.structured import schema_from_request

from .base import AdapterResponse, EmbeddingResponse, OperationResponse, ProviderModel
from .common import object_to_dict, provider_timeout_seconds, request_timeout_seconds, require_api_key

_IMAGE_MODELS = {"mimo-v2.5", "gemma4", "qwen3.6"}
_AUDIO_MODELS = {"mimo-v2.5"}
_CHAT_MODELS = {"deepseek-v4-flash", "mimo-v2.5", "gemma4", "qwen3.6"}
_EMBEDDING_MODEL = "qwen3-embedding"
_EMBEDDING_DIMENSIONS = 4096
_OPERATION_MODELS = {
    "reranker": {"rerank"},
    "transcription": {"whisper"},
    "tts": {"kokoro"},
    "image_generation": {"flux-2-klein"},
}
_AUDIO_RESPONSE_FORMATS = {"mp3", "wav", "flac", "aac", "pcm", "opus"}
_TRANSCRIPTION_RESPONSE_FORMATS = {"json", "verbose_json"}
_IMAGE_RESPONSE_FORMATS = {"url", "b64_json"}
_MAX_UPLOAD_BYTES = 25_000_000
_IMAGE_SIZE_RE = re.compile(r"^(\d+)x(\d+)$")


class NaNAdapter:
    provider = "nan"

    def __init__(self, settings: ProviderSettings):
        self.settings = settings
        self._client: Any = None

    @staticmethod
    def supports_file_kind(*, model: str, kind: str) -> bool:
        if kind == "image":
            return model in _IMAGE_MODELS
        if kind == "audio":
            return model in _AUDIO_MODELS
        return False

    def generate(self, *, model: str, prompt: str, request: RequestEnvelope) -> AdapterResponse:
        if model not in _CHAT_MODELS:
            raise CrupierModelUnsupportedError(f"NaN model {model!r} is not a chat-generation model.")
        client = self._client or self._build_client()
        self._client = client
        messages, multimodal = _messages(model=model, prompt=prompt, request=request)
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
                    "name": _schema_name(request),
                    "strict": bool(request.constraints.get("strict_response_schema", True)),
                    "schema": schema,
                },
            }
        _apply_reasoning_options(payload, model=model, request=request)

        try:
            response = client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)
        choice = _first_choice(response)
        message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
        return AdapterResponse(
            text=_message_text(message),
            raw=response,
            usage=object_to_dict(getattr(response, "usage", None)),
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "chat.completions.create",
                "multimodal_images": multimodal["images"],
                "multimodal_audio": multimodal["audio"],
                "response_format": "json_schema" if schema is not None else None,
                "reasoning_mode": _reasoning_mode(model, request),
            },
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
        for item in data or []:
            model_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
            if model_id:
                metadata = item if isinstance(item, dict) else object_to_dict(item)
                models.append(ProviderModel(id=str(model_id), provider=self.provider, metadata=metadata))
        return sorted(models, key=lambda item: item.id)

    def embed(self, *, model: str, input: Any, dimensions: int | None = None) -> EmbeddingResponse:
        if model != _EMBEDDING_MODEL:
            raise CrupierModelUnsupportedError(f"NaN model {model!r} is not its embedding model.")
        if dimensions is not None and dimensions != _EMBEDDING_DIMENSIONS:
            raise CrupierModelUnsupportedError(
                f"NaN {_EMBEDDING_MODEL} returns {_EMBEDDING_DIMENSIONS} dimensions; requested {dimensions}."
            )
        client = self._client or self._build_client()
        self._client = client
        try:
            response = client.embeddings.create(model=model, input=input)
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)
        data = getattr(response, "data", response.get("data", []) if isinstance(response, dict) else [])
        embeddings: list[list[float]] = []
        for item in data or []:
            vector = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
            if vector is not None:
                embeddings.append([float(value) for value in vector])
        return EmbeddingResponse(
            embeddings=embeddings,
            raw=response,
            usage=object_to_dict(getattr(response, "usage", None)),
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "embeddings.create",
                "embedding_dimensions": _EMBEDDING_DIMENSIONS,
            },
        )

    @staticmethod
    def supports_operation(*, operation: str, model: str) -> bool:
        return model in _OPERATION_MODELS.get(operation, set())

    def execute_operation(
        self,
        *,
        operation: str,
        model: str,
        request: RequestEnvelope,
        payload: dict[str, Any],
    ) -> OperationResponse:
        if not self.supports_operation(operation=operation, model=model):
            raise CrupierModelUnsupportedError(
                f"NaN model {model!r} cannot execute operation {operation!r}."
            )
        client = self._client or self._build_client()
        self._client = client
        timeout = request_timeout_seconds(request)
        with_options = getattr(client, "with_options", None)
        if timeout is not None and callable(with_options):
            client = with_options(timeout=timeout)
        try:
            if operation == "reranker":
                return self._rerank(client, model=model, payload=payload)
            if operation == "transcription":
                return self._transcribe(client, model=model, payload=payload)
            if operation == "tts":
                return self._synthesize(client, model=model, payload=payload)
            if operation == "image_generation":
                return self._image(client, model=model, payload=payload)
        except CrupierModelUnsupportedError:
            raise
        except (ValueError, TypeError) as exc:
            raise CrupierModelUnsupportedError(f"Invalid {operation} payload: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)
        raise CrupierModelUnsupportedError(f"Unsupported NaN operation {operation!r}.")

    def _rerank(self, client: Any, *, model: str, payload: dict[str, Any]) -> OperationResponse:
        query = str(payload.get("query") or "").strip()
        documents = payload.get("documents")
        if not query:
            raise CrupierModelUnsupportedError("Rerank requires a non-empty query.")
        if not isinstance(documents, list) or not documents or not all(isinstance(item, str) for item in documents):
            raise CrupierModelUnsupportedError("Rerank documents must be a non-empty list of strings.")
        body: dict[str, Any] = {"model": model, "query": query, "documents": documents}
        top_n = payload.get("top_n")
        if top_n is not None:
            top_n = int(top_n)
            if top_n <= 0 or top_n > len(documents):
                raise CrupierModelUnsupportedError("Rerank top_n must be between 1 and the document count.")
            body["top_n"] = top_n
        response = client.post(path="/rerank", cast_to=object, body=body)
        data = object_to_dict(response)
        raw_results = data.get("results", []) if isinstance(data, dict) else []
        results = [
            {
                "index": int(item.get("index", 0)),
                "relevance_score": float(item.get("relevance_score", 0.0)),
                **({"document": item["document"]} if "document" in item else {}),
            }
            for item in raw_results
            if isinstance(item, dict)
        ]
        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        tokens = meta.get("tokens", {}) if isinstance(meta, dict) else {}
        usage = {
            "input_tokens": int(tokens.get("input_tokens", 0))
        } if isinstance(tokens, dict) and tokens.get("input_tokens") is not None else {}
        return OperationResponse(
            operation="reranker",
            output=results,
            raw=response,
            usage=usage,
            metadata={"provider": self.provider, "model": model, "api": "/v1/rerank"},
        )

    def _transcribe(self, client: Any, *, model: str, payload: dict[str, Any]) -> OperationResponse:
        upload = _upload_tuple(
            payload.get("file"),
            filename=payload.get("filename"),
            default_name="audio.mp3",
        )
        response_format = str(payload.get("response_format") or "json")
        if response_format not in _TRANSCRIPTION_RESPONSE_FORMATS:
            raise CrupierModelUnsupportedError("NaN transcription response_format must be json or verbose_json.")
        body: dict[str, Any] = {"model": model, "file": upload, "response_format": response_format}
        for key in ("language", "temperature"):
            if payload.get(key) is not None:
                body[key] = payload[key]
        granularities = payload.get("timestamp_granularities")
        if granularities is not None:
            if response_format != "verbose_json":
                raise CrupierModelUnsupportedError(
                    "timestamp_granularities requires response_format='verbose_json'."
                )
            if not isinstance(granularities, list) or not set(granularities).issubset({"word", "segment"}):
                raise CrupierModelUnsupportedError(
                    "timestamp_granularities must contain only 'word' or 'segment'."
                )
            body["timestamp_granularities"] = granularities
        response = client.audio.transcriptions.create(**body)
        output = object_to_dict(response)
        if not output and getattr(response, "text", None) is not None:
            output = {"text": str(response.text)}
        return OperationResponse(
            operation="transcription",
            output=output,
            raw=response,
            metadata={"provider": self.provider, "model": model, "api": "/v1/audio/transcriptions"},
        )

    def _synthesize(self, client: Any, *, model: str, payload: dict[str, Any]) -> OperationResponse:
        input_text = str(payload.get("input") or "").strip()
        voice = str(payload.get("voice") or "").strip()
        response_format = str(payload.get("response_format") or "mp3")
        speed = float(payload.get("speed", 1.0))
        if not input_text or not voice:
            raise CrupierModelUnsupportedError("Text-to-speech requires non-empty input and voice.")
        if response_format not in _AUDIO_RESPONSE_FORMATS:
            raise CrupierModelUnsupportedError(
                "NaN speech response_format must be one of: " + ", ".join(sorted(_AUDIO_RESPONSE_FORMATS)) + "."
            )
        if speed <= 0:
            raise CrupierModelUnsupportedError("Text-to-speech speed must be positive.")
        response = client.audio.speech.create(
            model=model,
            input=input_text,
            voice=voice,
            response_format=response_format,
            speed=speed,
        )
        audio = _response_bytes(response)
        if not audio:
            raise CrupierProviderUnavailableError("NaN text-to-speech returned no audio bytes.", retryable=False)
        return OperationResponse(
            operation="tts",
            output=audio,
            raw=response,
            metadata={
                "provider": self.provider,
                "model": model,
                "api": "/v1/audio/speech",
                "response_format": response_format,
                "bytes": len(audio),
            },
        )

    def _image(self, client: Any, *, model: str, payload: dict[str, Any]) -> OperationResponse:
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise CrupierModelUnsupportedError("Image generation requires a non-empty prompt.")
        n = int(payload.get("n", 1))
        if n < 1 or n > 4:
            raise CrupierModelUnsupportedError("NaN image n must be between 1 and 4.")
        size = str(payload.get("size") or "1024x1024")
        _validate_image_size(size)
        response_format = str(payload.get("response_format") or "url")
        if response_format not in _IMAGE_RESPONSE_FORMATS:
            raise CrupierModelUnsupportedError("NaN image response_format must be url or b64_json.")
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": response_format,
        }
        extra_body = {
            key: payload[key]
            for key in ("seed", "guidance")
            if payload.get(key) is not None
        }
        if extra_body:
            body["extra_body"] = extra_body
        images = payload.get("images")
        if images is not None:
            if payload.get("mask") is not None:
                raise CrupierModelUnsupportedError("NaN flux-2-klein image edits do not support masks.")
            raw_images = images if isinstance(images, list) else [images]
            if not raw_images or len(raw_images) > 4:
                raise CrupierModelUnsupportedError("NaN image edits require between 1 and 4 reference images.")
            body["image"] = [
                _upload_tuple(item, default_name=f"reference-{index + 1}.png")
                for index, item in enumerate(raw_images)
            ]
            response = client.images.edit(**body)
            api = "/v1/images/edits"
        else:
            response = client.images.generate(**body)
            api = "/v1/images/generations"
        data = object_to_dict(response)
        output = data.get("data", []) if isinstance(data, dict) else []
        return OperationResponse(
            operation="image_generation",
            output=output,
            raw=response,
            metadata={"provider": self.provider, "model": model, "api": api, "count": len(output)},
        )

    def probe_capability(self, *, model: str, probe: str, request: RequestEnvelope) -> AdapterResponse:
        if probe == "structured_output":
            schema = {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}, "probe": {"type": "string"}},
                "required": ["ok", "probe"],
                "additionalProperties": False,
            }
            probe_request = RequestEnvelope(
                task="Return a successful Crupier probe.",
                response_schema=schema,
                constraints={
                    **request.constraints,
                    "max_output_tokens": max(512, int(request.constraints.get("max_output_tokens", 512))),
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
            response.text = ""
            response.metadata.update(
                {
                    "native_probe": True,
                    "ok": parsed.get("ok") is True and parsed.get("probe") == "crupier",
                    "probe_status": "verified"
                    if parsed.get("ok") is True and parsed.get("probe") == "crupier"
                    else "failed",
                }
            )
            return response
        if probe == "tool_call":
            return self._probe_tool_call(model=model, request=request)
        if probe == "streaming":
            return self._probe_streaming(model=model, request=request)
        raise NotImplementedError(f"NaN adapter has no native probe registered for {probe!r}.")

    def _probe_tool_call(self, *, model: str, request: RequestEnvelope) -> AdapterResponse:
        client = self._client or self._build_client()
        self._client = client
        tool_name = "crupier_probe_tool"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Call crupier_probe_tool with ok=true."}],
            "max_tokens": 128,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": "Report a successful capability probe.",
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
        try:
            response = client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001
            self._raise_mapped_error(exc)
        choice = _first_choice(response)
        message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
        ok = _has_tool_call(message, tool_name)
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
            "max_tokens": int(request.constraints.get("max_output_tokens", 512)),
            "stream": True,
        }
        if model in {"qwen3.6", "gemma4"}:
            payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        timeout = request_timeout_seconds(request)
        if timeout is not None:
            payload["timeout"] = timeout
        try:
            stream = client.chat.completions.create(**payload)
            event_count = 0
            text_seen = False
            for event in stream:
                event_count += 1
                choice = _first_choice(event)
                delta = choice.get("delta") if isinstance(choice, dict) else getattr(choice, "delta", None)
                text_seen = text_seen or bool(_message_text(delta))
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
                "api": "chat.completions.create",
                "native_probe": True,
                "ok": ok,
                "probe_status": "verified" if ok else "failed",
                "event_count": event_count,
                "text_event_seen": text_seen,
            },
        )

    def _build_client(self) -> Any:
        api_key = require_api_key(self.settings, "NAN_API_KEY", provider=self.provider)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise CrupierProviderUnavailableError(
                "NaN adapter requires the optional dependency: pip install 'crupier[nan]'.",
                retryable=False,
            ) from exc
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": self.settings.host or NAN_DEFAULT_HOST,
        }
        timeout = provider_timeout_seconds(self.settings)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return OpenAI(**kwargs)

    def _raise_mapped_error(self, exc: Exception) -> NoReturn:
        name = exc.__class__.__name__.lower()
        status = getattr(exc, "status_code", None)
        if status in {401, 403} or "auth" in name or "permission" in name:
            raise CrupierProviderAuthError(str(exc), provider=self.provider, env_key=self.settings.env_key) from exc
        if status == 429 or "ratelimit" in name or "rate_limit" in name:
            raise CrupierProviderRateLimitError(str(exc)) from exc
        raise CrupierProviderUnavailableError(f"NaN request failed: {exc}") from exc


def _messages(*, model: str, prompt: str, request: RequestEnvelope) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not request.files:
        return [{"role": "user", "content": prompt}], {"images": 0, "audio": 0}
    allowed: set[str] = set()
    if model in _IMAGE_MODELS:
        allowed.add("image")
    if model in _AUDIO_MODELS:
        allowed.add("audio")
    payloads = native_file_payloads(
        request.files,
        allowed_kinds=allowed,
        max_bytes=int(request.constraints.get("max_native_file_bytes", 20_000_000)),
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    counts = {"images": 0, "audio": 0}
    for item in payloads:
        if item["kind"] == "image":
            content.append({"type": "image_url", "image_url": {"url": item["data_url"]}})
            counts["images"] += 1
        elif item["kind"] == "audio":
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": item["base64"], "format": _audio_format(item)},
                }
            )
            counts["audio"] += 1
    return [{"role": "user", "content": content}], counts


def _audio_format(payload: dict[str, str]) -> str:
    suffix = Path(payload["name"]).suffix.lower().lstrip(".")
    if suffix in {"wav", "mp3"}:
        return suffix
    mime = payload["mime_type"].lower()
    if "wav" in mime:
        return "wav"
    if "mpeg" in mime or "mp3" in mime:
        return "mp3"
    raise CrupierModelUnsupportedError("NaN native audio currently requires WAV or MP3 input.")


def _apply_reasoning_options(payload: dict[str, Any], *, model: str, request: RequestEnvelope) -> None:
    if model == "deepseek-v4-flash" and "reasoning_effort" in request.constraints:
        effort = str(request.constraints["reasoning_effort"])
        if effort not in {"low", "medium", "high"}:
            raise CrupierModelUnsupportedError("reasoning_effort must be low, medium, or high for deepseek-v4-flash.")
        payload["reasoning_effort"] = effort
    if model in {"qwen3.6", "gemma4"}:
        enabled = request.constraints.get("enable_thinking")
        if "disable_thinking" in request.constraints:
            enabled = not bool(request.constraints["disable_thinking"])
        if enabled is not None:
            payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(enabled)}}


def _reasoning_mode(model: str, request: RequestEnvelope) -> str:
    if model == "mimo-v2.5":
        return "always"
    if model == "deepseek-v4-flash":
        return str(request.constraints.get("reasoning_effort", "provider_default"))
    if model in {"qwen3.6", "gemma4"}:
        if "disable_thinking" in request.constraints:
            return "disabled" if request.constraints["disable_thinking"] else "enabled"
        if "enable_thinking" in request.constraints:
            return "enabled" if request.constraints["enable_thinking"] else "disabled"
    return "provider_default"


def _first_choice(response: Any) -> Any:
    choices = response.get("choices", []) if isinstance(response, dict) else getattr(response, "choices", [])
    return choices[0] if choices else {}


def _message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text:
                chunks.append(str(text))
        return "".join(chunks)
    return "" if content is None else str(content)


def _has_tool_call(message: Any, tool_name: str) -> bool:
    calls = message.get("tool_calls", []) if isinstance(message, dict) else getattr(message, "tool_calls", [])
    for call in calls or []:
        function = call.get("function", {}) if isinstance(call, dict) else getattr(call, "function", None)
        name = function.get("name") if isinstance(function, dict) else getattr(function, "name", None)
        if name == tool_name:
            return True
    return False


def _schema_name(request: RequestEnvelope) -> str:
    value = request.constraints.get("response_schema_name")
    return value.strip() if isinstance(value, str) and value.strip() else "crupier_response"


def _upload_tuple(
    value: Any,
    *,
    filename: Any = None,
    default_name: str,
) -> tuple[str, bytes, str]:
    name = str(filename or "").strip()
    data: bytes
    if isinstance(value, tuple) and len(value) in {2, 3}:
        tuple_name, tuple_data = value[0], value[1]
        name = name or str(tuple_name or default_name)
        if not isinstance(tuple_data, bytes | bytearray):
            raise CrupierModelUnsupportedError("Upload tuple content must be bytes.")
        data = bytes(tuple_data)
        tuple_mime = str(value[2]) if len(value) == 3 and value[2] else None
        if not data:
            raise CrupierModelUnsupportedError("Upload input cannot be empty.")
        if len(data) >= _MAX_UPLOAD_BYTES:
            raise CrupierModelUnsupportedError("NaN uploads must be smaller than 25 MB.")
        return name, data, tuple_mime or mimetypes.guess_type(name)[0] or "application/octet-stream"
    if isinstance(value, str | Path):
        path = Path(value).expanduser()
        if not path.is_file():
            raise CrupierModelUnsupportedError(f"Upload file does not exist: {path}")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise CrupierModelUnsupportedError(f"Upload file cannot be inspected: {path}") from exc
        if size >= _MAX_UPLOAD_BYTES:
            raise CrupierModelUnsupportedError("NaN uploads must be smaller than 25 MB.")
        name = name or path.name
        try:
            with path.open("rb") as handle:
                data = handle.read(_MAX_UPLOAD_BYTES)
        except OSError as exc:
            raise CrupierModelUnsupportedError(f"Upload file cannot be read: {path}") from exc
    elif isinstance(value, bytes | bytearray):
        name = name or default_name
        data = bytes(value)
    else:
        read = getattr(value, "read", None)
        if not callable(read):
            raise CrupierModelUnsupportedError("Upload input must be a path, bytes, or binary file object.")
        position = None
        tell = getattr(value, "tell", None)
        seek = getattr(value, "seek", None)
        if callable(tell):
            try:
                position = tell()
            except (OSError, ValueError):
                position = None
        try:
            raw = read(_MAX_UPLOAD_BYTES)
        except TypeError as exc:
            raise CrupierModelUnsupportedError(
                "Upload file objects must support bounded read(size) calls."
            ) from exc
        finally:
            if position is not None and callable(seek):
                try:
                    seek(position)
                except (OSError, ValueError):
                    pass
        if not isinstance(raw, bytes | bytearray):
            raise CrupierModelUnsupportedError("Upload file object must return bytes.")
        name = name or str(getattr(value, "name", "") or default_name)
        data = bytes(raw)
    if not data:
        raise CrupierModelUnsupportedError("Upload input cannot be empty.")
    if len(data) >= _MAX_UPLOAD_BYTES:
        raise CrupierModelUnsupportedError("NaN uploads must be smaller than 25 MB.")
    mime_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return name, data, mime_type


def _response_bytes(response: Any) -> bytes:
    if isinstance(response, bytes | bytearray):
        return bytes(response)
    content = getattr(response, "content", None)
    if isinstance(content, bytes | bytearray):
        return bytes(content)
    read = getattr(response, "read", None)
    if callable(read):
        value = read()
        if isinstance(value, bytes | bytearray):
            return bytes(value)
    return b""


def _validate_image_size(size: str) -> None:
    if size == "auto":
        return
    match = _IMAGE_SIZE_RE.fullmatch(size)
    if not match:
        raise CrupierModelUnsupportedError("NaN image size must be 'auto' or WIDTHxHEIGHT.")
    width, height = (int(value) for value in match.groups())
    if not 256 <= width <= 1536 or not 256 <= height <= 1536:
        raise CrupierModelUnsupportedError("NaN image width and height must be between 256 and 1536 pixels.")
    if width % 16 or height % 16:
        raise CrupierModelUnsupportedError("NaN image width and height must be divisible by 16.")
    ratio = width / height
    if ratio < 1 / 3 or ratio > 3:
        raise CrupierModelUnsupportedError("NaN image aspect ratio must be between 1:3 and 3:1.")
