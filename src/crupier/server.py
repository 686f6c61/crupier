"""Small OpenAI-compatible HTTP server for drop-in adoption."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from email import policy
from email.parser import BytesParser
from hmac import compare_digest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from .client import Crupier
from .compat.openai import OpenAI
from .errors import (
    CrupierApprovalRequired,
    CrupierBudgetExceededError,
    CrupierConfigError,
    CrupierError,
    CrupierExecutionLimitError,
    CrupierModelUnsupportedError,
    CrupierPolicyError,
    CrupierProviderAuthError,
    CrupierProviderRateLimitError,
    CrupierProviderUnavailableError,
    CrupierRouteValidationError,
    CrupierStructuredOutputError,
    CrupierToolApprovalRequired,
    CrupierUpdateRequiresConfirmation,
)

_OPENAI_HTTP_ALLOWED_FIELDS = {
    "/v1/responses": frozenset(
        {
            "background",
            "frequency_penalty",
            "include",
            "include_obfuscation",
            "input",
            "instructions",
            "logit_bias",
            "max_output_tokens",
            "max_tokens",
            "model",
            "parallel_tool_calls",
            "presence_penalty",
            "previous_response_id",
            "reasoning",
            "response_format",
            "seed",
            "service_tier",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "truncation",
            "user",
        }
    ),
    "/v1/chat/completions": frozenset(
        {
            "audio",
            "frequency_penalty",
            "functions",
            "function_call",
            "logit_bias",
            "logprobs",
            "max_completion_tokens",
            "max_tokens",
            "messages",
            "modalities",
            "model",
            "n",
            "parallel_tool_calls",
            "prediction",
            "presence_penalty",
            "reasoning_effort",
            "response_format",
            "seed",
            "service_tier",
            "stop",
            "store",
            "stream",
            "stream_options",
            "temperature",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "user",
        }
    ),
    "/v1/embeddings": frozenset({"dimensions", "encoding_format", "input", "model", "user"}),
    "/v1/rerank": frozenset({"documents", "model", "query", "return_documents", "top_n"}),
    "/v2/rerank": frozenset({"documents", "model", "query", "return_documents", "top_n"}),
    "/v1/images/generations": frozenset(
        {"guidance", "model", "n", "prompt", "quality", "response_format", "seed", "size", "style", "user"}
    ),
    "/v1/images/edits": frozenset(
        {"image", "mask", "model", "n", "prompt", "response_format", "size", "user"}
    ),
    "/v1/audio/speech": frozenset({"input", "model", "response_format", "speed", "voice"}),
    "/v1/audio/transcriptions": frozenset(
        {"file", "language", "model", "prompt", "response_format", "temperature", "timestamp_granularities"}
    ),
}


def build_openai_compatible_server(
    *,
    crupier: Crupier,
    host: str = "127.0.0.1",
    port: int = 8787,
    dry_run: bool | None = None,
    compat_mode: str = "balanced",
    allow_remote: bool = False,
    cors_origin: str | None = None,
    max_request_bytes: int = 10_000_000,
    file_root: str | Path | None = None,
    bearer_token: str | None = None,
    authenticator: Callable[[str], bool] | None = None,
) -> ThreadingHTTPServer:
    """Create a stdlib HTTP server exposing a small OpenAI-compatible API."""

    if not allow_remote and not _is_loopback_bind_host(host):
        raise CrupierConfigError(
            "crupier serve binds to loopback by default. Pass allow_remote=True or CLI --allow-remote "
            "only when this compatibility server is protected by your own network/auth boundary."
        )
    if bearer_token is not None and not bearer_token.strip():
        raise CrupierConfigError("The server bearer token cannot be empty.")
    authentication_configured = bearer_token is not None or authenticator is not None
    if allow_remote and not authentication_configured:
        raise CrupierConfigError("Remote binds require a configured authentication token or authenticator.")
    if dry_run is not True and not authentication_configured:
        raise CrupierConfigError("Live server execution requires a configured authentication token or authenticator.")
    compat_client = OpenAI(
        crupier=crupier,
        dry_run=dry_run,
        compat_mode=compat_mode,
        allow_local_file_uris=False,
        file_root=file_root,
        allow_request_controls=False,
    )

    class Handler(_OpenAICompatibleHandler):
        client = compat_client
        crupier_client = crupier
        browser_origin = cors_origin
        request_body_limit = max(1, int(max_request_bytes))
        bind_host = host
        expected_bearer_token = bearer_token
        request_authenticator = authenticator
        require_authentication = allow_remote or dry_run is not True

    return ThreadingHTTPServer((host, port), Handler)


def serve_openai_compatible(
    *,
    crupier: Crupier,
    host: str = "127.0.0.1",
    port: int = 8787,
    dry_run: bool | None = None,
    compat_mode: str = "balanced",
    allow_remote: bool = False,
    cors_origin: str | None = None,
    max_request_bytes: int = 10_000_000,
    file_root: str | Path | None = None,
    bearer_token: str | None = None,
    authenticator: Callable[[str], bool] | None = None,
) -> None:
    server = build_openai_compatible_server(
        crupier=crupier,
        host=host,
        port=port,
        dry_run=dry_run,
        compat_mode=compat_mode,
        allow_remote=allow_remote,
        cors_origin=cors_origin,
        max_request_bytes=max_request_bytes,
        file_root=file_root,
        bearer_token=bearer_token,
        authenticator=authenticator,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


class _OpenAICompatibleHandler(BaseHTTPRequestHandler):
    server_version = "crupier-openai-compat/0.1"
    client: OpenAI
    crupier_client: Crupier
    browser_origin: str | None = None
    request_body_limit: int = 10_000_000
    bind_host: str = "127.0.0.1"
    expected_bearer_token: str | None = None
    request_authenticator: Callable[[str], bool] | None = None
    require_authentication: bool = True

    def do_OPTIONS(self) -> None:
        if not self._validate_request_context(authenticate=False):
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_common_headers("application/json")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._validate_request_context(authenticate=False):
            return
        path = self._request_path()
        if path in {"/health", "/v1/health"}:
            self._write_json({"ok": True, "service": "crupier", "compat": "openai"})
            return
        if path == "/v1/models":
            self._write_json(_models_payload(self.crupier_client))
            return
        self._write_error(HTTPStatus.NOT_FOUND, f"Unknown endpoint {path!r}.", error_type="invalid_request_error")

    def do_POST(self) -> None:
        try:
            path = self._request_path()
            if not self._validate_request_context(authenticate=self.require_authentication):
                return
            if not self._validate_content_type(path):
                return
            if path in {"/v1/audio/transcriptions", "/v1/images/edits"}:
                payload = self._read_multipart()
            else:
                payload = self._read_json()
            _validate_openai_http_payload(path, payload)
            if path in {"/v1/responses", "/v1/chat/completions"}:
                self._apply_control_headers(payload)
            if path == "/v1/responses":
                self._handle_response(payload)
            elif path == "/v1/chat/completions":
                self._handle_chat_completion(payload)
            elif path == "/v1/embeddings":
                self._handle_embeddings(payload)
            elif path in {"/v1/rerank", "/v2/rerank"}:
                self._handle_rerank(payload)
            elif path == "/v1/images/generations":
                self._handle_image_generation(payload)
            elif path == "/v1/images/edits":
                self._handle_image_edit(payload)
            elif path == "/v1/audio/speech":
                self._handle_audio_speech(payload)
            elif path == "/v1/audio/transcriptions":
                self._handle_audio_transcription(payload)
            else:
                self._write_error(HTTPStatus.NOT_FOUND, f"Unknown endpoint {path!r}.", error_type="invalid_request_error")
        except CrupierError as exc:
            self._write_crupier_error(exc)
        except _RequestBodyTooLarge as exc:
            self.close_connection = True
            self._write_error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                str(exc),
                error_type="invalid_request_error",
                code="request_too_large",
            )
        except _LengthRequired as exc:
            self.close_connection = True
            self._write_error(
                HTTPStatus.LENGTH_REQUIRED,
                str(exc),
                error_type="invalid_request_error",
                code="length_required",
            )
        except ValueError as exc:
            self._write_error(
                HTTPStatus.BAD_REQUEST,
                str(exc),
                error_type="invalid_request_error",
                code="invalid_request",
            )
        except TypeError as exc:
            self._write_error(
                HTTPStatus.BAD_REQUEST,
                str(exc),
                error_type="invalid_request_error",
                code="invalid_request",
            )
        except Exception as exc:  # noqa: BLE001 - server boundary converts unexpected errors
            del exc
            self._write_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Internal server error.",
                error_type="server_error",
                code="internal_error",
            )

    def _validate_request_context(self, *, authenticate: bool) -> bool:
        host = self.headers.get("host", "")
        if not _host_matches_bind(host, self.bind_host):
            self._write_error(
                HTTPStatus.BAD_REQUEST,
                "Host header is not allowed.",
                error_type="invalid_request_error",
                code="host_not_allowed",
            )
            return False
        origin = self.headers.get("origin")
        if origin is not None and self.browser_origin != "*" and origin != self.browser_origin:
            self._write_error(
                HTTPStatus.FORBIDDEN,
                "Origin is not allowed.",
                error_type="invalid_request_error",
                code="origin_not_allowed",
            )
            return False
        if not authenticate:
            return True
        authorization = self.headers.get("authorization", "")
        scheme, separator, credential = authorization.partition(" ")
        valid = separator == " " and scheme.lower() == "bearer" and bool(credential)
        if valid and self.expected_bearer_token is not None:
            valid = compare_digest(credential, self.expected_bearer_token)
        elif valid and self.request_authenticator is not None:
            try:
                valid = bool(self.request_authenticator(credential))
            except Exception:  # noqa: BLE001 - custom authenticator fails closed
                valid = False
        else:
            valid = False
        if valid:
            return True
        self._write_error(
            HTTPStatus.UNAUTHORIZED,
            "Missing or invalid bearer token.",
            error_type="authentication_error",
            code="invalid_api_key",
        )
        return False

    def _validate_content_type(self, path: str) -> bool:
        raw_content_type = self.headers.get("content-type", "")
        media_type = raw_content_type.partition(";")[0].strip().lower()
        expected = (
            "multipart/form-data"
            if path in {"/v1/audio/transcriptions", "/v1/images/edits"}
            else "application/json"
        )
        if media_type == expected:
            return True
        self._write_error(
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            f"This endpoint requires {expected}.",
            error_type="invalid_request_error",
            code="unsupported_media_type",
        )
        return False

    def _handle_response(self, payload: dict[str, Any]) -> None:
        stream = bool(payload.get("stream", False))
        result = self.client.responses.create(**payload)
        if stream:
            self._write_sse(result)
        else:
            self._write_json(_plain(result))

    def _handle_chat_completion(self, payload: dict[str, Any]) -> None:
        if "messages" not in payload:
            raise ValueError("Missing required parameter: 'messages'.")
        if not isinstance(payload["messages"], list):
            raise ValueError("Parameter 'messages' must be a list.")  # noqa: TRY004 - public API error contract
        stream = bool(payload.get("stream", False))
        result = self.client.chat.completions.create(**payload)
        if stream:
            self._write_sse(result)
        else:
            self._write_json(_plain(result))

    def _handle_embeddings(self, payload: dict[str, Any]) -> None:
        if "model" not in payload:
            raise ValueError("Missing required parameter: 'model'.")
        if "input" not in payload:
            raise ValueError("Missing required parameter: 'input'.")
        result = self.client.embeddings.create(**payload)
        self._write_json(_plain(result))

    def _handle_rerank(self, payload: dict[str, Any]) -> None:
        if "query" not in payload or "documents" not in payload:
            raise ValueError("Rerank requires 'query' and 'documents'.")
        result = self.client.rerank.create(**payload)
        self._write_json(_plain(result))

    def _handle_image_generation(self, payload: dict[str, Any]) -> None:
        if "prompt" not in payload:
            raise ValueError("Missing required parameter: 'prompt'.")
        result = self.client.images.generate(**payload)
        self._write_json(_plain(result))

    def _handle_image_edit(self, payload: dict[str, Any]) -> None:
        if "prompt" not in payload or "image" not in payload:
            raise ValueError("Image edits require 'prompt' and at least one 'image'.")
        result = self.client.images.edit(**payload)
        self._write_json(_plain(result))

    def _handle_audio_speech(self, payload: dict[str, Any]) -> None:
        if "input" not in payload or "voice" not in payload:
            raise ValueError("Speech generation requires 'input' and 'voice'.")
        result = self.client.audio.speech.create(**payload)
        response_format = str(payload.get("response_format") or "mp3")
        self._write_bytes(result.read(), content_type=_audio_content_type(response_format))

    def _handle_audio_transcription(self, payload: dict[str, Any]) -> None:
        if "file" not in payload:
            raise ValueError("Missing required multipart field: 'file'.")
        result = self.client.audio.transcriptions.create(**payload)
        self._write_json(_plain(result))

    def _apply_control_headers(self, payload: dict[str, Any]) -> None:
        experiment = self.headers.get("x-crupier-experiment")
        approval_token = self.headers.get("x-crupier-approval")
        if experiment and "experiment" not in payload:
            payload["experiment"] = experiment
        if approval_token and "approval_token" not in payload:
            payload["approval_token"] = approval_token

    def _request_path(self) -> str:
        return urlparse(self.path).path

    def _read_json(self) -> dict[str, Any]:
        raw = self._read_body()
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON request body must be an object.")  # noqa: TRY004 - public API error contract
        return data

    def _read_multipart(self) -> dict[str, Any]:
        content_type = self.headers.get("content-type", "")
        if "multipart/form-data" not in content_type.lower():
            raise ValueError("This endpoint requires multipart/form-data.")
        raw = self._read_body()
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: " + content_type.encode("utf-8") + b"\r\nMIME-Version: 1.0\r\n\r\n" + raw
        )
        if not message.is_multipart():
            raise ValueError("Invalid multipart/form-data body.")
        payload: dict[str, Any] = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            raw_name = str(name)
            is_array = raw_name.endswith("[]")
            key = raw_name.removesuffix("[]")
            filename = part.get_filename()
            decoded_payload = part.get_payload(decode=True)
            if decoded_payload is None:
                raw_value = b""
            elif isinstance(decoded_payload, bytes):
                raw_value = decoded_payload
            else:
                raise ValueError(f"Multipart field {key!r} has an invalid body.")
            if filename:
                value: Any = (filename, raw_value, part.get_content_type())
            else:
                try:
                    value = raw_value.decode(part.get_content_charset() or "utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"Multipart field {key!r} is not valid text.") from exc
                value = _coerce_form_value(key, value)
            if key in payload:
                existing = payload[key]
                payload[key] = [*existing, value] if isinstance(existing, list) else [existing, value]
            elif is_array:
                payload[key] = [value]
            else:
                payload[key] = value
        return payload

    def _read_body(self) -> bytes:
        transfer_encoding = self.headers.get("transfer-encoding")
        if transfer_encoding and transfer_encoding.strip().lower() != "identity":
            self.close_connection = True
            raise ValueError("Transfer-Encoding request bodies are not supported; send Content-Length.")
        content_lengths = self.headers.get_all("content-length", [])
        if not content_lengths:
            raise _LengthRequired("POST requests require a Content-Length header.")
        if len(content_lengths) != 1:
            self.close_connection = True
            raise ValueError("Multiple Content-Length headers are not allowed.")
        raw_length = content_lengths[0]
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        if length < 0:
            raise ValueError("Content-Length cannot be negative.")
        if length > self.request_body_limit:
            raise _RequestBodyTooLarge(
                f"Request body is {length} bytes, above the configured limit of {self.request_body_limit} bytes."
            )
        return self.rfile.read(length) if length else b""

    def _write_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self._send_common_headers("application/json")
        self.send_header("content-length", str(len(body)))
        self.send_header("x-request-id", self._request_id())
        self.end_headers()
        self.wfile.write(body)

    def _write_bytes(self, payload: bytes, *, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self._send_common_headers(content_type)
        self.send_header("content-length", str(len(payload)))
        self.send_header("x-request-id", self._request_id())
        self.end_headers()
        self.wfile.write(payload)

    def _write_sse(self, events: Any) -> None:
        if not isinstance(events, Iterator):
            events = iter([events])
        self.send_response(HTTPStatus.OK)
        self._send_common_headers("text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("x-request-id", self._request_id())
        self.end_headers()
        try:
            for event in events:
                self._write_sse_event(event)
        except Exception as exc:  # noqa: BLE001 - SSE cannot change status after headers
            self._write_sse_event({"type": "error", "error": _openai_error_payload(exc)["error"]})
        self.wfile.write(b"data: [DONE]\n\n")

    def _write_sse_event(self, event: Any) -> None:
        payload = _plain(event)
        if isinstance(payload, dict) and isinstance(payload.get("type"), str):
            self.wfile.write(f"event: {payload['type']}\n".encode())
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self.wfile.write(f"data: {data}\n\n".encode())

    def _write_crupier_error(self, exc: CrupierError) -> None:
        status, error_type, code = _error_contract(exc)
        self._write_error(status, str(exc), error_type=error_type, code=code)

    def _write_error(
        self,
        status: HTTPStatus,
        message: str,
        *,
        error_type: str = "crupier_error",
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        self._write_json(
            {
                "error": {
                    "message": _sanitize_error_message(message),
                    "type": error_type,
                    "param": param,
                    "code": code,
                }
            },
            status=status,
        )

    def _send_common_headers(self, content_type: str) -> None:
        self.send_header("content-type", content_type)
        if self.browser_origin:
            self.send_header("access-control-allow-origin", self.browser_origin)
            self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
            self.send_header(
                "access-control-allow-headers",
                "authorization, content-type, x-crupier-approval, x-crupier-experiment",
            )
            if self.browser_origin != "*":
                self.send_header("vary", "Origin")

    def _request_id(self) -> str:
        request_id = getattr(self, "_crupier_request_id", None)
        if request_id is None:
            request_id = f"req_{uuid4().hex[:24]}"
            self._crupier_request_id = request_id
        return request_id

    def log_message(self, format: str, *args: Any) -> None:
        return


class _RequestBodyTooLarge(Exception):
    pass


class _LengthRequired(Exception):
    pass


def _models_payload(client: Crupier) -> dict[str, Any]:
    data = []
    for card in client.models.list(allowed_only=True):
        data.append(
            {
                "id": card.model_ref.key,
                "object": "model",
                "owned_by": card.model_ref.provider,
                "created": 0,
            }
        )
    return {"object": "list", "data": data}


def _is_loopback_bind_host(host: str) -> bool:
    lowered = host.strip().lower()
    if lowered == "localhost":
        return True
    try:
        return ip_address(lowered).is_loopback
    except ValueError:
        return False


def _host_matches_bind(host_header: str, bind_host: str) -> bool:
    if not host_header or any(character.isspace() for character in host_header):
        return False
    parsed = urlparse(f"//{host_header}")
    request_host = parsed.hostname
    if request_host is None:
        return False
    normalized_bind = bind_host.strip().lower()
    normalized_request = request_host.lower()
    if _is_loopback_bind_host(normalized_bind):
        return normalized_request == "localhost" or _is_loopback_bind_host(normalized_request)
    if normalized_bind in {"0.0.0.0", "::"}:
        return True
    return normalized_request == normalized_bind


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _validate_openai_http_payload(path: str, payload: dict[str, Any]) -> None:
    """Rechaza cualquier parámetro de red que no pertenezca al contrato OpenAI."""

    allowed = _OPENAI_HTTP_ALLOWED_FIELDS.get(path)
    if allowed is None:
        return
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        joined = ", ".join(repr(key) for key in unexpected)
        raise ValueError(f"Unsupported HTTP parameter(s): {joined}.")


def _coerce_form_value(name: str, value: str) -> Any:
    if name in {"n", "seed", "top_n"}:
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"Multipart field {name!r} must be an integer.") from exc
    if name in {"guidance", "speed", "temperature"}:
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Multipart field {name!r} must be a number.") from exc
    return value


def _audio_content_type(response_format: str) -> str:
    return {
        "aac": "audio/aac",
        "flac": "audio/flac",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "pcm": "audio/L16",
        "wav": "audio/wav",
    }.get(response_format.lower(), "application/octet-stream")


def _error_contract(exc: Exception) -> tuple[HTTPStatus, str, str]:
    if isinstance(exc, CrupierApprovalRequired):
        return HTTPStatus.CONFLICT, "invalid_request_error", "approval_required"
    if isinstance(exc, CrupierProviderAuthError):
        return HTTPStatus.UNAUTHORIZED, "authentication_error", "invalid_api_key"
    if isinstance(exc, CrupierProviderRateLimitError):
        return HTTPStatus.TOO_MANY_REQUESTS, "rate_limit_error", "rate_limit_exceeded"
    if isinstance(exc, CrupierProviderUnavailableError):
        return HTTPStatus.SERVICE_UNAVAILABLE, "server_error", "provider_unavailable"
    if isinstance(exc, CrupierModelUnsupportedError):
        return HTTPStatus.BAD_REQUEST, "invalid_request_error", "model_not_supported"
    if isinstance(exc, CrupierBudgetExceededError):
        return HTTPStatus.BAD_REQUEST, "invalid_request_error", "budget_exceeded"
    if isinstance(exc, CrupierExecutionLimitError):
        return HTTPStatus.REQUEST_TIMEOUT, "server_error", "execution_limit_exceeded"
    if isinstance(exc, CrupierToolApprovalRequired):
        return HTTPStatus.BAD_REQUEST, "invalid_request_error", "tool_approval_required"
    if isinstance(exc, CrupierStructuredOutputError):
        return HTTPStatus.BAD_REQUEST, "invalid_request_error", "structured_output_error"
    if isinstance(exc, CrupierUpdateRequiresConfirmation):
        return HTTPStatus.CONFLICT, "invalid_request_error", "update_requires_confirmation"
    if isinstance(exc, CrupierConfigError):
        return HTTPStatus.BAD_REQUEST, "invalid_request_error", "configuration_error"
    if isinstance(exc, CrupierPolicyError):
        return HTTPStatus.BAD_REQUEST, "invalid_request_error", "policy_error"
    if isinstance(exc, CrupierRouteValidationError):
        return HTTPStatus.BAD_REQUEST, "invalid_request_error", "route_validation_error"
    return HTTPStatus.BAD_REQUEST, "invalid_request_error", exc.__class__.__name__


def _openai_error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, CrupierError):
        _, error_type, code = _error_contract(exc)
        message = str(exc)
    else:
        error_type = "server_error"
        code = "internal_error"
        message = "Internal server error."
    payload: dict[str, Any] = {
        "error": {
            "message": _sanitize_error_message(message),
            "type": error_type,
            "param": None,
            "code": code,
        }
    }
    if isinstance(exc, CrupierApprovalRequired):
        payload["error"]["crupier"] = {
            "approval_id": exc.approval_id,
            "trace_id": exc.trace_id,
        }
    return payload


_SECRET_PATTERNS = (
    re.compile(("s" + "k-") + r"[A-Za-z0-9_\-]{10,}"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{12,}", re.IGNORECASE),
    re.compile(r"([A-Z][A-Z0-9_]*_API_KEY=)[^\s]+"),
)


def _sanitize_error_message(message: str) -> str:
    sanitized = message
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.startswith("("):
            sanitized = pattern.sub(r"\1[redacted]", sanitized)
        else:
            sanitized = pattern.sub("[redacted]", sanitized)
    return sanitized
