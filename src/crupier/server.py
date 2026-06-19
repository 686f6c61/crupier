"""Small OpenAI-compatible HTTP server for drop-in adoption."""

from __future__ import annotations

import json
import re
from ipaddress import ip_address
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from .client import Crupier
from .compat.openai import OpenAI
from .errors import (
    CrupierBudgetExceededError,
    CrupierConfigError,
    CrupierError,
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


def build_openai_compatible_server(
    *,
    crupier: Crupier,
    host: str = "127.0.0.1",
    port: int = 8787,
    dry_run: bool | None = None,
    compat_mode: str = "balanced",
    allow_remote: bool = False,
    cors_origin: str | None = None,
) -> ThreadingHTTPServer:
    """Create a stdlib HTTP server exposing a small OpenAI-compatible API."""

    if not allow_remote and not _is_loopback_bind_host(host):
        raise CrupierConfigError(
            "crupier serve binds to loopback by default. Pass allow_remote=True or CLI --allow-remote "
            "only when this compatibility server is protected by your own network/auth boundary."
        )
    compat_client = OpenAI(crupier=crupier, dry_run=dry_run, compat_mode=compat_mode)

    class Handler(_OpenAICompatibleHandler):
        client = compat_client
        crupier_client = crupier
        browser_origin = cors_origin

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
) -> None:
    server = build_openai_compatible_server(
        crupier=crupier,
        host=host,
        port=port,
        dry_run=dry_run,
        compat_mode=compat_mode,
        allow_remote=allow_remote,
        cors_origin=cors_origin,
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

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler API
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_common_headers("application/json")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = self._request_path()
        if path in {"/health", "/v1/health"}:
            self._write_json({"ok": True, "service": "crupier", "compat": "openai"})
            return
        if path == "/v1/models":
            self._write_json(_models_payload(self.crupier_client))
            return
        self._write_error(HTTPStatus.NOT_FOUND, f"Unknown endpoint {path!r}.", error_type="invalid_request_error")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        try:
            path = self._request_path()
            payload = self._read_json()
            if path == "/v1/responses":
                self._handle_response(payload)
            elif path == "/v1/chat/completions":
                self._handle_chat_completion(payload)
            elif path == "/v1/embeddings":
                self._handle_embeddings(payload)
            else:
                self._write_error(HTTPStatus.NOT_FOUND, f"Unknown endpoint {path!r}.", error_type="invalid_request_error")
        except CrupierError as exc:
            self._write_crupier_error(exc)
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
            raise ValueError("Parameter 'messages' must be a list.")
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

    def _request_path(self) -> str:
        return urlparse(self.path).path

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON request body must be an object.")
        return data

    def _write_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self._send_common_headers("application/json")
        self.send_header("content-length", str(len(body)))
        self.send_header("x-request-id", self._request_id())
        self.end_headers()
        self.wfile.write(body)

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
            self.wfile.write(f"event: {payload['type']}\n".encode("utf-8"))
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))

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
            self.send_header("access-control-allow-headers", "authorization, content-type")
            if self.browser_origin != "*":
                self.send_header("vary", "Origin")

    def _request_id(self) -> str:
        request_id = getattr(self, "_crupier_request_id", None)
        if request_id is None:
            request_id = f"req_{uuid4().hex[:24]}"
            setattr(self, "_crupier_request_id", request_id)
        return request_id

    def log_message(self, format: str, *args: Any) -> None:
        return


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


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _error_contract(exc: Exception) -> tuple[HTTPStatus, str, str]:
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
    return {
        "error": {
            "message": _sanitize_error_message(message),
            "type": error_type,
            "param": None,
            "code": code,
        }
    }


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
