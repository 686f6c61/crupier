import http.client
import json
import threading
from http import HTTPStatus
from uuid import uuid4

import pytest

import crupier.server as server_module
from crupier import Crupier
from crupier.config import CrupierConfig
from crupier.errors import (
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
from crupier.server import (
    _OpenAICompatibleHandler,
    _audio_content_type,
    _coerce_form_value,
    _error_contract,
    _is_loopback_bind_host,
    _openai_error_payload,
    _plain,
    build_openai_compatible_server,
    serve_openai_compatible,
)


def make_crupier(tmp_path) -> Crupier:
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "server-edges"},
            "providers": {"openai": {"enabled": True, "env_key": "OPENAI_API_KEY"}},
            "models": {"allow": ["openai:gpt-5.4-mini"]},
            "routing": {"require_operational_providers": False},
        }
    )
    config.root = tmp_path
    return Crupier(config, adapters={})


def run_server(tmp_path, callback, *, cors_origin=None):
    server = build_openai_compatible_server(
        crupier=make_crupier(tmp_path),
        host="127.0.0.1",
        port=0,
        cors_origin=cors_origin,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        callback(server, server.server_address)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(address, method, path, *, body=None, headers=None):
    connection = http.client.HTTPConnection(address[0], address[1], timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    raw = response.read()
    result = response.status, dict(response.getheaders()), raw
    connection.close()
    return result


def json_request(address, method, path, payload):
    status, headers, raw = request(
        address,
        method,
        path,
        body=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    return status, headers, json.loads(raw)


def multipart_body(*, fields=(), files=()):
    boundary = f"crupier-{uuid4().hex}"
    chunks = []
    for name, value in fields:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode(),
                b"\r\n",
            ]
        )
    for name, filename, content_type, content in files:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode(),
                content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def test_options_unknown_get_and_unknown_post(tmp_path):
    def check(server, address):
        status, _, raw = request(address, "OPTIONS", "/v1/responses")
        assert status == 204
        assert raw == b""

        status, _, data = json_request(address, "GET", "/missing", {})
        assert status == 404
        assert data["error"]["type"] == "invalid_request_error"

        status, _, data = json_request(address, "POST", "/missing", {})
        assert status == 404
        assert data["error"]["code"] is None

    run_server(tmp_path, check)


@pytest.mark.parametrize(
    ("path", "payload", "message"),
    [
        ("/v1/chat/completions", {"messages": "bad"}, "must be a list"),
        ("/v1/embeddings", {"input": "x"}, "model"),
        ("/v1/embeddings", {"model": "x"}, "input"),
        ("/v1/rerank", {"query": "q"}, "documents"),
        ("/v1/images/generations", {}, "prompt"),
        ("/v1/audio/speech", {"input": "x"}, "voice"),
    ],
)
def test_required_json_fields_return_400(tmp_path, path, payload, message):
    def check(server, address):
        status, _, data = json_request(address, "POST", path, payload)
        assert status == 400
        assert message in data["error"]["message"]

    run_server(tmp_path, check)


def test_required_multipart_fields_and_content_type(tmp_path):
    def check(server, address):
        body, content_type = multipart_body(fields=[("model", "x")])
        status, _, raw = request(
            address,
            "POST",
            "/v1/audio/transcriptions",
            body=body,
            headers={"content-type": content_type},
        )
        assert status == 400
        assert "file" in json.loads(raw)["error"]["message"]

        status, _, raw = request(
            address,
            "POST",
            "/v1/images/edits",
            body=body,
            headers={"content-type": content_type},
        )
        assert status == 400
        assert "prompt" in json.loads(raw)["error"]["message"]

        status, _, raw = request(
            address,
            "POST",
            "/v1/audio/transcriptions",
            body=b"{}",
            headers={"content-type": "application/json"},
        )
        assert status == 400
        assert "multipart/form-data" in json.loads(raw)["error"]["message"]

    run_server(tmp_path, check)


def test_invalid_json_shapes_and_content_lengths(tmp_path):
    def check(server, address):
        for body in (b"{bad", b"[]"):
            status, _, raw = request(
                address,
                "POST",
                "/v1/responses",
                body=body,
                headers={"content-type": "application/json"},
            )
            assert status == 400
            assert json.loads(raw)["error"]["code"] == "invalid_request"

        for length in ("not-an-int", "-1"):
            connection = http.client.HTTPConnection(address[0], address[1], timeout=5)
            connection.putrequest("POST", "/v1/responses")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", length)
            connection.endheaders()
            response = connection.getresponse()
            data = json.loads(response.read())
            connection.close()
            assert response.status == 400
            assert data["error"]["code"] == "invalid_request"

    run_server(tmp_path, check)


def test_type_error_and_unexpected_errors_are_sanitized_at_http_boundary(tmp_path):
    def check(server, address):
        def type_error(**kwargs):
            raise TypeError("bad argument")

        server.RequestHandlerClass.client.responses.create = type_error
        status, _, data = json_request(address, "POST", "/v1/responses", {"input": "x"})
        assert status == 400
        assert data["error"]["message"] == "bad argument"

        def unexpected(**kwargs):
            raise RuntimeError("do not expose sk-supersecret12345")

        server.RequestHandlerClass.client.responses.create = unexpected
        status, _, data = json_request(address, "POST", "/v1/responses", {"input": "x"})
        assert status == 500
        assert data["error"]["message"] == "Internal server error."

    run_server(tmp_path, check)


def test_sse_wraps_non_iterator_and_reports_late_stream_errors(tmp_path):
    def check(server, address):
        server.RequestHandlerClass.client.responses.create = lambda **kwargs: {"type": "custom", "ok": True}
        status, _, raw = json_request_raw(address, "/v1/responses", {"input": "x", "stream": True})
        assert status == 200
        assert b"event: custom" in raw
        assert b"data: [DONE]" in raw

        def broken_stream(**kwargs):
            def events():
                yield {"type": "first", "ok": True}
                raise RuntimeError("late failure")

            return events()

        server.RequestHandlerClass.client.responses.create = broken_stream
        status, _, raw = json_request_raw(address, "/v1/responses", {"input": "x", "stream": True})
        assert status == 200
        assert b"event: error" in raw
        assert b"internal_error" in raw

    run_server(tmp_path, check)


def json_request_raw(address, path, payload):
    return request(
        address,
        "POST",
        path,
        body=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )


def test_serve_always_closes_server(tmp_path, monkeypatch):
    calls = []

    class FakeServer:
        def serve_forever(self):
            calls.append("serve")

        def server_close(self):
            calls.append("close")

    monkeypatch.setattr(server_module, "build_openai_compatible_server", lambda **kwargs: FakeServer())

    serve_openai_compatible(crupier=make_crupier(tmp_path))

    assert calls == ["serve", "close"]


def test_server_helpers_cover_loopback_plain_values_and_form_coercion():
    class Serializable:
        def to_dict(self):
            return {"value": 1}

    assert _is_loopback_bind_host("localhost") is True
    assert _is_loopback_bind_host("not-an-address") is False
    assert _plain([Serializable(), {"nested": Serializable()}]) == [
        {"value": 1},
        {"nested": {"value": 1}},
    ]
    assert _coerce_form_value("n", "2") == 2
    assert _coerce_form_value("speed", "1.25") == 1.25
    assert _coerce_form_value("model", "x") == "x"
    with pytest.raises(ValueError, match="integer"):
        _coerce_form_value("top_n", "bad")
    with pytest.raises(ValueError, match="number"):
        _coerce_form_value("temperature", "bad")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("aac", "audio/aac"),
        ("flac", "audio/flac"),
        ("mp3", "audio/mpeg"),
        ("opus", "audio/ogg"),
        ("pcm", "audio/L16"),
        ("wav", "audio/wav"),
        ("unknown", "application/octet-stream"),
    ],
)
def test_audio_content_types(value, expected):
    assert _audio_content_type(value) == expected


@pytest.mark.parametrize(
    ("exc", "status", "error_type", "code"),
    [
        (CrupierProviderAuthError("x"), HTTPStatus.UNAUTHORIZED, "authentication_error", "invalid_api_key"),
        (CrupierProviderRateLimitError("x"), HTTPStatus.TOO_MANY_REQUESTS, "rate_limit_error", "rate_limit_exceeded"),
        (CrupierProviderUnavailableError("x"), HTTPStatus.SERVICE_UNAVAILABLE, "server_error", "provider_unavailable"),
        (CrupierModelUnsupportedError("x"), HTTPStatus.BAD_REQUEST, "invalid_request_error", "model_not_supported"),
        (CrupierBudgetExceededError("x"), HTTPStatus.BAD_REQUEST, "invalid_request_error", "budget_exceeded"),
        (CrupierExecutionLimitError("x"), HTTPStatus.REQUEST_TIMEOUT, "server_error", "execution_limit_exceeded"),
        (CrupierToolApprovalRequired("x"), HTTPStatus.BAD_REQUEST, "invalid_request_error", "tool_approval_required"),
        (CrupierStructuredOutputError("x"), HTTPStatus.BAD_REQUEST, "invalid_request_error", "structured_output_error"),
        (CrupierUpdateRequiresConfirmation("x"), HTTPStatus.CONFLICT, "invalid_request_error", "update_requires_confirmation"),
        (CrupierConfigError("x"), HTTPStatus.BAD_REQUEST, "invalid_request_error", "configuration_error"),
        (CrupierPolicyError("x"), HTTPStatus.BAD_REQUEST, "invalid_request_error", "policy_error"),
        (CrupierRouteValidationError("x"), HTTPStatus.BAD_REQUEST, "invalid_request_error", "route_validation_error"),
        (CrupierError("x"), HTTPStatus.BAD_REQUEST, "invalid_request_error", "CrupierError"),
    ],
)
def test_error_contracts(exc, status, error_type, code):
    assert _error_contract(exc) == (status, error_type, code)


def test_openai_error_payload_distinguishes_expected_and_unexpected_errors():
    expected = _openai_error_payload(CrupierPolicyError("Bearer abcdefghijklmnop"))
    unexpected = _openai_error_payload(RuntimeError("secret details"))

    assert expected["error"]["code"] == "policy_error"
    assert expected["error"]["message"] == "Bearer [redacted]"
    assert unexpected["error"]["code"] == "internal_error"
    assert unexpected["error"]["message"] == "Internal server error."


def test_request_id_is_stable_per_handler_instance():
    handler = object.__new__(_OpenAICompatibleHandler)

    first = handler._request_id()

    assert first.startswith("req_")
    assert handler._request_id() == first
