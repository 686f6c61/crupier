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
from crupier.server import (
    _audio_content_type,
    _coerce_form_value,
    _error_contract,
    _is_loopback_bind_host,
    _openai_error_payload,
    _OpenAICompatibleHandler,
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
        bearer_token="test-server-token",
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
        headers={"content-type": "application/json", "authorization": "Bearer test-server-token"},
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
            headers={"content-type": content_type, "authorization": "Bearer test-server-token"},
        )
        assert status == 400
        assert "file" in json.loads(raw)["error"]["message"]

        status, _, raw = request(
            address,
            "POST",
            "/v1/images/edits",
            body=body,
            headers={"content-type": content_type, "authorization": "Bearer test-server-token"},
        )
        assert status == 400
        assert "prompt" in json.loads(raw)["error"]["message"]

        status, _, raw = request(
            address,
            "POST",
            "/v1/audio/transcriptions",
            body=b"{}",
            headers={"content-type": "application/json", "authorization": "Bearer test-server-token"},
        )
        assert status == 415
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
                headers={"content-type": "application/json", "authorization": "Bearer test-server-token"},
            )
            assert status == 400
            assert json.loads(raw)["error"]["code"] == "invalid_request"

        for length in ("not-an-int", "-1"):
            connection = http.client.HTTPConnection(address[0], address[1], timeout=5)
            connection.putrequest("POST", "/v1/responses")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Authorization", "Bearer test-server-token")
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
        headers={"content-type": "application/json", "authorization": "Bearer test-server-token"},
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
        (
            CrupierApprovalRequired("x", approval_id="apr_1", trace_id="trc_1"),
            HTTPStatus.CONFLICT,
            "invalid_request_error",
            "approval_required",
        ),
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

    approval = _openai_error_payload(
        CrupierApprovalRequired(
            "approval needed",
            approval_id="apr_test",
            trace_id="trc_test",
        )
    )
    assert approval["error"]["crupier"] == {
        "approval_id": "apr_test",
        "trace_id": "trc_test",
    }


def test_control_headers_reach_chat_and_response_compat_calls(tmp_path):
    captured = []

    def callback(server, address):
        server.RequestHandlerClass.client.responses.create = lambda **kwargs: (
            captured.append(kwargs) or {"ok": True}
        )
        status, _, raw = request(
            address,
            "POST",
            "/v1/responses",
            body=json.dumps({"input": "hello"}).encode(),
            headers={
                "content-type": "application/json",
                "authorization": "Bearer test-server-token",
                "x-crupier-experiment": "model-rollout",
                "x-crupier-approval": "apr_test.secret-value-long-enough",
            },
        )
        assert status == HTTPStatus.OK
        assert json.loads(raw) == {"ok": True}

    run_server(tmp_path, callback)

    assert captured == [
        {
            "input": "hello",
            "experiment": "model-rollout",
            "approval_token": "apr_test.secret-value-long-enough",
        }
    ]


def test_http_body_cannot_escalate_dry_run_to_live_execution(tmp_path):
    crupier = make_crupier(tmp_path)
    calls = []
    original_deal = crupier.deal

    def recording_deal(*args, **kwargs):
        calls.append(kwargs)
        return original_deal(*args, **kwargs)

    crupier.deal = recording_deal
    server = build_openai_compatible_server(
        crupier=crupier,
        host="127.0.0.1",
        port=0,
        dry_run=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        address = server.server_address
        status, _, raw = request(
            address,
            "POST",
            "/v1/responses",
            body=json.dumps({"input": "no ejecutar", "dry_run": False}).encode(),
            headers={"content-type": "application/json"},
        )
        assert status == HTTPStatus.BAD_REQUEST
        assert json.loads(raw)["error"]["code"] == "invalid_request"
        assert calls == []

        status, _, raw = request(
            address,
            "POST",
            "/v1/responses",
            body=json.dumps({"input": "simular"}).encode(),
            headers={"content-type": "application/json"},
        )
        assert status == HTTPStatus.OK, raw
        assert calls[-1]["dry_run"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_body_cannot_set_storage_constraints(tmp_path):
    def callback(server, address):
        status, _, data = json_request(
            address,
            "POST",
            "/v1/responses",
            {
                "input": "dato privado",
                "constraints": {
                    "store_trace": True,
                    "store_prompt": True,
                    "store_response": True,
                },
            },
        )
        assert status == HTTPStatus.BAD_REQUEST
        assert data["error"]["code"] == "invalid_request"
        traces_dir = server.RequestHandlerClass.crupier_client.config.traces_dir
        assert not traces_dir.exists() or list(traces_dir.iterdir()) == []

    run_server(tmp_path, callback)


def test_http_body_cannot_raise_cost_ceiling_or_extra_body(tmp_path):
    crupier = make_crupier(tmp_path)
    crupier.config.routing.max_cost_per_request_usd = 0.01
    crupier.config.providers["openai"].options["extra_body"] = {"operator": "fixed"}
    server = build_openai_compatible_server(
        crupier=crupier,
        host="127.0.0.1",
        port=0,
        bearer_token="test-server-token",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, data = json_request(
            server.server_address,
            "POST",
            "/v1/responses",
            {
                "input": "intento",
                "constraints": {
                    "max_cost_usd": 1000,
                    "extra_body": {"operator": "attacker"},
                },
            },
        )
        assert status == HTTPStatus.BAD_REQUEST
        assert data["error"]["code"] == "invalid_request"
        assert crupier.config.routing.max_cost_per_request_usd == 0.01
        assert crupier.config.providers["openai"].options["extra_body"] == {"operator": "fixed"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_allowlisted_openai_parameters_still_work(tmp_path):
    captured = []

    def callback(server, address):
        server.RequestHandlerClass.client.responses.create = lambda **kwargs: (
            captured.append(("responses", kwargs)) or {"ok": True}
        )
        server.RequestHandlerClass.client.chat.completions.create = lambda **kwargs: (
            captured.append(("chat", kwargs)) or {"ok": True}
        )
        response_format = {"type": "json_object"}
        status, _, _ = json_request(
            address,
            "POST",
            "/v1/responses",
            {
                "model": "gpt-5.4-mini",
                "input": "hola",
                "stream": False,
                "response_format": response_format,
                "temperature": 0.2,
            },
        )
        assert status == HTTPStatus.OK
        status, _, _ = json_request(
            address,
            "POST",
            "/v1/chat/completions",
            {
                "model": "gpt-5.4-mini",
                "messages": [{"role": "user", "content": "hola"}],
                "stream": False,
                "response_format": response_format,
                "temperature": 0.2,
            },
        )
        assert status == HTTPStatus.OK

    run_server(tmp_path, callback)

    assert captured[0][1]["input"] == "hola"
    assert captured[0][1]["response_format"] == {"type": "json_object"}
    assert captured[0][1]["temperature"] == 0.2
    assert captured[1][1]["messages"] == [{"role": "user", "content": "hola"}]
    assert captured[1][1]["response_format"] == {"type": "json_object"}
    assert captured[1][1]["temperature"] == 0.2


def test_request_id_is_stable_per_handler_instance():
    handler = object.__new__(_OpenAICompatibleHandler)

    first = handler._request_id()

    assert first.startswith("req_")
    assert handler._request_id() == first
