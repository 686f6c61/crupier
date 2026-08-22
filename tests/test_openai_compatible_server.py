import http.client
import json
import socket
import threading
from pathlib import Path
from uuid import uuid4

import pytest
from _synthetic_secrets import SYNTHETIC_GOOGLE_API_KEY

import crupier.server as server_module
from crupier import Crupier
from crupier.adapters import AdapterResponse, EmbeddingResponse, OperationResponse
from crupier.config import CrupierConfig
from crupier.errors import CrupierConfigError, CrupierProviderAuthError
from crupier.server import build_openai_compatible_server


class FakeAdapter:
    provider = "openai"

    def __init__(self):
        self.calls = []

    def generate(self, *, model, prompt, request):
        self.calls.append({"model": model, "prompt": prompt, "files": list(request.files)})
        return AdapterResponse(
            text=f"server {model}",
            usage={"input_tokens": 5, "output_tokens": 6},
            metadata={"provider": "openai", "model": model},
        )

    def embed(self, *, model, input, dimensions=None):
        vector = [1.0, 2.0, 3.0]
        if dimensions is not None:
            vector = vector[:dimensions]
        return EmbeddingResponse(
            embeddings=[vector],
            usage={"prompt_tokens": 2, "total_tokens": 2},
            metadata={"provider": "openai", "model": model},
        )


class AuthFailAdapter(FakeAdapter):
    def generate(self, *, model, prompt, request):
        secret = "s" + "k-test-secret-value"
        raise CrupierProviderAuthError(
            f"Provider rejected bearer token {secret}",
            provider="openai",
            env_key="OPENAI_API_KEY",
        )


class GoogleSecretFailAdapter(FakeAdapter):
    def generate(self, *, model, prompt, request):
        del model, prompt, request
        raise CrupierProviderAuthError(
            f"Provider rejected {SYNTHETIC_GOOGLE_API_KEY}",
            provider="google",
            env_key="GOOGLE_API_KEY",
        )


class FakeOperationAdapter:
    provider = "nan"

    def __init__(self):
        self.calls = []

    @staticmethod
    def supports_operation(*, operation, model):
        expected = {
            "reranker": "rerank",
            "transcription": "whisper",
            "tts": "kokoro",
            "image_generation": "flux-2-klein",
        }
        return expected.get(operation) == model

    def execute_operation(self, *, operation, model, request, payload):
        self.calls.append({"operation": operation, "model": model, "payload": payload})
        if operation == "reranker":
            output = [{"index": 1, "relevance_score": 0.99, "document": payload["documents"][1]}]
        elif operation == "transcription":
            output = {"text": "server transcript", "language": payload.get("language")}
        elif operation == "tts":
            output = b"ID3-server-audio"
        elif operation == "image_generation":
            output = [{"url": "https://example.test/generated.png"}]
        else:
            raise AssertionError(f"Unexpected operation {operation!r}")
        return OperationResponse(
            operation=operation,
            output=output,
            usage={"input_tokens": 2} if operation == "reranker" else {},
            metadata={"provider": "nan", "model": model},
        )


def make_crupier(tmp_path, *, adapter=None, operation_adapter=None):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "server", "default_profile": "agentic"},
            "providers": {
                "openai": {"enabled": True, "env_key": "OPENAI_API_KEY"},
                "nan": {"enabled": True, "env_key": "NAN_API_KEY"},
            },
            "models": {
                "allow": [
                    "openai:gpt-5.5",
                    "openai:gpt-5.4-mini",
                    "openai:text-embedding-3-small",
                    "nan:rerank",
                    "nan:kokoro",
                    "nan:whisper",
                    "nan:flux-2-klein",
                ]
            },
            "routing": {"default_strategy": "single", "require_operational_providers": False},
        }
    )
    config.root = tmp_path
    return Crupier(
        config,
        adapters={
            "openai": adapter or FakeAdapter(),
            "nan": operation_adapter or FakeOperationAdapter(),
        },
    )


def with_server(
    tmp_path,
    fn,
    *,
    dry_run=False,
    adapter=None,
    operation_adapter=None,
    cors_origin=None,
    max_request_bytes=10_000_000,
    file_root=None,
    bearer_token="test-server-token",
    **server_options,
):
    kwargs = {}
    if file_root is not None:
        kwargs["file_root"] = file_root
    server = build_openai_compatible_server(
        crupier=make_crupier(tmp_path, adapter=adapter, operation_adapter=operation_adapter),
        host="127.0.0.1",
        port=0,
        dry_run=dry_run,
        cors_origin=cors_origin,
        max_request_bytes=max_request_bytes,
        bearer_token=bearer_token,
        **server_options,
        **kwargs,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        return fn(server.server_address)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request_json(address, method, path, payload=None, *, token="test-server-token", headers=None):
    conn = http.client.HTTPConnection(address[0], address[1], timeout=5)
    body = json.dumps(payload or {})
    request_headers = {"content-type": "application/json", **(headers or {})}
    if token is not None:
        request_headers["authorization"] = f"Bearer {token}"
    conn.request(method, path, body=body if method == "POST" else None, headers=request_headers)
    response = conn.getresponse()
    data = response.read().decode("utf-8")
    conn.close()
    return response.status, response.getheaders(), json.loads(data)


def request_text(address, method, path, payload=None):
    conn = http.client.HTTPConnection(address[0], address[1], timeout=5)
    body = json.dumps(payload or {})
    conn.request(
        method,
        path,
        body=body if method == "POST" else None,
        headers={"content-type": "application/json", "authorization": "Bearer test-server-token"},
    )
    response = conn.getresponse()
    data = response.read().decode("utf-8")
    conn.close()
    return response.status, dict(response.getheaders()), data


def request_bytes(address, method, path, payload=None):
    conn = http.client.HTTPConnection(address[0], address[1], timeout=5)
    body = json.dumps(payload or {})
    conn.request(
        method,
        path,
        body=body,
        headers={"content-type": "application/json", "authorization": "Bearer test-server-token"},
    )
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, dict(response.getheaders()), data


def request_multipart(address, path, *, fields, files):
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
    body = b"".join(chunks)
    conn = http.client.HTTPConnection(address[0], address[1], timeout=5)
    conn.request(
        "POST",
        path,
        body=body,
        headers={
            "content-type": f"multipart/form-data; boundary={boundary}",
            "authorization": "Bearer test-server-token",
        },
    )
    response = conn.getresponse()
    data = response.read()
    headers = dict(response.getheaders())
    conn.close()
    return response.status, headers, data


def request_chunked(address, path, chunks):
    body = b"".join(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n" for chunk in chunks) + b"0\r\n\r\n"
    raw_request = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {address[0]}:{address[1]}\r\n"
        "Content-Type: application/json\r\n"
        "Authorization: Bearer test-server-token\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Connection: close\r\n\r\n"
    ).encode() + body
    connection = socket.create_connection(address, timeout=5)
    connection.sendall(raw_request)
    response = http.client.HTTPResponse(connection)
    response.begin()
    data = json.loads(response.read())
    status = response.status
    connection.close()
    return status, data


def request_without_content_length(address, path):
    conn = http.client.HTTPConnection(address[0], address[1], timeout=5)
    conn.putrequest("POST", path)
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Authorization", "Bearer test-server-token")
    conn.endheaders()
    response = conn.getresponse()
    data = json.loads(response.read())
    conn.close()
    return response.status, data


def sse_payloads(text):
    payloads = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


def test_health_endpoint(tmp_path):
    def run(address):
        status, headers, data = request_json(address, "GET", "/health")
        assert status == 200
        assert "access-control-allow-origin" not in dict(headers)
        assert data["ok"] is True
        assert data["compat"] == "openai"

    with_server(tmp_path, run)


def test_cors_headers_are_opt_in(tmp_path):
    def run(address):
        status, headers, data = request_json(address, "GET", "/health")
        assert status == 200
        assert dict(headers)["access-control-allow-origin"] == "http://localhost:3000"
        assert data["ok"] is True

    with_server(tmp_path, run, cors_origin="http://localhost:3000")


def test_remote_bind_requires_explicit_opt_in(tmp_path):
    try:
        build_openai_compatible_server(
            crupier=make_crupier(tmp_path),
            host="0.0.0.0",
            port=0,
        )
    except CrupierConfigError as exc:
        assert "--allow-remote" in str(exc)
    else:
        raise AssertionError("remote bind should require explicit opt-in")


def test_live_server_rejects_missing_or_invalid_bearer_before_body_processing(tmp_path):
    adapter = FakeAdapter()

    def run(address):
        for token in (None, "wrong-token"):
            status, _, data = request_json(
                address,
                "POST",
                "/v1/responses",
                {"model": "gpt-5.4-mini", "input": "must not execute"},
                token=token,
            )
            assert status == 401
            assert data["error"]["code"] == "invalid_api_key"
        assert adapter.calls == []

    with_server(tmp_path, run, adapter=adapter)


def test_server_emits_access_log_without_body_or_authorization(tmp_path):
    lines = []

    def run(address):
        status, _, _ = request_json(
            address,
            "POST",
            "/v1/responses",
            {"model": "gpt-5.4-mini", "input": "cuerpo-secreto"},
            token="token-super-secreto",
        )
        assert status == 401

    with_server(
        tmp_path,
        run,
        bearer_token="token-distinto",
        access_log_sink=lines.append,
    )

    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["path"] == "/v1/responses"
    assert record["status"] == 401
    assert record["request_id"].startswith("req_")
    assert record["response_bytes"] > 0
    assert "cuerpo-secreto" not in lines[0]
    assert "token-super-secreto" not in lines[0]


def test_models_endpoint_requires_authentication_when_token_is_configured(tmp_path):
    def run(address):
        status, _, data = request_json(address, "GET", "/v1/models", token=None)
        assert status == 401
        assert data["error"]["code"] == "invalid_api_key"

    with_server(tmp_path, run, dry_run=True)


def test_response_metadata_omits_policy_internals_by_default(tmp_path):
    def run(address):
        status, _, data = request_json(
            address,
            "POST",
            "/v1/responses",
            {"model": "gpt-5.4-mini", "input": "Hello"},
        )
        assert status == 200
        assert "policy_filters_applied" not in data["crupier"]["route"]
        trace = data["crupier"]["trace"] or {}
        assert "candidate_models" not in trace
        assert "excluded_models" not in trace
        assert "policy_filters" not in trace

    with_server(tmp_path, run, dry_run=True)


def test_chunked_request_body_is_rejected_instead_of_ignored(tmp_path):
    def run(address):
        status, data = request_chunked(
            address,
            "/v1/responses",
            [b'{"model":"gpt-5.4-mini",', b'"input":"must not execute"}'],
        )
        assert status == 400
        assert data["error"]["code"] == "invalid_request"

    with_server(tmp_path, run)


def test_missing_content_length_on_post_is_rejected(tmp_path):
    def run(address):
        status, data = request_without_content_length(address, "/v1/responses")
        assert status == 411
        assert data["error"]["code"] == "length_required"

    with_server(tmp_path, run)


def test_chunked_rejection_happens_before_provider_execution(tmp_path):
    adapter = FakeAdapter()

    def run(address):
        status, _ = request_chunked(address, "/v1/responses", [b'{"input":"ignored"}'])
        assert status == 400
        assert adapter.calls == []

    with_server(tmp_path, run, adapter=adapter)


def test_loopback_live_server_rejects_cross_origin_simple_request(tmp_path):
    def run(address):
        status, _, data = request_json(
            address,
            "POST",
            "/v1/responses",
            {"model": "gpt-5.4-mini", "input": "must not execute"},
            headers={"origin": "https://attacker.example", "content-type": "text/plain"},
        )
        assert status == 403
        assert data["error"]["code"] == "origin_not_allowed"

    with_server(tmp_path, run)


def test_remote_bind_requires_authentication_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "ThreadingHTTPServer", lambda address, handler: object())
    with pytest.raises(CrupierConfigError, match="authentication"):
        build_openai_compatible_server(
            crupier=make_crupier(tmp_path),
            host="0.0.0.0",
            port=0,
            allow_remote=True,
            dry_run=True,
        )


def test_remote_bind_checks_configured_authentication_even_in_dry_run(tmp_path, monkeypatch):
    captured = {}

    def fake_server(address, handler):
        captured["handler"] = handler
        return object()

    monkeypatch.setattr(server_module, "ThreadingHTTPServer", fake_server)
    build_openai_compatible_server(
        crupier=make_crupier(tmp_path),
        host="0.0.0.0",
        port=0,
        allow_remote=True,
        dry_run=True,
        bearer_token="remote-test-token",
    )

    assert captured["handler"].require_authentication is True


def test_responses_endpoint_returns_openai_like_json(tmp_path):
    def run(address):
        status, _, data = request_json(
            address,
            "POST",
            "/v1/responses",
            {"model": "gpt-5.4-mini", "input": "Hello", "instructions": "Reply."},
        )
        assert status == 200
        assert data["object"] == "response"
        assert data["output_text"] == "server gpt-5.5"
        assert data["usage"]["input_tokens"] == 5
        assert data["crupier"]["route"]["strategy"] == "single"

    with_server(tmp_path, run)


def test_chat_completions_endpoint_returns_choices(tmp_path):
    def run(address):
        status, _, data = request_json(
            address,
            "POST",
            "/v1/chat/completions",
            {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert status == 200
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "server gpt-5.5"

    with_server(tmp_path, run)


def test_chat_stream_endpoint_returns_sse(tmp_path):
    def run(address):
        status, headers, text = request_text(
            address,
            "POST",
            "/v1/chat/completions",
            {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
        )
        assert status == 200
        assert headers["content-type"] == "text/event-stream"
        payloads = sse_payloads(text)
        assert payloads[0]["object"] == "chat.completion.chunk"
        assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
        assert payloads[1]["choices"][0]["delta"]["content"] == "server gpt-5.5"
        assert payloads[2]["choices"][0]["finish_reason"] == "stop"
        assert "data: [DONE]" in text

    with_server(tmp_path, run)


def test_responses_stream_endpoint_returns_typed_sse(tmp_path):
    def run(address):
        status, headers, text = request_text(
            address,
            "POST",
            "/v1/responses",
            {"model": "gpt-5.4-mini", "input": "Hi", "stream": True, "include_obfuscation": False},
        )
        assert status == 200
        assert headers["content-type"] == "text/event-stream"
        assert "event: response.created" in text
        assert "event: response.output_text.delta" in text
        assert "event: response.completed" in text
        payloads = sse_payloads(text)
        assert [payload["type"] for payload in payloads] == [
            "response.created",
            "response.output_text.delta",
            "response.output_text.done",
            "response.completed",
        ]
        assert payloads[1]["delta"] == "server gpt-5.5"

    with_server(tmp_path, run)


def test_models_endpoint_lists_allowed_models(tmp_path):
    def run(address):
        status, _, data = request_json(address, "GET", "/v1/models?limit=20")
        ids = [item["id"] for item in data["data"]]
        assert status == 200
        assert ids == [
            "nan:flux-2-klein",
            "nan:kokoro",
            "nan:rerank",
            "nan:whisper",
            "openai:gpt-5.4-mini",
            "openai:gpt-5.5",
            "openai:text-embedding-3-small",
        ]

    with_server(tmp_path, run)


def test_missing_chat_messages_returns_openai_like_error(tmp_path):
    def run(address):
        status, headers, data = request_json(
            address,
            "POST",
            "/v1/chat/completions",
            {"model": "gpt-5.4-mini"},
        )
        assert status == 400
        assert dict(headers)["x-request-id"].startswith("req_")
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["code"] == "invalid_request"
        assert "messages" in data["error"]["message"]

    with_server(tmp_path, run)


def test_provider_auth_error_maps_to_401_and_redacts_secret(tmp_path):
    def run(address):
        status, _, data = request_json(
            address,
            "POST",
            "/v1/responses",
            {"model": "gpt-5.4-mini", "input": "Hello"},
        )
        assert status == 401
        assert data["error"]["type"] == "authentication_error"
        assert data["error"]["code"] == "invalid_api_key"
        assert "[redacted]" in data["error"]["message"]
        assert "test-secret-value" not in data["error"]["message"]

    with_server(tmp_path, run, adapter=AuthFailAdapter())


def test_server_error_message_uses_central_redactor(tmp_path):
    secret = SYNTHETIC_GOOGLE_API_KEY

    def run(address):
        status, _, data = request_json(
            address,
            "POST",
            "/v1/responses",
            {"model": "gpt-5.4-mini", "input": "Hello"},
        )
        assert status == 401
        assert secret not in data["error"]["message"]
        assert "[redacted]" in data["error"]["message"]

    with_server(tmp_path, run, adapter=GoogleSecretFailAdapter())


def test_embeddings_endpoint_returns_openai_like_json(tmp_path):
    def run(address):
        status, _, data = request_json(
            address,
            "POST",
            "/v1/embeddings",
            {"model": "text-embedding-3-small", "input": "hello", "dimensions": 2},
        )
        assert status == 200
        assert data["object"] == "list"
        assert data["data"][0]["object"] == "embedding"
        assert data["data"][0]["embedding"] == [1.0, 2.0]
        assert data["usage"]["prompt_tokens"] == 2

    with_server(tmp_path, run)


def test_rerank_endpoint_routes_to_specialized_model(tmp_path):
    def run(address):
        status, _, data = request_json(
            address,
            "POST",
            "/v1/rerank",
            {"model": "rerank", "query": "capital", "documents": ["Berlin", "Paris"], "top_n": 1},
        )
        assert status == 200
        assert data["model"] == "nan:rerank"
        assert data["results"][0]["document"] == "Paris"
        assert data["crupier"]["route"]["steps"][0]["model"] == "nan:rerank"

    with_server(tmp_path, run)


def test_image_generation_endpoint_returns_openai_like_json(tmp_path):
    def run(address):
        status, _, data = request_json(
            address,
            "POST",
            "/v1/images/generations",
            {"model": "flux-2-klein", "prompt": "A lighthouse", "size": "512x512"},
        )
        assert status == 200
        assert data["data"] == [{"url": "https://example.test/generated.png"}]
        assert data["model"] == "nan:flux-2-klein"
        assert data["crupier"]["operation"] == "image_generation"

    with_server(tmp_path, run)


def test_audio_speech_endpoint_returns_binary_content(tmp_path):
    def run(address):
        status, headers, data = request_bytes(
            address,
            "POST",
            "/v1/audio/speech",
            {"model": "kokoro", "input": "Hola", "voice": "ef_dora", "response_format": "mp3"},
        )
        assert status == 200
        assert headers["content-type"] == "audio/mpeg"
        assert data == b"ID3-server-audio"

    with_server(tmp_path, run)


def test_audio_transcription_endpoint_parses_multipart_upload(tmp_path):
    def run(address):
        status, headers, raw = request_multipart(
            address,
            "/v1/audio/transcriptions",
            fields=[("model", "whisper"), ("language", "es"), ("response_format", "verbose_json")],
            files=[("file", "sample.wav", "audio/wav", b"RIFF-audio")],
        )
        data = json.loads(raw)
        assert status == 200
        assert headers["content-type"] == "application/json"
        assert data["text"] == "server transcript"
        assert data["language"] == "es"
        assert data["model"] == "nan:whisper"

    with_server(tmp_path, run)


def test_image_edit_endpoint_preserves_repeated_multipart_images(tmp_path):
    adapter = FakeOperationAdapter()

    def run(address):
        status, _, raw = request_multipart(
            address,
            "/v1/images/edits",
            fields=[("model", "flux-2-klein"), ("prompt", "Merge references"), ("n", "1")],
            files=[
                ("image[]", "one.png", "image/png", b"one"),
                ("image[]", "two.png", "image/png", b"two"),
            ],
        )
        data = json.loads(raw)
        assert status == 200
        assert data["data"][0]["url"].endswith("generated.png")
        images = adapter.calls[0]["payload"]["images"]
        assert images == [
            ("one.png", b"one", "image/png"),
            ("two.png", b"two", "image/png"),
        ]

    with_server(tmp_path, run, operation_adapter=adapter)


def test_multipart_endpoint_rejects_body_above_server_limit(tmp_path):
    def run(address):
        status, _, raw = request_multipart(
            address,
            "/v1/audio/transcriptions",
            fields=[("model", "whisper")],
            files=[("file", "sample.wav", "audio/wav", b"x" * 512)],
        )
        data = json.loads(raw)
        assert status == 413
        assert data["error"]["code"] == "request_too_large"

    with_server(tmp_path, run, max_request_bytes=128)


def _track_secret_reads(monkeypatch, secret: Path) -> list[str]:
    reads: list[str] = []
    target = secret.resolve()
    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes

    def read_text(self, *args, **kwargs):
        if self.expanduser().resolve() == target:
            reads.append("text")
        return original_read_text(self, *args, **kwargs)

    def read_bytes(self, *args, **kwargs):
        if self.expanduser().resolve() == target:
            reads.append("bytes")
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    return reads


def test_http_responses_rejects_absolute_input_file_path(tmp_path, monkeypatch):
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"\x89PNG\r\n\x1a\nPII-TOKEN-SHOULD-NOT-LEAK")
    adapter = FakeAdapter()
    reads = _track_secret_reads(monkeypatch, secret)

    def run(address):
        status, _, data = request_json(
            address,
            "POST",
            "/v1/responses",
            {
                "model": "gpt-5.4-mini",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Summarize the file"},
                            {"type": "input_file", "filename": str(secret)},
                        ],
                    }
                ],
            },
        )
        assert status == 400
        assert data["error"]["type"] == "invalid_request_error"
        assert "local" in data["error"]["message"].lower()
        assert adapter.calls == []
        assert reads == []

    with_server(tmp_path, run, adapter=adapter)


def test_http_chat_rejects_parent_traversal_and_symlink_escape(tmp_path):
    allowed = tmp_path / "root"
    allowed.mkdir()
    (allowed / "inside.png").write_bytes(b"\x89PNG\r\n\x1a\ninside")
    secret = tmp_path / "outside.png"
    secret.write_bytes(b"\x89PNG\r\n\x1a\nOUTSIDE-SECRET")
    (allowed / "escape.png").symlink_to(secret)
    adapter = FakeAdapter()

    def run(address):
        for filename in ("../outside.png", str(allowed / "escape.png")):
            status, _, data = request_json(
                address,
                "POST",
                "/v1/chat/completions",
                {
                    "model": "gpt-5.4-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Read this"},
                                {"type": "file", "filename": filename},
                            ],
                        }
                    ],
                },
            )
            assert status == 400, filename
            assert data["error"]["type"] == "invalid_request_error"
            assert "local" in data["error"]["message"].lower()
        assert adapter.calls == []

    with_server(tmp_path, run, adapter=adapter, file_root=allowed)


def test_http_multipart_bytes_still_reach_transcription_and_image_edit(tmp_path):
    operation_adapter = FakeOperationAdapter()

    def run(address):
        transcribe_status, _, transcribe_raw = request_multipart(
            address,
            "/v1/audio/transcriptions",
            fields=[("model", "whisper"), ("language", "es")],
            files=[("file", "sample.wav", "audio/wav", b"RIFF-audio")],
        )
        transcribe = json.loads(transcribe_raw)
        assert transcribe_status == 200
        assert transcribe["text"] == "server transcript"

        edit_status, _, edit_raw = request_multipart(
            address,
            "/v1/images/edits",
            fields=[("model", "flux-2-klein"), ("prompt", "Merge references")],
            files=[
                ("image[]", "one.png", "image/png", b"one"),
                ("image[]", "two.png", "image/png", b"two"),
            ],
        )
        edit = json.loads(edit_raw)
        assert edit_status == 200
        images = operation_adapter.calls[1]["payload"]["images"]
        assert images == [
            ("one.png", b"one", "image/png"),
            ("two.png", b"two", "image/png"),
        ]
        assert edit["data"][0]["url"].endswith("generated.png")

    with_server(tmp_path, run, operation_adapter=operation_adapter)
