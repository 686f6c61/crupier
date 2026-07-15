import http.client
import json
import threading
from uuid import uuid4

from crupier import Crupier
from crupier.adapters import AdapterResponse, EmbeddingResponse, OperationResponse
from crupier.config import CrupierConfig
from crupier.errors import CrupierConfigError, CrupierProviderAuthError
from crupier.server import build_openai_compatible_server


class FakeAdapter:
    provider = "openai"

    def generate(self, *, model, prompt, request):
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
):
    server = build_openai_compatible_server(
        crupier=make_crupier(tmp_path, adapter=adapter, operation_adapter=operation_adapter),
        host="127.0.0.1",
        port=0,
        dry_run=dry_run,
        cors_origin=cors_origin,
        max_request_bytes=max_request_bytes,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        return fn(server.server_address)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request_json(address, method, path, payload=None):
    conn = http.client.HTTPConnection(address[0], address[1], timeout=5)
    body = json.dumps(payload or {})
    conn.request(method, path, body=body if method == "POST" else None, headers={"content-type": "application/json"})
    response = conn.getresponse()
    data = response.read().decode("utf-8")
    conn.close()
    return response.status, response.getheaders(), json.loads(data)


def request_text(address, method, path, payload=None):
    conn = http.client.HTTPConnection(address[0], address[1], timeout=5)
    body = json.dumps(payload or {})
    conn.request(method, path, body=body if method == "POST" else None, headers={"content-type": "application/json"})
    response = conn.getresponse()
    data = response.read().decode("utf-8")
    conn.close()
    return response.status, dict(response.getheaders()), data


def request_bytes(address, method, path, payload=None):
    conn = http.client.HTTPConnection(address[0], address[1], timeout=5)
    body = json.dumps(payload or {})
    conn.request(method, path, body=body, headers={"content-type": "application/json"})
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
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )
    response = conn.getresponse()
    data = response.read()
    headers = dict(response.getheaders())
    conn.close()
    return response.status, headers, data


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
