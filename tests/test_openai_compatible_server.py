import http.client
import json
import threading

from crupier import Crupier
from crupier.adapters import AdapterResponse, EmbeddingResponse
from crupier.config import CrupierConfig
from crupier.errors import CrupierProviderAuthError
from crupier.server import build_openai_compatible_server


class FakeAdapter:
    provider = "openai"

    def generate(self, *, model, prompt, request):
        return AdapterResponse(
            text=f"server {model}",
            usage={"input_tokens": 5, "output_tokens": 6},
            metadata={"provider": "openai", "model": model},
        )

    def embed(self, *, model, input):
        return EmbeddingResponse(
            embeddings=[[1.0, 2.0, 3.0]],
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


def make_crupier(tmp_path, *, adapter=None):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "server", "default_profile": "agentic"},
            "providers": {"openai": {"enabled": True, "env_key": "OPENAI_API_KEY"}},
            "models": {"allow": ["openai:gpt-5.5", "openai:gpt-5.4-mini"]},
            "routing": {"default_strategy": "single"},
        }
    )
    config.root = tmp_path
    return Crupier(config, adapters={"openai": adapter or FakeAdapter()})


def with_server(tmp_path, fn, *, dry_run=False, adapter=None):
    server = build_openai_compatible_server(
        crupier=make_crupier(tmp_path, adapter=adapter),
        host="127.0.0.1",
        port=0,
        dry_run=dry_run,
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


def sse_payloads(text):
    payloads = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


def test_health_endpoint(tmp_path):
    def run(address):
        status, _, data = request_json(address, "GET", "/health")
        assert status == 200
        assert data["ok"] is True
        assert data["compat"] == "openai"

    with_server(tmp_path, run)


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
        assert ids == ["openai:gpt-5.4-mini", "openai:gpt-5.5"]

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
