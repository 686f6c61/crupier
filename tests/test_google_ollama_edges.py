import io
import json
import sys
import urllib.error
from types import SimpleNamespace

import pytest

import crupier.adapters.google as google_module
import crupier.adapters.ollama as ollama_module
from crupier.adapters.google import GoogleAdapter
from crupier.adapters.ollama import OllamaAdapter
from crupier.config import ProviderSettings
from crupier.errors import (
    CrupierModelUnsupportedError,
    CrupierProviderAuthError,
    CrupierProviderRateLimitError,
    CrupierProviderUnavailableError,
)
from crupier.models import RequestEnvelope


def _http_error(code, body=b"error"):
    return urllib.error.HTTPError("https://example.test", code, "failure", {}, io.BytesIO(body))


def test_google_supports_only_native_images():
    assert GoogleAdapter.supports_file_kind(model="any", kind="image") is True
    assert GoogleAdapter.supports_file_kind(model="any", kind="pdf") is False


def test_google_generate_builds_prompt_and_full_generation_config():
    class Models:
        def __init__(self):
            self.payload = None

        def generate_content(self, **payload):
            self.payload = payload
            return SimpleNamespace(text="ok", usage_metadata=SimpleNamespace(total_tokens=3))

    models = Models()
    adapter = GoogleAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(models=models)
    schema = {"type": "object"}

    result = adapter.generate(
        model="gemini-test",
        prompt="",
        request=RequestEnvelope(
            task="Return data",
            input="record",
            response_schema=schema,
            constraints={
                "timeout_seconds": 1.25,
                "temperature": 0.3,
                "max_output_tokens": 500,
                "thinking_level": "high",
            },
        ),
    )

    assert models.payload["contents"] == "Task:\nReturn data\n\nInput:\nrecord"
    assert models.payload["config"] == {
        "http_options": {"timeout": 1250},
        "temperature": 0.3,
        "max_output_tokens": 500,
        "thinking_config": {"thinking_level": "high"},
        "response_mime_type": "application/json",
        "response_json_schema": schema,
    }
    assert result.usage == {"total_tokens": 3}


def test_google_generate_and_catalog_map_errors_and_skip_missing_models():
    class PermissionDenied(Exception):
        pass

    adapter = GoogleAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **payload: (_ for _ in ()).throw(PermissionDenied("forbidden")))
    )
    with pytest.raises(CrupierProviderAuthError, match="forbidden"):
        adapter.generate(model="gemini-test", prompt="x", request=RequestEnvelope(task="x"))

    adapter._client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: [
                SimpleNamespace(name="models/z", displayName="Z"),
                {"id": "a", "displayName": "A"},
                {"display_name": "missing"},
            ]
        )
    )
    models = adapter.list_models()
    assert [item.id for item in models] == ["a", "z"]
    assert [item.name for item in models] == ["A", "Z"]

    class ResourceExhausted(Exception):
        pass

    adapter._client = SimpleNamespace(
        models=SimpleNamespace(list=lambda: (_ for _ in ()).throw(ResourceExhausted("quota")))
    )
    with pytest.raises(CrupierProviderRateLimitError, match="quota"):
        adapter.list_models()


def test_google_embeddings_validate_dimensions_parse_shapes_and_map_errors():
    class Models:
        def __init__(self):
            self.payload = None

        def embed_content(self, **payload):
            self.payload = payload
            return SimpleNamespace(
                embedding=SimpleNamespace(values=[1, 2]),
                usage_metadata={"total_tokens": 4},
            )

    models = Models()
    adapter = GoogleAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(models=models)

    result = adapter.embed(model="embed-test", input="x", dimensions=2)
    assert models.payload["config"] == {"output_dimensionality": 2}
    assert result.embeddings == [[1.0, 2.0]]
    assert result.usage == {"total_tokens": 4}

    with pytest.raises(CrupierProviderUnavailableError, match="must be positive"):
        adapter.embed(model="embed-test", input="x", dimensions=-1)

    adapter._client = SimpleNamespace(
        models=SimpleNamespace(embed_content=lambda **payload: (_ for _ in ()).throw(Exception("offline")))
    )
    with pytest.raises(CrupierProviderUnavailableError, match="Google request failed: offline"):
        adapter.embed(model="embed-test", input="x")


def test_google_probe_failures_errors_stream_limit_and_unknown_probe():
    class Models:
        def __init__(self):
            self.calls = []

        def generate_content(self, **payload):
            self.calls.append(payload)
            if payload.get("config") and isinstance(payload["config"], dict):
                return {"text": "not-json"}
            return {"candidates": []}

        def generate_content_stream(self, **payload):
            return [{"text": "stream-ok"}, *({} for _ in range(25))]

    models = Models()
    adapter = GoogleAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(models=models)
    request = RequestEnvelope(task="probe")

    structured = adapter.probe_capability(model="gemini-test", probe="structured_output", request=request)
    tool = adapter.probe_capability(model="gemini-test", probe="tool_call", request=request)
    stream = adapter.probe_capability(model="gemini-test", probe="streaming", request=request)

    assert structured.metadata["probe_status"] == "failed"
    assert tool.metadata["probe_status"] == "failed"
    assert stream.metadata["event_count"] == 20
    with pytest.raises(NotImplementedError, match="no native probe"):
        adapter.probe_capability(model="gemini-test", probe="vision", request=request)

    class BrokenModels:
        def generate_content(self, **payload):
            raise Exception("probe failed")

        def generate_content_stream(self, **payload):
            raise Exception("stream failed")

    adapter._client = SimpleNamespace(models=BrokenModels())
    for probe in ["structured_output", "tool_call", "streaming"]:
        with pytest.raises(CrupierProviderUnavailableError):
            adapter.probe_capability(model="gemini-test", probe=probe, request=request)


def test_google_tool_probe_exposes_callable_contract(monkeypatch):
    calls = []

    def plain_config(**kwargs):
        return kwargs

    class Models:
        def generate_content(self, **payload):
            tool = payload["config"]["tools"][0]
            calls.append(tool(ok=True, probe="crupier"))
            return {"candidates": []}

    monkeypatch.setattr(google_module, "_tool_probe_config", plain_config)
    adapter = GoogleAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(models=Models())

    adapter.probe_capability(model="gemini-test", probe="tool_call", request=RequestEnvelope(task="x"))

    assert calls == [{"ok": True, "probe": "crupier"}]


def test_google_build_client_reports_missing_dependency(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "google.genai", None)
    with pytest.raises(CrupierProviderUnavailableError, match="optional dependency"):
        GoogleAdapter(ProviderSettings(enabled=True))._build_client()


def test_google_key_presence_and_labels_cover_custom_and_default_envs(monkeypatch):
    for key in ["GOOGLE_API_KEY", "GEMINI_API_KEY", "CUSTOM_GOOGLE_KEY"]:
        monkeypatch.delenv(key, raising=False)

    assert google_module.google_env_present(None) is False
    assert google_module.google_env_label(None) == "GOOGLE_API_KEY/GEMINI_API_KEY"
    defaults = ProviderSettings(enabled=True)
    assert google_module.google_env_present(defaults) is False
    assert google_module.google_env_label(defaults) == "GOOGLE_API_KEY/GEMINI_API_KEY"
    alias = ProviderSettings(enabled=True, env_key="GOOGLE_API_KEY")
    monkeypatch.setenv("GEMINI_API_KEY", "alias")
    assert google_module.google_env_present(alias) is True
    monkeypatch.delenv("GEMINI_API_KEY")

    custom = ProviderSettings(enabled=True, env_key="CUSTOM_GOOGLE_KEY")
    monkeypatch.setenv("CUSTOM_GOOGLE_KEY", "custom")
    assert google_module.google_api_key(custom) == "custom"
    assert google_module.google_env_present(custom) is True
    assert google_module.google_env_label(custom) == "CUSTOM_GOOGLE_KEY"


@pytest.mark.parametrize(
    ("constraints", "mode", "expected"),
    [
        ({"thinking_config": {"include_thoughts": True}}, None, {"include_thoughts": True}),
        ({"thinking_budget": 42}, None, {"thinking_budget": 42}),
        ({"disable_thinking": True}, None, {"thinking_budget": 0}),
        ({}, "fast", {"thinking_level": "minimal"}),
        ({}, None, {}),
    ],
)
def test_google_thinking_config_precedence(constraints, mode, expected):
    assert google_module._google_thinking_config(RequestEnvelope(task="x", mode=mode, constraints=constraints)) == expected


def test_google_helpers_cover_sdk_fallbacks_text_usage_embeddings_and_tools(monkeypatch):
    assert google_module.extract_google_text(SimpleNamespace(text="direct")) == "direct"
    assert google_module.extract_google_text({"text": "dict"}) == "dict"
    assert google_module.extract_google_text({"parts": [{"text": "one"}, SimpleNamespace(text=" two")]}) == "one two"
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="nested")]))
    assert google_module.extract_google_text({"candidates": [candidate]}) == "nested"
    assert google_module._google_usage({"usageMetadata": {"totalTokenCount": 5}}) == {"totalTokenCount": 5}

    assert google_module._google_embeddings({}) == []
    assert google_module._google_embeddings({"embeddings": []}) == []
    assert google_module._google_embeddings({"embeddings": [[1, 2], {"values": [3, 4]}, {"missing": 1}]}) == [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    assert google_module._looks_like_vector([]) is False
    assert google_module._looks_like_vector([[1]]) is False

    assert google_module._google_has_tool_call(
        {"parts": [{"functionCall": {"name": "target"}}]}, "target"
    ) is True
    assert google_module._google_has_tool_call(
        {"parts": [SimpleNamespace(function_call=SimpleNamespace(name="target"))]}, "target"
    ) is True
    assert google_module._google_has_tool_call({"parts": [{"text": "none"}]}, "target") is False

    assert google_module._json_probe_ok('prefix {"ok":true,"probe":"crupier"} suffix') is True
    assert google_module._json_probe_ok("no object") is False
    assert google_module._json_probe_ok("prefix {invalid} suffix") is False

    monkeypatch.setitem(sys.modules, "google.genai", None)
    monkeypatch.setitem(sys.modules, "google.genai.types", None)
    assert google_module._google_text_part("hello") == {"text": "hello"}
    image = {"base64": "aW1hZ2U=", "mime_type": "image/png"}
    assert google_module._google_image_part(image) == {
        "inline_data": {"mime_type": "image/png", "data": "aW1hZ2U="}
    }
    fallback = google_module._tool_probe_config(tools=["tool"], max_output_tokens=5, temperature=0)
    assert fallback["automatic_function_calling"] == {"disable": True}


@pytest.mark.parametrize(
    ("error", "error_type"),
    [
        (type("StatusError", (Exception,), {"status_code": 401})("bad"), CrupierProviderAuthError),
        (type("StatusError", (Exception,), {"code": 429})("slow"), CrupierProviderRateLimitError),
        (Exception("offline"), CrupierProviderUnavailableError),
    ],
)
def test_google_error_mapping_by_status(error, error_type):
    with pytest.raises(error_type):
        GoogleAdapter(ProviderSettings(enabled=True))._raise_mapped_error(error)


def test_ollama_supports_only_images_and_builds_local_payload(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"message": {"content": "ok"}, "prompt_eval_count": 2}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: requests.append(request) or Response())
    monkeypatch.setenv("OLLAMA_API_KEY", "optional-local-key")
    adapter = OllamaAdapter(ProviderSettings(enabled=True, host="http://127.0.0.1:11434/"))
    result = adapter.generate(
        model="local",
        prompt="",
        request=RequestEnvelope(task="hello", constraints={"temperature": 0.4}),
    )

    payload = json.loads(requests[0].data)
    assert OllamaAdapter.supports_file_kind(model="any", kind="image") is True
    assert OllamaAdapter.supports_file_kind(model="any", kind="audio") is False
    assert payload["messages"][0]["content"] == "Task:\nhello"
    assert payload["options"] == {"temperature": 0.4}
    assert result.usage == {"prompt_eval_count": 2}
    assert result.metadata["host"] == "http://127.0.0.1:11434"
    assert requests[0].headers["Authorization"] == "Bearer optional-local-key"


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("generate", "Ollama request failed: offline"),
        ("list_models", "Ollama model listing failed: offline"),
        ("embed", "Ollama embedding request failed: offline"),
    ],
)
def test_ollama_public_calls_map_network_errors(monkeypatch, method, message):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )
    adapter = OllamaAdapter(ProviderSettings(enabled=True, host="http://localhost:11434"))

    with pytest.raises(CrupierProviderUnavailableError, match=message):
        if method == "generate":
            adapter.generate(model="x", prompt="x", request=RequestEnvelope(task="x"))
        elif method == "list_models":
            adapter.list_models()
        else:
            adapter.embed(model="x", input="x")


def test_ollama_catalog_skips_missing_ids_and_embedding_constraints(monkeypatch):
    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.body).encode()

    responses = iter(
        [
            Response({"models": [{"name": "z"}, {"size": 1}, {"model": "a"}]}),
            Response({"embedding": [1, 2], "eval_count": 3, "load_duration": 4}),
            Response({}),
        ]
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: next(responses))
    adapter = OllamaAdapter(ProviderSettings(enabled=True, host="http://localhost:11434"))

    assert [item.id for item in adapter.list_models()] == ["a", "z"]
    embedded = adapter.embed(model="embed", input="x")
    assert embedded.embeddings == [[1.0, 2.0]]
    assert embedded.usage == {"eval_count": 3, "load_duration": 4}
    assert adapter.embed(model="embed", input="x").metadata.get("embedding_dimensions") is None
    with pytest.raises(CrupierModelUnsupportedError, match="do not expose"):
        adapter.embed(model="embed", input="x", dimensions=2)


def test_ollama_probe_failures_stream_limit_and_unknown_probe(monkeypatch):
    adapter = OllamaAdapter(ProviderSettings(enabled=True, host="http://localhost:11434"))

    def fake_json(payload, *, timeout):
        if payload.get("tools"):
            return {"message": {"tool_calls": [{"function": {"name": "other"}}]}}
        return {"message": {"content": "not-json"}}

    def fake_stream(payload, *, timeout):
        yield {"message": {"content": "stream-ok"}}
        for _ in range(25):
            yield {}

    monkeypatch.setattr(adapter, "_chat_json", fake_json)
    monkeypatch.setattr(adapter, "_chat_stream", fake_stream)
    request = RequestEnvelope(task="probe")

    assert adapter.probe_capability(model="x", probe="structured_output", request=request).metadata[
        "probe_status"
    ] == "failed"
    assert adapter.probe_capability(model="x", probe="tool_call", request=request).metadata["probe_status"] == "failed"
    assert adapter.probe_capability(model="x", probe="streaming", request=request).metadata["event_count"] == 20
    with pytest.raises(NotImplementedError, match="no native probe"):
        adapter.probe_capability(model="x", probe="vision", request=request)


def test_ollama_chat_helpers_parse_json_lines_and_map_network_errors(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"message":{"content":"ok"}}'

        def __iter__(self):
            return iter([b"\n", b'{"message":{"content":"one"}}\n'])

    adapter = OllamaAdapter(ProviderSettings(enabled=True, host="http://localhost:11434"))
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    assert adapter._chat_json({"model": "x"}, timeout=1)["message"]["content"] == "ok"
    assert list(adapter._chat_stream({"model": "x"}, timeout=1)) == [{"message": {"content": "one"}}]

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )
    with pytest.raises(CrupierProviderUnavailableError, match="offline"):
        adapter._chat_json({"model": "x"}, timeout=1)
    with pytest.raises(CrupierProviderUnavailableError, match="offline"):
        list(adapter._chat_stream({"model": "x"}, timeout=1))


def test_ollama_headers_urls_and_cloud_auth(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    local = OllamaAdapter(ProviderSettings(enabled=True, host="http://localhost:11434"))
    assert local._headers() == {"Content-Type": "application/json"}
    assert local._chat_url().endswith("/api/chat")
    assert local._tags_url().endswith("/api/tags")
    assert local._embed_url().endswith("/api/embed")
    assert local._requires_cloud_auth() is False

    api_host = OllamaAdapter(ProviderSettings(enabled=True, host="https://custom.example/api"))
    assert api_host._chat_url() == "https://custom.example/api/chat"
    assert api_host._tags_url() == "https://custom.example/api/tags"
    assert api_host._embed_url() == "https://custom.example/api/embed"

    cloud = OllamaAdapter(ProviderSettings(enabled=True, host="https://ollama.com/api"))
    with pytest.raises(CrupierProviderAuthError, match="requires OLLAMA_API_KEY"):
        cloud._headers()
    with pytest.raises(CrupierProviderAuthError, match="requires OLLAMA_API_KEY"):
        cloud.list_models()
    monkeypatch.setenv("OLLAMA_API_KEY", "test")
    assert cloud._headers()["Authorization"] == "Bearer test"


@pytest.mark.parametrize("operation", ["generate", "list", "embed", "chat", "stream"])
def test_ollama_http_errors_are_mapped_at_each_transport_boundary(monkeypatch, operation):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(_http_error(500, b"server failed")),
    )
    adapter = OllamaAdapter(ProviderSettings(enabled=True, host="http://localhost:11434"))

    with pytest.raises(CrupierProviderUnavailableError, match="Ollama HTTP 500"):
        if operation == "generate":
            adapter.generate(model="x", prompt="x", request=RequestEnvelope(task="x"))
        elif operation == "list":
            adapter.list_models()
        elif operation == "embed":
            adapter.embed(model="x", input="x")
        elif operation == "chat":
            adapter._chat_json({"model": "x"}, timeout=1)
        else:
            list(adapter._chat_stream({"model": "x"}, timeout=1))


@pytest.mark.parametrize(
    ("code", "error_type", "retryable"),
    [
        (401, CrupierProviderAuthError, None),
        (429, CrupierProviderRateLimitError, None),
        (503, CrupierProviderUnavailableError, True),
        (400, CrupierProviderUnavailableError, False),
    ],
)
def test_ollama_http_error_mapping(code, error_type, retryable):
    adapter = OllamaAdapter(ProviderSettings(enabled=True))
    with pytest.raises(error_type) as exc_info:
        adapter._raise_http_error(_http_error(code, b"provider body"))
    if retryable is not None:
        assert exc_info.value.retryable is retryable


def test_ollama_probe_and_embedding_helpers_cover_invalid_shapes():
    assert ollama_module._json_probe_ok('prefix {"ok":true,"probe":"crupier"} suffix') is True
    assert ollama_module._json_probe_ok("no object") is False
    assert ollama_module._json_probe_ok("prefix {invalid} suffix") is False
    assert ollama_module._ollama_has_tool_call({"message": {"tool_calls": [None]}}, "target") is False
    assert ollama_module._ollama_has_tool_call({}, "target") is False
    assert ollama_module._ollama_usage({}) == {}
    assert ollama_module._ollama_embeddings({"embeddings": [None, [1, 2]]}) == [[1.0, 2.0]]
