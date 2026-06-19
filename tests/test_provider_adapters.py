import json
from datetime import datetime

from crupier.adapters import ProviderModel
from crupier.adapters.anthropic import AnthropicAdapter
from crupier.adapters.factory import build_default_adapters
from crupier.adapters.google import GoogleAdapter, google_api_key
from crupier.adapters.ollama import OllamaAdapter
from crupier.adapters.openai import OpenAIAdapter
from crupier.config import CrupierConfig, ProviderSettings
from crupier.errors import CrupierProviderAuthError
from crupier.models import FileAsset, RequestEnvelope


class FakeOpenAIResponses:
    def __init__(self):
        self.payload = None

    def create(self, **payload):
        self.payload = payload

        class Response:
            output_text = "openai text"
            usage = {"input_tokens": 1, "output_tokens": 2}

        return Response()


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeOpenAIResponses()


def test_openai_adapter_uses_responses_create():
    adapter = OpenAIAdapter(ProviderSettings(enabled=True))
    client = FakeOpenAIClient()
    adapter._client = client

    response = adapter.generate(model="gpt-5.5", prompt="hello", request=RequestEnvelope(task="x"))

    assert response.text == "openai text"
    assert client.responses.payload["model"] == "gpt-5.5"
    assert client.responses.payload["input"] == "hello"


def test_openai_adapter_sends_native_image_content(tmp_path):
    image = tmp_path / "receipt.png"
    image.write_bytes(b"image-bytes")
    adapter = OpenAIAdapter(ProviderSettings(enabled=True))
    client = FakeOpenAIClient()
    adapter._client = client

    response = adapter.generate(
        model="gpt-4.1-mini",
        prompt="read image",
        request=RequestEnvelope(task="x", files=[FileAsset(kind="image", name="receipt.png", uri=str(image))]),
    )

    content = client.responses.payload["input"][0]["content"]
    assert response.metadata["multimodal_images"] == 1
    assert content[0] == {"type": "input_text", "text": "read image"}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")


def test_openai_adapter_lists_models():
    adapter = OpenAIAdapter(ProviderSettings(enabled=True))
    client = FakeOpenAIClient()
    client.models = type(
        "Models",
        (),
        {"list": lambda self: type("ModelList", (), {"data": [{"id": "gpt-5.5"}, {"id": "gpt-5.4-mini"}]})()},
    )()
    adapter._client = client

    models = adapter.list_models()

    assert [model.model_ref for model in models] == ["openai:gpt-5.4-mini", "openai:gpt-5.5"]


def test_provider_model_to_dict_is_json_serializable():
    model = ProviderModel(
        id="claude-sonnet-4-6",
        provider="anthropic",
        metadata={"created_at": datetime(2026, 6, 19, 9, 30), "tags": {"chat", "vision"}},
    )

    payload = model.to_dict()

    assert payload["metadata"]["created_at"] == "2026-06-19T09:30:00"
    assert payload["metadata"]["tags"] == ["chat", "vision"]
    json.dumps(payload)


class FakeAnthropicMessages:
    def __init__(self):
        self.payload = None

    def create(self, **payload):
        self.payload = payload

        class Block:
            text = "claude text"

        class Message:
            content = [Block()]
            usage = {"input_tokens": 3, "output_tokens": 4}

        return Message()


class FakeAnthropicClient:
    def __init__(self):
        self.messages = FakeAnthropicMessages()


def test_anthropic_adapter_uses_messages_create():
    adapter = AnthropicAdapter(ProviderSettings(enabled=True))
    client = FakeAnthropicClient()
    adapter._client = client

    response = adapter.generate(model="claude-opus-4-8", prompt="hello", request=RequestEnvelope(task="x"))

    assert response.text == "claude text"
    assert client.messages.payload["model"] == "claude-opus-4-8"
    assert client.messages.payload["messages"] == [{"role": "user", "content": "hello"}]
    assert client.messages.payload["max_tokens"] == 1024


def test_anthropic_adapter_sends_native_image_content(tmp_path):
    image = tmp_path / "receipt.png"
    image.write_bytes(b"image-bytes")
    adapter = AnthropicAdapter(ProviderSettings(enabled=True))
    client = FakeAnthropicClient()
    adapter._client = client

    response = adapter.generate(
        model="claude-sonnet-4-6",
        prompt="read image",
        request=RequestEnvelope(task="x", files=[FileAsset(kind="image", name="receipt.png", uri=str(image))]),
    )

    content = client.messages.payload["messages"][0]["content"]
    assert response.metadata["multimodal_images"] == 1
    assert content[0] == {"type": "text", "text": "read image"}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"
    assert content[1]["source"]["media_type"] == "image/png"


def test_anthropic_adapter_retries_without_deprecated_temperature():
    class TemperatureDeprecatedMessages:
        def __init__(self):
            self.payloads = []

        def create(self, **payload):
            self.payloads.append(payload)
            if "temperature" in payload:
                raise Exception("`temperature` is deprecated for this model.")

            class Block:
                text = "claude text"

            class Message:
                content = [Block()]
                usage = {"input_tokens": 3, "output_tokens": 4}

            return Message()

    adapter = AnthropicAdapter(ProviderSettings(enabled=True))
    client = FakeAnthropicClient()
    client.messages = TemperatureDeprecatedMessages()
    adapter._client = client

    response = adapter.generate(
        model="claude-opus-4-8",
        prompt="hello",
        request=RequestEnvelope(task="x", constraints={"temperature": 0}),
    )

    assert response.text == "claude text"
    assert "temperature" in client.messages.payloads[0]
    assert "temperature" not in client.messages.payloads[1]
    assert response.metadata["removed_params"] == ["temperature"]


def test_anthropic_adapter_lists_models():
    adapter = AnthropicAdapter(ProviderSettings(enabled=True))
    client = FakeAnthropicClient()
    client.models = type(
        "Models",
        (),
        {
            "list": lambda self: type(
                "ModelList",
                (),
                {"data": [{"id": "claude-opus-4-8", "display_name": "Claude Opus 4.8"}]},
            )()
        },
    )()
    adapter._client = client

    models = adapter.list_models()

    assert models[0].model_ref == "anthropic:claude-opus-4-8"
    assert models[0].name == "Claude Opus 4.8"


class FakeGoogleModels:
    def __init__(self):
        self.generate_payload = None
        self.embed_payload = None
        self.list_called = False

    def generate_content(self, **payload):
        self.generate_payload = payload
        return {"text": "gemini text", "usage_metadata": {"prompt_token_count": 1, "candidates_token_count": 2}}

    def list(self):
        self.list_called = True
        return [
            {
                "name": "models/gemini-3.5-flash",
                "display_name": "Gemini 3.5 Flash",
                "supported_actions": ["generateContent"],
            },
            {"name": "models/gemini-embedding-001", "supported_actions": ["embedContent"]},
        ]

    def embed_content(self, **payload):
        self.embed_payload = payload
        return {"embeddings": [{"values": [0.1, 0.2, 0.3]}]}


class FakeGoogleClient:
    def __init__(self):
        self.models = FakeGoogleModels()


def test_google_adapter_uses_generate_content():
    adapter = GoogleAdapter(ProviderSettings(enabled=True))
    client = FakeGoogleClient()
    adapter._client = client

    response = adapter.generate(model="gemini-3.5-flash", prompt="hello", request=RequestEnvelope(task="x"))

    assert response.text == "gemini text"
    assert client.models.generate_payload["model"] == "gemini-3.5-flash"
    assert client.models.generate_payload["contents"] == "hello"
    assert response.metadata["api"] == "models.generate_content"


def test_google_adapter_sends_native_image_content(tmp_path):
    image = tmp_path / "receipt.png"
    image.write_bytes(b"image-bytes")
    adapter = GoogleAdapter(ProviderSettings(enabled=True))
    client = FakeGoogleClient()
    adapter._client = client

    response = adapter.generate(
        model="gemini-3.5-flash",
        prompt="read image",
        request=RequestEnvelope(task="x", files=[FileAsset(kind="image", name="receipt.png", uri=str(image))]),
    )

    contents = client.models.generate_payload["contents"]
    assert response.metadata["multimodal_images"] == 1
    assert contents[0] == {"text": "read image"}
    assert contents[1]["inline_data"]["mime_type"] == "image/png"
    assert contents[1]["inline_data"]["data"]


def test_google_adapter_lists_models():
    adapter = GoogleAdapter(ProviderSettings(enabled=True))
    client = FakeGoogleClient()
    adapter._client = client

    models = adapter.list_models()

    assert [model.model_ref for model in models] == ["google:gemini-3.5-flash", "google:gemini-embedding-001"]
    assert models[0].name == "Gemini 3.5 Flash"


def test_google_adapter_embeds_content():
    adapter = GoogleAdapter(ProviderSettings(enabled=True))
    client = FakeGoogleClient()
    adapter._client = client

    response = adapter.embed(model="gemini-embedding-001", input=["hello"])

    assert response.embeddings == [[0.1, 0.2, 0.3]]
    assert client.models.embed_payload == {"model": "gemini-embedding-001", "contents": ["hello"]}
    assert response.metadata["embedding_dimensions"] == 3


def test_google_adapter_accepts_gemini_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    adapter = GoogleAdapter(ProviderSettings(enabled=True, env_key="GOOGLE_API_KEY"))

    assert google_api_key(adapter.settings) == "test-key"


def test_google_adapter_missing_key_raises_auth(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    adapter = GoogleAdapter(ProviderSettings(enabled=True, env_key="GOOGLE_API_KEY"))

    try:
        adapter.list_models()
    except CrupierProviderAuthError as exc:
        assert exc.provider == "google"
        assert "GOOGLE_API_KEY" in exc.env_key
    else:
        raise AssertionError("Google adapter should require an API key")


def test_default_adapter_factory_builds_google_adapter():
    config = CrupierConfig.from_dict({"providers": {"google": {"enabled": True, "env_key": "GOOGLE_API_KEY"}}})

    adapters = build_default_adapters(config)

    assert isinstance(adapters["google"], GoogleAdapter)


def test_ollama_cloud_requires_api_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    adapter = OllamaAdapter(ProviderSettings(enabled=True))

    try:
        adapter.generate(model="gpt-oss:120b", prompt="hello", request=RequestEnvelope(task="x"))
    except CrupierProviderAuthError as exc:
        assert exc.provider == "ollama"
        assert exc.env_key == "OLLAMA_API_KEY"
    else:
        raise AssertionError("Ollama Cloud should require an API key")


def test_ollama_adapter_posts_to_api_chat(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"message": {"content": "ollama text"}, "eval_count": 2}).encode()

    def fake_urlopen(req, timeout):
        requests.append((req, timeout))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OllamaAdapter(ProviderSettings(enabled=True, host="http://localhost:11434"))

    response = adapter.generate(model="llama3.2", prompt="hello", request=RequestEnvelope(task="x"))

    assert response.text == "ollama text"
    assert requests[0][0].full_url == "http://localhost:11434/api/chat"


def test_ollama_adapter_sends_native_image_content(tmp_path, monkeypatch):
    image = tmp_path / "receipt.png"
    image.write_bytes(b"image-bytes")
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"message": {"content": "ollama vision"}}).encode()

    def fake_urlopen(req, timeout):
        requests.append((req, timeout))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OllamaAdapter(ProviderSettings(enabled=True, host="http://localhost:11434"))

    response = adapter.generate(
        model="llava",
        prompt="read image",
        request=RequestEnvelope(task="x", files=[FileAsset(kind="image", name="receipt.png", uri=str(image))]),
    )

    payload = json.loads(requests[0][0].data.decode("utf-8"))
    assert response.metadata["multimodal_images"] == 1
    assert payload["messages"][0]["content"] == "read image"
    assert len(payload["messages"][0]["images"]) == 1


def test_ollama_adapter_lists_models(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"models": [{"model": "gpt-oss:120b", "name": "gpt-oss:120b"}]}).encode()

    def fake_urlopen(req, timeout):
        requests.append((req, timeout))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OllamaAdapter(ProviderSettings(enabled=True, host="https://ollama.com/api"))
    monkeypatch.setenv("OLLAMA_API_KEY", "test")

    models = adapter.list_models()

    assert models[0].model_ref == "ollama:gpt-oss:120b"
    assert requests[0][0].full_url == "https://ollama.com/api/tags"


def test_ollama_adapter_posts_to_api_embed(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {"embeddings": [[0.1, 0.2, 0.3]], "prompt_eval_count": 4, "total_duration": 10}
            ).encode()

    def fake_urlopen(req, timeout):
        requests.append((req, timeout))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("OLLAMA_API_KEY", "test")
    adapter = OllamaAdapter(ProviderSettings(enabled=True, host="https://ollama.com/api"))

    response = adapter.embed(model="all-minilm", input=["hello"])

    assert response.embeddings == [[0.1, 0.2, 0.3]]
    assert response.usage["prompt_eval_count"] == 4
    assert response.metadata["embedding_dimensions"] == 3
    assert requests[0][0].full_url == "https://ollama.com/api/embed"
