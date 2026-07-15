import json
import sys
from datetime import datetime
from types import SimpleNamespace

from crupier.adapters import ProviderModel
from crupier.adapters.anthropic import AnthropicAdapter
from crupier.adapters.factory import build_default_adapters
from crupier.adapters.google import GoogleAdapter, google_api_key
from crupier.adapters.nan import NaNAdapter
from crupier.adapters.ollama import OllamaAdapter
from crupier.adapters.openai import OpenAIAdapter
from crupier.adapters.openrouter import OpenRouterAdapter
from crupier.config import NAN_DEFAULT_HOST, OPENROUTER_DEFAULT_HOST, CrupierConfig, ProviderSettings
from crupier.errors import CrupierModelUnsupportedError, CrupierProviderAuthError, CrupierProviderUnavailableError
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


class FakeNaNChatCompletions:
    def __init__(self):
        self.payload = None

    def create(self, **payload):
        self.payload = payload
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
            usage={"prompt_tokens": 3, "completion_tokens": 4},
        )


class FakeNaNClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeNaNChatCompletions())


class FakeNaNSpecializedClient:
    def __init__(self):
        self.post_payload = None
        self.transcription_payload = None
        self.speech_payload = None
        self.image_generate_payload = None
        self.image_edit_payload = None
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=self._transcribe),
            speech=SimpleNamespace(create=self._speech),
        )
        self.images = SimpleNamespace(
            generate=self._generate_image,
            edit=self._edit_image,
        )

    def with_options(self, **kwargs):
        self.options = kwargs
        return self

    def post(self, **payload):
        self.post_payload = payload
        return {
            "results": [
                {"index": 1, "relevance_score": 0.9, "document": {"text": "Paris"}},
                {"index": 0, "relevance_score": 0.2, "document": {"text": "Berlin"}},
            ],
            "meta": {"tokens": {"input_tokens": 12}},
        }

    def _transcribe(self, **payload):
        self.transcription_payload = payload
        return {"text": "hola mundo", "language": "es", "duration": 1.5}

    def _speech(self, **payload):
        self.speech_payload = payload
        return SimpleNamespace(content=b"ID3-audio")

    def _generate_image(self, **payload):
        self.image_generate_payload = payload
        return {"data": [{"url": "https://example.test/generated.png"}]}

    def _edit_image(self, **payload):
        self.image_edit_payload = payload
        return {"data": [{"b64_json": "aW1hZ2U="}]}


def test_openai_adapter_uses_responses_create():
    adapter = OpenAIAdapter(ProviderSettings(enabled=True))
    client = FakeOpenAIClient()
    adapter._client = client

    response = adapter.generate(model="gpt-5.5", prompt="hello", request=RequestEnvelope(task="x"))

    assert response.text == "openai text"
    assert client.responses.payload["model"] == "gpt-5.5"
    assert client.responses.payload["input"] == "hello"


def test_openai_adapter_sends_request_timeout():
    adapter = OpenAIAdapter(ProviderSettings(enabled=True))
    client = FakeOpenAIClient()
    adapter._client = client

    adapter.generate(
        model="gpt-5.5",
        prompt="hello",
        request=RequestEnvelope(task="x", constraints={"timeout_seconds": 12.5}),
    )

    assert client.responses.payload["timeout"] == 12.5


def test_openai_adapter_retries_without_unsupported_temperature():
    class UnsupportedTemperatureResponses:
        def __init__(self):
            self.payloads = []

        def create(self, **payload):
            self.payloads.append(payload)
            if "temperature" in payload:
                raise Exception("Unsupported parameter: 'temperature' is not supported with this model.")

            class Response:
                output_text = "openai text"
                usage = {"input_tokens": 1, "output_tokens": 2}

            return Response()

    adapter = OpenAIAdapter(ProviderSettings(enabled=True))
    client = FakeOpenAIClient()
    client.responses = UnsupportedTemperatureResponses()
    adapter._client = client

    response = adapter.generate(
        model="gpt-5.5",
        prompt="hello",
        request=RequestEnvelope(task="x", constraints={"temperature": 0}),
    )

    assert response.text == "openai text"
    assert "temperature" in client.responses.payloads[0]
    assert "temperature" not in client.responses.payloads[1]
    assert response.metadata["removed_params"] == ["temperature"]


def test_openai_adapter_builds_client_with_provider_timeout(monkeypatch):
    calls = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    adapter = OpenAIAdapter(ProviderSettings(enabled=True, options={"timeout_seconds": 30}))

    client = adapter._build_client()

    assert isinstance(client, FakeOpenAI)
    assert calls["timeout"] == 30.0


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


def test_openai_adapter_sends_native_pdf_content(tmp_path):
    pdf = tmp_path / "contract.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    adapter = OpenAIAdapter(ProviderSettings(enabled=True))
    client = FakeOpenAIClient()
    adapter._client = client

    response = adapter.generate(
        model="gpt-5.4-mini",
        prompt="read PDF",
        request=RequestEnvelope(task="x", files=[FileAsset(kind="pdf", name="contract.pdf", uri=str(pdf))]),
    )

    content = client.responses.payload["input"][0]["content"]
    assert response.metadata["native_files"] == 1
    assert content[1]["type"] == "input_file"
    assert content[1]["filename"] == "contract.pdf"
    assert content[1]["file_data"].startswith("data:application/pdf;base64,")


def test_openai_adapter_sends_native_json_schema_format():
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    adapter = OpenAIAdapter(ProviderSettings(enabled=True))
    client = FakeOpenAIClient()
    adapter._client = client

    response = adapter.generate(
        model="gpt-5.4-mini",
        prompt="return json",
        request=RequestEnvelope(
            task="x",
            response_schema=schema,
            constraints={"response_schema_name": "route_check"},
        ),
    )

    assert response.metadata["response_format"] == "json_schema"
    assert client.responses.payload["text"]["format"] == {
        "type": "json_schema",
        "name": "route_check",
        "schema": schema,
        "strict": True,
    }


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


def test_openrouter_adapter_builds_openai_compatible_client(monkeypatch):
    calls = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    adapter = OpenRouterAdapter(
        ProviderSettings(
            enabled=True,
            options={"http_referer": "https://example.com", "title": "Crupier", "timeout_seconds": 9},
        )
    )

    client = adapter._build_client()

    assert isinstance(client, FakeOpenAI)
    assert calls["api_key"] == "test-openrouter-key"
    assert calls["base_url"] == OPENROUTER_DEFAULT_HOST
    assert calls["default_headers"] == {
        "HTTP-Referer": "https://example.com",
        "X-OpenRouter-Title": "Crupier",
    }
    assert calls["timeout"] == 9.0


def test_openrouter_adapter_lists_models_with_openrouter_provider():
    adapter = OpenRouterAdapter(ProviderSettings(enabled=True))
    client = FakeOpenAIClient()
    client.models = type(
        "Models",
        (),
        {
            "list": lambda self: type(
                "ModelList",
                (),
                {"data": [{"id": "openai/gpt-4o"}, {"id": "anthropic/claude-sonnet-4.5"}]},
            )()
        },
    )()
    adapter._client = client

    models = adapter.list_models()

    assert [model.model_ref for model in models] == [
        "openrouter:anthropic/claude-sonnet-4.5",
        "openrouter:openai/gpt-4o",
    ]


def test_nan_adapter_builds_openai_compatible_client(monkeypatch):
    calls = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("NAN_API_KEY", "test-nan-key")
    adapter = NaNAdapter(ProviderSettings(enabled=True, options={"timeout_seconds": 11}))

    adapter._build_client()

    assert calls["api_key"] == "test-nan-key"
    assert calls["base_url"] == NAN_DEFAULT_HOST
    assert calls["timeout"] == 11.0


def test_nan_qwen_adapter_sends_image_schema_and_thinking_control(tmp_path):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"image")
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    client = FakeNaNClient()
    adapter._client = client

    response = adapter.generate(
        model="qwen3.6",
        prompt="classify image",
        request=RequestEnvelope(
            task="x",
            files=[FileAsset(kind="image", name="diagram.png", uri=str(image))],
            response_schema=schema,
            constraints={"disable_thinking": True, "timeout_seconds": 7},
        ),
    )

    payload = client.chat.completions.payload
    assert response.text == '{"ok": true}'
    assert response.metadata["multimodal_images"] == 1
    assert payload["messages"][0]["content"][1]["type"] == "image_url"
    assert payload["response_format"]["json_schema"]["schema"] == schema
    assert payload["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert payload["timeout"] == 7.0


def test_nan_mimo_adapter_sends_native_audio(tmp_path):
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"RIFF test")
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    client = FakeNaNClient()
    adapter._client = client

    response = adapter.generate(
        model="mimo-v2.5",
        prompt="summarize",
        request=RequestEnvelope(task="x", files=[FileAsset(kind="audio", name="call.wav", uri=str(audio))]),
    )

    content = client.chat.completions.payload["messages"][0]["content"]
    assert response.metadata["multimodal_audio"] == 1
    assert response.metadata["reasoning_mode"] == "always"
    assert content[1]["type"] == "input_audio"
    assert content[1]["input_audio"]["format"] == "wav"


def test_nan_deepseek_adapter_validates_reasoning_effort():
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    adapter._client = FakeNaNClient()

    try:
        adapter.generate(
            model="deepseek-v4-flash",
            prompt="reason",
            request=RequestEnvelope(task="x", constraints={"reasoning_effort": "maximum"}),
        )
    except CrupierModelUnsupportedError as exc:
        assert "low, medium, or high" in str(exc)
    else:
        raise AssertionError("invalid NaN reasoning effort should fail before the provider call")


def test_nan_embedding_adapter_enforces_fixed_dimensions():
    adapter = NaNAdapter(ProviderSettings(enabled=True))

    try:
        adapter.embed(model="qwen3-embedding", input="hello", dimensions=512)
    except CrupierModelUnsupportedError as exc:
        assert "4096" in str(exc)
    else:
        raise AssertionError("NaN fixed embedding dimensions should not be silently truncated")


def test_nan_adapter_executes_rerank_with_bounded_top_n():
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    client = FakeNaNSpecializedClient()
    adapter._client = client

    response = adapter.execute_operation(
        operation="reranker",
        model="rerank",
        request=RequestEnvelope(task="rank", constraints={"timeout_seconds": 4}),
        payload={"query": "capital of France", "documents": ["Berlin", "Paris"], "top_n": 2},
    )

    assert response.output[0]["index"] == 1
    assert response.usage == {"input_tokens": 12}
    assert client.post_payload["path"] == "/rerank"
    assert client.post_payload["body"]["top_n"] == 2
    assert client.options == {"timeout": 4.0}


def test_nan_adapter_executes_transcription_and_tts(tmp_path):
    audio = tmp_path / "short.mp3"
    audio.write_bytes(b"ID3-audio")
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    client = FakeNaNSpecializedClient()
    adapter._client = client

    transcript = adapter.execute_operation(
        operation="transcription",
        model="whisper",
        request=RequestEnvelope(task="transcribe"),
        payload={
            "file": audio,
            "language": "es",
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        },
    )
    speech = adapter.execute_operation(
        operation="tts",
        model="kokoro",
        request=RequestEnvelope(task="speak"),
        payload={"input": "Hola", "voice": "ef_dora", "response_format": "mp3", "speed": 1.1},
    )

    assert transcript.output["text"] == "hola mundo"
    assert client.transcription_payload["file"][0] == "short.mp3"
    assert client.transcription_payload["timestamp_granularities"] == ["segment"]
    assert speech.output == b"ID3-audio"
    assert speech.metadata["bytes"] == 9
    assert client.speech_payload["voice"] == "ef_dora"


def test_nan_upload_limits_are_enforced_before_unbounded_reads(tmp_path):
    oversized = tmp_path / "oversized.wav"
    with oversized.open("wb") as handle:
        handle.truncate(25_000_000)
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    adapter._client = FakeNaNSpecializedClient()

    try:
        adapter.execute_operation(
            operation="transcription",
            model="whisper",
            request=RequestEnvelope(task="transcribe"),
            payload={"file": oversized},
        )
    except CrupierModelUnsupportedError as exc:
        assert "smaller than 25 MB" in str(exc)
    else:
        raise AssertionError("oversized path must fail before its content is loaded")

    class BoundedReader:
        name = "bounded.wav"

        def __init__(self):
            self.read_size = None
            self.position = 7

        def tell(self):
            return self.position

        def seek(self, position):
            self.position = position

        def read(self, size):
            self.read_size = size
            self.position += 1
            return b"RIFF"

    stream = BoundedReader()
    adapter.execute_operation(
        operation="transcription",
        model="whisper",
        request=RequestEnvelope(task="transcribe"),
        payload={"file": stream},
    )

    assert stream.read_size == 25_000_000
    assert stream.position == 7


def test_nan_adapter_executes_image_generation_and_edit(tmp_path):
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"png-data")
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    client = FakeNaNSpecializedClient()
    adapter._client = client

    generated = adapter.execute_operation(
        operation="image_generation",
        model="flux-2-klein",
        request=RequestEnvelope(task="image"),
        payload={
            "prompt": "A lighthouse",
            "size": "1024x768",
            "n": 1,
            "response_format": "url",
            "seed": 42,
        },
    )
    edited = adapter.execute_operation(
        operation="image_generation",
        model="flux-2-klein",
        request=RequestEnvelope(task="edit"),
        payload={
            "prompt": "Add snow",
            "images": [reference],
            "size": "1024x1024",
            "response_format": "b64_json",
        },
    )

    assert generated.output == [{"url": "https://example.test/generated.png"}]
    assert client.image_generate_payload["extra_body"] == {"seed": 42}
    assert edited.output == [{"b64_json": "aW1hZ2U="}]
    assert client.image_edit_payload["image"][0][0] == "reference.png"


def test_openrouter_adapter_reports_openrouter_request_failures():
    class BrokenResponses:
        def create(self, **payload):
            raise RuntimeError("provider down")

    client = FakeOpenAIClient()
    client.responses = BrokenResponses()
    adapter = OpenRouterAdapter(ProviderSettings(enabled=True))
    adapter._client = client

    try:
        adapter.generate(model="openai/gpt-4o", prompt="hello", request=RequestEnvelope(task="x"))
    except CrupierProviderUnavailableError as exc:
        assert "OpenRouter request failed" in str(exc)
    else:
        raise AssertionError("OpenRouter provider failures should be mapped with provider-specific text")


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


def test_anthropic_adapter_sends_request_timeout():
    adapter = AnthropicAdapter(ProviderSettings(enabled=True))
    client = FakeAnthropicClient()
    adapter._client = client

    adapter.generate(
        model="claude-opus-4-8",
        prompt="hello",
        request=RequestEnvelope(task="x", constraints={"timeout_seconds": 22}),
    )

    assert client.messages.payload["timeout"] == 22.0


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


def test_google_adapter_uses_minimal_thinking_for_short_outputs():
    adapter = GoogleAdapter(ProviderSettings(enabled=True))
    client = FakeGoogleClient()
    adapter._client = client

    adapter.generate(
        model="gemini-3.5-flash",
        prompt="hello",
        request=RequestEnvelope(task="x", constraints={"max_output_tokens": 64}),
    )

    assert client.models.generate_payload["config"]["thinking_config"] == {"thinking_level": "minimal"}


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
    first_text = contents[0].get("text") if isinstance(contents[0], dict) else getattr(contents[0], "text", None)
    assert first_text == "read image"
    inline_data = (
        contents[1]["inline_data"] if isinstance(contents[1], dict) else getattr(contents[1], "inline_data", None)
    )
    mime_type = inline_data["mime_type"] if isinstance(inline_data, dict) else getattr(inline_data, "mime_type", None)
    data = inline_data["data"] if isinstance(inline_data, dict) else getattr(inline_data, "data", None)
    assert mime_type == "image/png"
    assert data


def test_google_adapter_lists_models():
    adapter = GoogleAdapter(ProviderSettings(enabled=True))
    client = FakeGoogleClient()
    adapter._client = client

    models = adapter.list_models()

    assert [model.model_ref for model in models] == ["google:gemini-3.5-flash", "google:gemini-embedding-001"]
    assert models[0].name == "Gemini 3.5 Flash"


def test_factory_builds_openrouter_adapter_when_enabled():
    config = CrupierConfig.from_dict(
        {
            "providers": {
                "openrouter": {
                    "enabled": True,
                    "mode": "byok",
                    "host": "https://openrouter.ai/api/v1",
                    "env_key": "OPENROUTER_API_KEY",
                }
            }
        }
    )

    adapters = build_default_adapters(config)

    assert isinstance(adapters["openrouter"], OpenRouterAdapter)


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


def test_google_adapter_builds_client_with_http_options_timeout(monkeypatch):
    calls = {}

    class FakeHttpOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGenAI:
        class Client:
            def __init__(self, **kwargs):
                calls.update(kwargs)

    fake_types = SimpleNamespace(HttpOptions=FakeHttpOptions)
    fake_genai = SimpleNamespace(Client=FakeGenAI.Client, types=fake_types)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    adapter = GoogleAdapter(ProviderSettings(enabled=True, options={"timeout_seconds": 2.5}))

    adapter._build_client()

    assert calls["http_options"].kwargs == {"timeout": 2500}


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

    response = adapter.generate(
        model="llama3.2",
        prompt="hello",
        request=RequestEnvelope(task="x", constraints={"timeout_seconds": 7}),
    )

    assert response.text == "ollama text"
    assert requests[0][0].full_url == "http://localhost:11434/api/chat"
    assert requests[0][1] == 7.0


def test_ollama_adapter_sends_native_json_schema_format(monkeypatch):
    requests = []
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"message": {"content": '{"ok": true}'}}).encode()

    def fake_urlopen(req, timeout):
        requests.append((req, timeout))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OllamaAdapter(ProviderSettings(enabled=True, host="http://localhost:11434"))

    response = adapter.generate(
        model="llama3.2",
        prompt="return json",
        request=RequestEnvelope(task="x", response_schema=schema),
    )

    payload = json.loads(requests[0][0].data.decode("utf-8"))
    assert payload["format"] == schema
    assert response.metadata["response_format"] == "json_schema"


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
