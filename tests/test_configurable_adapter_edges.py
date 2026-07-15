import sys
from types import SimpleNamespace

import pytest

import crupier.adapters.nan as configurable_module
from crupier.adapters.nan import NaNAdapter
from crupier.config import ProviderSettings
from crupier.errors import (
    CrupierModelUnsupportedError,
    CrupierProviderAuthError,
    CrupierProviderRateLimitError,
    CrupierProviderUnavailableError,
)
from crupier.models import RequestEnvelope


class RecordingCompletions:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or {"choices": [{"message": {"content": "ok"}}]}

    def create(self, **payload):
        self.calls.append(payload)
        return self.response


def _adapter_with_completions(response=None):
    completions = RecordingCompletions(response)
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return adapter, completions


def test_configurable_adapter_file_and_chat_model_contracts():
    assert NaNAdapter.supports_file_kind(model="qwen3.6", kind="image") is True
    assert NaNAdapter.supports_file_kind(model="mimo-v2.5", kind="audio") is True
    assert NaNAdapter.supports_file_kind(model="deepseek-v4-flash", kind="image") is False
    assert NaNAdapter.supports_file_kind(model="qwen3.6", kind="pdf") is False

    with pytest.raises(CrupierModelUnsupportedError, match="not a chat-generation model"):
        NaNAdapter(ProviderSettings(enabled=True)).generate(
            model="qwen3-embedding", prompt="x", request=RequestEnvelope(task="x")
        )


def test_configurable_chat_honors_generation_schema_and_reasoning_parameters():
    adapter, completions = _adapter_with_completions(
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=[{"text": "one"}, SimpleNamespace(text=" two")]))],
            usage=SimpleNamespace(total_tokens=8),
        )
    )
    schema = {"type": "object"}

    result = adapter.generate(
        model="deepseek-v4-flash",
        prompt="reason",
        request=RequestEnvelope(
            task="x",
            response_schema=schema,
            constraints={
                "max_tokens": "20",
                "temperature": 0.2,
                "top_p": 0.8,
                "reasoning_effort": "high",
                "response_schema_name": " result ",
                "strict_response_schema": False,
            },
        ),
    )

    payload = completions.calls[0]
    assert payload["max_tokens"] == 20
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.8
    assert payload["reasoning_effort"] == "high"
    assert payload["response_format"]["json_schema"] == {
        "name": "result",
        "strict": False,
        "schema": schema,
    }
    assert result.text == "one two"
    assert result.usage == {"total_tokens": 8}
    assert result.metadata["reasoning_mode"] == "high"


def test_configurable_chat_maps_provider_errors():
    adapter, completions = _adapter_with_completions()
    completions.create = lambda **payload: (_ for _ in ()).throw(Exception("offline"))

    with pytest.raises(CrupierProviderUnavailableError, match="request failed: offline"):
        adapter.generate(model="qwen3.6", prompt="x", request=RequestEnvelope(task="x"))


def test_configurable_catalog_supports_objects_skips_missing_and_maps_errors():
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[SimpleNamespace(id="z", input_tokens=1), {"name": "missing"}, {"id": "a"}]
            )
        )
    )

    models = adapter.list_models()
    assert [item.id for item in models] == ["a", "z"]
    assert models[1].metadata == {"input_tokens": 1}

    class RateLimitError(Exception):
        pass

    adapter._client = SimpleNamespace(
        models=SimpleNamespace(list=lambda: (_ for _ in ()).throw(RateLimitError("slow")))
    )
    with pytest.raises(CrupierProviderRateLimitError, match="slow"):
        adapter.list_models()


def test_configurable_embeddings_enforce_model_dimensions_parse_and_map_errors():
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    with pytest.raises(CrupierModelUnsupportedError, match="not its embedding model"):
        adapter.embed(model="other", input="x")

    embeddings = SimpleNamespace(
        create=lambda **payload: SimpleNamespace(
            data=[{"embedding": [1, 2]}, SimpleNamespace(embedding=[3, 4]), {"missing": True}],
            usage=SimpleNamespace(total_tokens=2),
        )
    )
    adapter._client = SimpleNamespace(embeddings=embeddings)
    result = adapter.embed(model="qwen3-embedding", input="x", dimensions=4096)
    assert result.embeddings == [[1.0, 2.0], [3.0, 4.0]]
    assert result.usage == {"total_tokens": 2}

    adapter._client = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **payload: (_ for _ in ()).throw(Exception("offline")))
    )
    with pytest.raises(CrupierProviderUnavailableError, match="offline"):
        adapter.embed(model="qwen3-embedding", input="x")


def test_configurable_operation_dispatch_rejects_incompatible_and_invalid_payloads(monkeypatch):
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace()

    assert adapter.supports_operation(operation="reranker", model="rerank") is True
    assert adapter.supports_operation(operation="reranker", model="other") is False
    with pytest.raises(CrupierModelUnsupportedError, match="cannot execute"):
        adapter.execute_operation(
            operation="reranker", model="other", request=RequestEnvelope(task="x"), payload={}
        )

    monkeypatch.setattr(adapter, "supports_operation", lambda **kwargs: True)
    with pytest.raises(CrupierModelUnsupportedError, match="Unsupported NaN operation"):
        adapter.execute_operation(
            operation="custom", model="custom", request=RequestEnvelope(task="x"), payload={}
        )

    monkeypatch.setattr(adapter, "_rerank", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad int")))
    with pytest.raises(CrupierModelUnsupportedError, match="Invalid reranker payload: bad int"):
        adapter.execute_operation(
            operation="reranker", model="rerank", request=RequestEnvelope(task="x"), payload={}
        )

    monkeypatch.setattr(adapter, "_rerank", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    with pytest.raises(CrupierProviderUnavailableError, match="offline"):
        adapter.execute_operation(
            operation="reranker", model="rerank", request=RequestEnvelope(task="x"), payload={}
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"query": "", "documents": ["a"]}, "non-empty query"),
        ({"query": "q", "documents": []}, "non-empty list"),
        ({"query": "q", "documents": [1]}, "list of strings"),
        ({"query": "q", "documents": ["a"], "top_n": 0}, "between 1"),
        ({"query": "q", "documents": ["a"], "top_n": 2}, "between 1"),
    ],
)
def test_configurable_rerank_validates_payload(payload, message):
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace()
    with pytest.raises(CrupierModelUnsupportedError, match=message):
        adapter.execute_operation(
            operation="reranker", model="rerank", request=RequestEnvelope(task="x"), payload=payload
        )


def test_configurable_rerank_normalizes_sparse_provider_response():
    client = SimpleNamespace(
        post=lambda **payload: {
            "results": [None, {"index": "2", "relevance_score": "0.75"}],
            "meta": {"tokens": None},
        }
    )
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    adapter._client = client

    result = adapter.execute_operation(
        operation="reranker",
        model="rerank",
        request=RequestEnvelope(task="x"),
        payload={"query": "q", "documents": ["a", "b", "c"]},
    )

    assert result.output == [{"index": 2, "relevance_score": 0.75}]
    assert result.usage == {}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"file": b"audio", "response_format": "text"}, "json or verbose_json"),
        ({"file": b"audio", "timestamp_granularities": ["word"]}, "requires response_format"),
        (
            {"file": b"audio", "response_format": "verbose_json", "timestamp_granularities": ["frame"]},
            "only 'word' or 'segment'",
        ),
        (
            {"file": b"audio", "response_format": "verbose_json", "timestamp_granularities": "word"},
            "only 'word' or 'segment'",
        ),
    ],
)
def test_configurable_transcription_validates_formats(payload, message):
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace()
    with pytest.raises(CrupierModelUnsupportedError, match=message):
        adapter.execute_operation(
            operation="transcription", model="whisper", request=RequestEnvelope(task="x"), payload=payload
        )


def test_configurable_transcription_accepts_bytes_and_object_text_response():
    body = {}

    def create(**payload):
        body.update(payload)
        return SimpleNamespace(text="transcript")

    adapter = NaNAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    result = adapter.execute_operation(
        operation="transcription",
        model="whisper",
        request=RequestEnvelope(task="x"),
        payload={"file": b"audio", "filename": "call.wav", "language": "es", "temperature": 0},
    )

    assert body["file"] == ("call.wav", b"audio", "audio/x-wav")
    assert body["language"] == "es" and body["temperature"] == 0
    assert result.output == {"text": "transcript"}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"input": "", "voice": "v"}, "non-empty input and voice"),
        ({"input": "x", "voice": ""}, "non-empty input and voice"),
        ({"input": "x", "voice": "v", "response_format": "ogg"}, "response_format"),
        ({"input": "x", "voice": "v", "speed": 0}, "speed must be positive"),
    ],
)
def test_configurable_tts_validates_payload(payload, message):
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace()
    with pytest.raises(CrupierModelUnsupportedError, match=message):
        adapter.execute_operation(
            operation="tts", model="kokoro", request=RequestEnvelope(task="x"), payload=payload
        )


@pytest.mark.parametrize(
    "response",
    [b"audio", bytearray(b"audio"), SimpleNamespace(content=bytearray(b"audio")), SimpleNamespace(read=lambda: b"audio")],
)
def test_configurable_tts_accepts_common_binary_response_shapes(response):
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(
        audio=SimpleNamespace(speech=SimpleNamespace(create=lambda **payload: response))
    )
    result = adapter.execute_operation(
        operation="tts",
        model="kokoro",
        request=RequestEnvelope(task="x"),
        payload={"input": "hello", "voice": "voice"},
    )
    assert result.output == b"audio"


def test_configurable_tts_rejects_empty_provider_audio():
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(
        audio=SimpleNamespace(speech=SimpleNamespace(create=lambda **payload: SimpleNamespace(read=lambda: "text")))
    )
    with pytest.raises(CrupierProviderUnavailableError, match="returned no audio"):
        adapter.execute_operation(
            operation="tts",
            model="kokoro",
            request=RequestEnvelope(task="x"),
            payload={"input": "hello", "voice": "voice"},
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"prompt": ""}, "non-empty prompt"),
        ({"prompt": "x", "n": 0}, "between 1 and 4"),
        ({"prompt": "x", "n": 5}, "between 1 and 4"),
        ({"prompt": "x", "response_format": "bytes"}, "url or b64_json"),
        ({"prompt": "x", "images": [b"x"], "mask": b"m"}, "do not support masks"),
        ({"prompt": "x", "images": []}, "between 1 and 4"),
        ({"prompt": "x", "images": [b"x"] * 5}, "between 1 and 4"),
    ],
)
def test_configurable_image_operation_validates_payload(payload, message):
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace()
    with pytest.raises(CrupierModelUnsupportedError, match=message):
        adapter.execute_operation(
            operation="image_generation",
            model="flux-2-klein",
            request=RequestEnvelope(task="x"),
            payload=payload,
        )


def test_configurable_image_generation_accepts_auto_and_extra_controls():
    calls = []
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(
        images=SimpleNamespace(generate=lambda **payload: calls.append(payload) or {"data": []})
    )
    result = adapter.execute_operation(
        operation="image_generation",
        model="flux-2-klein",
        request=RequestEnvelope(task="x"),
        payload={"prompt": "x", "size": "auto", "guidance": 3.5},
    )
    assert calls[0]["extra_body"] == {"guidance": 3.5}
    assert result.metadata["count"] == 0


def test_configurable_structured_probe_handles_success_invalid_and_unknown(monkeypatch):
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    outputs = iter(['{"ok":true,"probe":"crupier"}', "not-json"])

    def fake_generate(**kwargs):
        return configurable_module.AdapterResponse(text=next(outputs), metadata={})

    monkeypatch.setattr(adapter, "generate", fake_generate)
    success = adapter.probe_capability(
        model="qwen3.6",
        probe="structured_output",
        request=RequestEnvelope(task="x", constraints={"max_output_tokens": 2}),
    )
    failed = adapter.probe_capability(
        model="qwen3.6", probe="structured_output", request=RequestEnvelope(task="x")
    )
    assert success.text == "" and success.metadata["probe_status"] == "verified"
    assert failed.metadata["probe_status"] == "failed"
    with pytest.raises(NotImplementedError, match="no native probe"):
        adapter.probe_capability(model="qwen3.6", probe="vision", request=RequestEnvelope(task="x"))


def test_configurable_tool_and_stream_probes_cover_success_failure_timeout_and_errors():
    class Completions:
        def __init__(self):
            self.calls = []
            self.fail = False

        def create(self, **payload):
            self.calls.append(payload)
            if self.fail:
                raise Exception("probe offline")
            if payload.get("stream"):
                return [
                    {"choices": [{"delta": {"content": "stream-ok"}}]},
                    *({"choices": []} for _ in range(25)),
                ]
            return {
                "choices": [
                    {"message": {"tool_calls": [{"function": {"name": "crupier_probe_tool"}}]}}
                ]
            }

    completions = Completions()
    adapter = NaNAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    request = RequestEnvelope(task="x", constraints={"timeout": 3})

    tool = adapter.probe_capability(model="qwen3.6", probe="tool_call", request=request)
    stream = adapter.probe_capability(model="qwen3.6", probe="streaming", request=request)
    assert tool.metadata["probe_status"] == "verified"
    assert stream.metadata["event_count"] == 20
    assert completions.calls[1]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert all(call["timeout"] == 3 for call in completions.calls)

    completions.fail = True
    for probe in ["tool_call", "streaming"]:
        with pytest.raises(CrupierProviderUnavailableError, match="probe offline"):
            adapter.probe_capability(model="qwen3.6", probe=probe, request=request)


def test_configurable_build_client_custom_host_and_missing_dependency(monkeypatch):
    calls = {}

    class OpenAI:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setenv("NAN_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=OpenAI))
    adapter = NaNAdapter(ProviderSettings(enabled=True, host="https://inference.example/v1"))
    adapter._build_client()
    assert calls["base_url"] == "https://inference.example/v1"

    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(CrupierProviderUnavailableError, match="optional dependency"):
        adapter._build_client()


@pytest.mark.parametrize(
    ("error", "error_type"),
    [
        (type("StatusError", (Exception,), {"status_code": 403})("bad"), CrupierProviderAuthError),
        (type("StatusError", (Exception,), {"status_code": 429})("slow"), CrupierProviderRateLimitError),
        (Exception("offline"), CrupierProviderUnavailableError),
    ],
)
def test_configurable_error_mapping(error, error_type):
    with pytest.raises(error_type):
        NaNAdapter(ProviderSettings(enabled=True))._raise_mapped_error(error)


def test_configurable_message_audio_reasoning_and_tool_helpers():
    assert configurable_module._messages(
        model="qwen3.6", prompt="x", request=RequestEnvelope(task="x")
    ) == ([{"role": "user", "content": "x"}], {"images": 0, "audio": 0})
    assert configurable_module._audio_format({"name": "call.bin", "mime_type": "audio/wav"}) == "wav"
    assert configurable_module._audio_format({"name": "call.bin", "mime_type": "audio/mpeg"}) == "mp3"
    with pytest.raises(CrupierModelUnsupportedError, match="requires WAV or MP3"):
        configurable_module._audio_format({"name": "call.flac", "mime_type": "audio/flac"})

    payload = {}
    configurable_module._apply_reasoning_options(
        payload,
        model="qwen3.6",
        request=RequestEnvelope(task="x", constraints={"enable_thinking": True}),
    )
    assert payload["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert configurable_module._reasoning_mode(
        "qwen3.6", RequestEnvelope(task="x", constraints={"enable_thinking": False})
    ) == "disabled"
    assert configurable_module._reasoning_mode(
        "qwen3.6", RequestEnvelope(task="x", constraints={"disable_thinking": False})
    ) == "enabled"
    assert configurable_module._reasoning_mode("other", RequestEnvelope(task="x")) == "provider_default"

    assert configurable_module._first_choice({"choices": []}) == {}
    assert configurable_module._message_text({"content": 7}) == "7"
    assert configurable_module._message_text({}) == ""
    assert configurable_module._has_tool_call(
        SimpleNamespace(
            tool_calls=[SimpleNamespace(function=SimpleNamespace(name="target"))]
        ),
        "target",
    ) is True
    assert configurable_module._has_tool_call({}, "target") is False


@pytest.mark.parametrize(
    ("size", "message"),
    [
        ("bad", "WIDTHxHEIGHT"),
        ("128x512", "between 256 and 1536"),
        ("300x512", "divisible by 16"),
        ("1536x256", "aspect ratio"),
    ],
)
def test_configurable_image_size_validation(size, message):
    with pytest.raises(CrupierModelUnsupportedError, match=message):
        configurable_module._validate_image_size(size)


def test_configurable_upload_tuple_accepts_tuple_bytes_path_and_file_object(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"data")
    assert configurable_module._upload_tuple(
        ("named.wav", b"audio", "audio/wav"), default_name="fallback.bin"
    ) == ("named.wav", b"audio", "audio/wav")
    assert configurable_module._upload_tuple(path, default_name="fallback.bin") == (
        "sample.bin",
        b"data",
        "application/octet-stream",
    )
    assert configurable_module._upload_tuple(bytearray(b"data"), default_name="sample.bin") == (
        "sample.bin",
        b"data",
        "application/octet-stream",
    )

    class Reader:
        name = "sample.wav"

        def read(self, size):
            return bytearray(b"audio")

    assert configurable_module._upload_tuple(Reader(), default_name="fallback.bin") == (
        "sample.wav",
        b"audio",
        "audio/x-wav",
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (("x", "text"), "content must be bytes"),
        (("x", b""), "cannot be empty"),
        ("/definitely/missing.bin", "does not exist"),
        (b"", "cannot be empty"),
        (object(), "path, bytes, or binary file object"),
    ],
)
def test_configurable_upload_tuple_rejects_invalid_inputs(value, message):
    with pytest.raises(CrupierModelUnsupportedError, match=message):
        configurable_module._upload_tuple(value, default_name="fallback.bin")


def test_configurable_upload_tuple_bounds_reads_and_restores_position(monkeypatch):
    monkeypatch.setattr(configurable_module, "_MAX_UPLOAD_BYTES", 4)
    with pytest.raises(CrupierModelUnsupportedError, match="smaller than 25 MB"):
        configurable_module._upload_tuple(("x.bin", b"1234"), default_name="x.bin")
    with pytest.raises(CrupierModelUnsupportedError, match="smaller than 25 MB"):
        configurable_module._upload_tuple(b"1234", default_name="x.bin")

    class BadReader:
        def tell(self):
            raise OSError("no tell")

        def read(self):
            return b"x"

    with pytest.raises(CrupierModelUnsupportedError, match="bounded read"):
        configurable_module._upload_tuple(BadReader(), default_name="x.bin")

    class TextReader:
        def read(self, size):
            return "text"

    with pytest.raises(CrupierModelUnsupportedError, match="must return bytes"):
        configurable_module._upload_tuple(TextReader(), default_name="x.bin")

    class SeekFails:
        def tell(self):
            return 1

        def seek(self, position):
            raise OSError("cannot seek")

        def read(self, size):
            return b"x"

    assert configurable_module._upload_tuple(SeekFails(), default_name="x.bin")[1] == b"x"


def test_configurable_response_bytes_rejects_nonbinary_reads():
    assert configurable_module._response_bytes(SimpleNamespace()) == b""
    assert configurable_module._response_bytes(SimpleNamespace(read=lambda: "text")) == b""
