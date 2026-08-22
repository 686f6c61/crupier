import sys
from types import SimpleNamespace

import pytest

from crupier import Crupier, CrupierResult, install
from crupier.adapters import AdapterResponse, EmbeddingResponse, OperationResponse
from crupier.compat import openai as openai_compat
from crupier.compat.openai import CompatBinaryResponse, CompatObject, OpenAI
from crupier.config import CrupierConfig
from crupier.errors import (
    CrupierConfigError,
    CrupierModelUnsupportedError,
    CrupierProviderUnavailableError,
)
from crupier.models import OperationResult


class FakeAdapter:
    provider = "openai"

    def __init__(self):
        self.calls = []

    def generate(self, *, model, prompt, request):
        self.calls.append({"model": model, "prompt": prompt, "messages": request.messages})
        return AdapterResponse(
            text=f"fake {model}",
            usage={"input_tokens": 3, "output_tokens": 4},
            metadata={"provider": "openai", "model": model},
        )

    def embed(self, *, model, input, dimensions=None):
        self.calls.append({"model": model, "embedding_input": input, "dimensions": dimensions})
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] if isinstance(input, list) else [[0.1, 0.2, 0.3]]
        if dimensions is not None:
            vectors = [vector[:dimensions] for vector in vectors]
        return EmbeddingResponse(
            embeddings=vectors,
            usage={"prompt_tokens": 7, "total_tokens": 7},
            metadata={"provider": "openai", "model": model, "api": "embeddings.create"},
        )


class FakeSpecializedAdapter:
    provider = "nan"

    def __init__(self):
        self.calls = []

    @staticmethod
    def supports_operation(*, operation, model):
        return model == {
            "reranker": "rerank",
            "transcription": "whisper",
            "tts": "kokoro",
            "image_generation": "flux-2-klein",
        }.get(operation)

    def execute_operation(self, *, operation, model, request, payload):
        self.calls.append({"operation": operation, "model": model, "payload": payload})
        output = {
            "reranker": [{"index": 1, "relevance_score": 0.98}],
            "transcription": {"text": "hola mundo", "language": "es"},
            "tts": b"audio-bytes",
            "image_generation": [{"url": "https://example.test/image.png"}],
        }[operation]
        return OperationResponse(
            operation=operation,
            output=output,
            metadata={"provider": "nan", "model": model},
        )

    def embed(self, *, model, input, dimensions=None):
        self.calls.append({"operation": "embedding", "model": model, "input": input, "dimensions": dimensions})
        return EmbeddingResponse(
            embeddings=[[0.9, 0.8, 0.7]],
            usage={"prompt_tokens": 3, "total_tokens": 3},
            metadata={"provider": "nan", "model": model},
        )


def make_client(tmp_path, *, dry_run=False):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "compat", "default_profile": "agentic"},
            "providers": {"openai": {"enabled": True, "env_key": "OPENAI_API_KEY"}},
            "models": {
                "allow": [
                    "openai:gpt-5.5",
                    "openai:gpt-5.4-mini",
                    "openai:text-embedding-3-small",
                ]
            },
            "routing": {"default_strategy": "single"},
        }
    )
    config.root = tmp_path
    adapter = FakeAdapter()
    crupier = Crupier(config, adapters={"openai": adapter})
    return OpenAI(crupier=crupier, dry_run=dry_run), adapter


def test_compat_client_rejects_api_key_and_base_url():
    with pytest.raises(CrupierConfigError) as caught:
        OpenAI(crupier=SimpleNamespace(), api_key="secret", base_url="https://example.test/v1")

    assert "api_key" in str(caught.value)
    assert "base_url" in str(caught.value)


def test_compat_client_warns_on_unknown_arguments():
    with pytest.warns(UserWarning, match="unknown_option"):
        OpenAI(crupier=SimpleNamespace(), unknown_option=True)


def test_compat_client_translates_timeout_and_retries():
    class SpyCrupier:
        def __init__(self):
            self.kwargs = None

        def deal(self, **kwargs):
            self.kwargs = kwargs
            return CrupierResult(output_text="ok")

    spy = SpyCrupier()
    client = OpenAI(crupier=spy, timeout=3.5, max_retries=2)

    client.responses.create(input="hello")

    assert spy.kwargs["constraints"]["timeout_seconds"] == 3.5
    assert spy.kwargs["constraints"]["max_provider_retries"] == 2


def test_compat_forwards_control_options_constraints_and_metadata():
    class SpyCrupier:
        def __init__(self):
            self.kwargs = None

        def deal(self, **kwargs):
            self.kwargs = kwargs
            return CrupierResult(output_text="ok")

    spy = SpyCrupier()
    client = OpenAI(crupier=spy)

    response = client.responses.create(
        input="hello",
        model="gpt-5.4-mini",
        crupier={
            "experiment": "model-rollout",
            "approval_token": "apr_test.secret-value-long-enough",
            "constraints": {"max_cost_usd": 0.02},
        },
        constraints={"max_latency_ms": 1500},
        metadata={"session_id": "ses_123"},
    )

    assert spy.kwargs["experiment"] == "model-rollout"
    assert spy.kwargs["approval_token"] == "apr_test.secret-value-long-enough"
    assert spy.kwargs["metadata"] == {"session_id": "ses_123"}
    assert spy.kwargs["constraints"]["max_cost_usd"] == 0.02
    assert spy.kwargs["constraints"]["max_latency_ms"] == 1500
    assert response.crupier.experiment is None


def make_specialized_client(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "compat-specialized"},
            "providers": {"nan": {"enabled": True, "env_key": "NAN_API_KEY"}},
            "models": {
                "allow": [
                    "nan:rerank",
                    "nan:whisper",
                    "nan:kokoro",
                    "nan:flux-2-klein",
                    "nan:qwen3-embedding",
                ]
            },
            "routing": {"require_operational_providers": False},
        }
    )
    config.root = tmp_path
    adapter = FakeSpecializedAdapter()
    crupier = Crupier(config, adapters={"nan": adapter})
    return OpenAI(crupier=crupier), adapter


def test_responses_create_returns_openai_like_object(tmp_path):
    client, adapter = make_client(tmp_path)

    response = client.responses.create(
        model="gpt-5.4-mini",
        input="Say hi",
        instructions="Reply briefly.",
        trace="summary",
    )

    assert response.object == "response"
    assert response.output_text == "fake gpt-5.5"
    assert response.output[0].content[0].text == "fake gpt-5.5"
    assert response.usage.input_tokens == 3
    assert response.usage.output_tokens == 4
    assert response.crupier.route["strategy"] == "single"
    assert adapter.calls[0]["model"] == "gpt-5.5"
    assert response.model_dump()["output_text"] == "fake gpt-5.5"


def test_chat_completions_create_returns_openai_like_choices(tmp_path):
    client, adapter = make_client(tmp_path)

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "Summarize this"}],
    )

    assert response.object == "chat.completion"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "fake gpt-5.5"
    assert adapter.calls[0]["messages"][0]["content"] == "Summarize this"


def test_strict_mode_forces_requested_openai_model(tmp_path):
    client, adapter = make_client(tmp_path)

    response = client.responses.create(
        model="gpt-5.4-mini",
        input="Use exact model",
        compat_mode="strict",
    )

    assert response.model == "openai:gpt-5.4-mini"
    assert adapter.calls[0]["model"] == "gpt-5.4-mini"


def test_chat_completion_stream_yields_compatible_chunk(tmp_path):
    client, _ = make_client(tmp_path)

    chunks = list(
        client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": "Stream this"}],
            stream=True,
        )
    )

    assert len(chunks) == 3
    assert {chunk.id for chunk in chunks} == {chunks[0].id}
    assert chunks[0].object == "chat.completion.chunk"
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[0].choices[0].finish_reason is None
    assert chunks[1].choices[0].delta.content == "fake gpt-5.5"
    assert chunks[1].choices[0].finish_reason is None
    assert chunks[2].choices[0].delta == {}
    assert chunks[2].choices[0].finish_reason == "stop"


def test_chat_completion_stream_can_include_usage_chunk(tmp_path):
    client, _ = make_client(tmp_path)

    chunks = list(
        client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": "Stream this"}],
            stream=True,
            stream_options={"include_usage": True},
        )
    )

    assert chunks[-1].choices == []
    assert chunks[-1].usage.total_tokens == 7


def test_responses_stream_yields_typed_events(tmp_path):
    client, _ = make_client(tmp_path)

    events = list(
        client.responses.create(
            model="gpt-5.4-mini",
            input="Stream this",
            stream=True,
            include_obfuscation=False,
        )
    )

    assert [event.type for event in events] == [
        "response.created",
        "response.output_text.delta",
        "response.output_text.done",
        "response.completed",
    ]
    assert events[0].response.status == "in_progress"
    assert events[1].delta == "fake gpt-5.5"
    assert events[2].text == "fake gpt-5.5"
    assert events[3].response.status == "completed"


def test_content_parts_extract_file_plan_without_leaking_uri(tmp_path):
    client, _ = make_client(tmp_path, dry_run=True)

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this receipt"},
                    {"type": "image_url", "image_url": {"url": "/tmp/private/receipt.png"}},
                ],
            }
        ],
    )

    file_plan = response.crupier.route["input_plan"]["files"]
    assert file_plan["representations"][0]["representation"] == "native_vision"
    assert "private" not in str(file_plan)


def test_install_patches_openai_module_with_compat_client(tmp_path):
    client, _ = make_client(tmp_path)
    original_module = sys.modules.get("openai")
    fake_openai = SimpleNamespace(OpenAI=object)
    sys.modules["openai"] = fake_openai
    try:
        patched = install("openai", crupier=client._crupier, dry_run=False)
        response = fake_openai.OpenAI().responses.create(input="Hello", model="gpt-5.4-mini")
    finally:
        if original_module is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = original_module

    assert patched == ["openai"]
    assert response.output_text == "fake gpt-5.5"


def test_autopatched_openai_constructor_fails_loudly_on_ignored_credentials(tmp_path):
    client, _ = make_client(tmp_path)
    original_module = sys.modules.get("openai")
    fake_openai = SimpleNamespace(OpenAI=object)
    sys.modules["openai"] = fake_openai
    try:
        install("openai", crupier=client._crupier)
        with pytest.raises(CrupierConfigError, match="api_key"):
            fake_openai.OpenAI(api_key="must-not-be-ignored")
    finally:
        if original_module is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = original_module


def test_autopatch_silent_noop_reports_omitted_when_sdk_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)

    with pytest.warns(UserWarning, match=r"omitidos.*openai"):
        patched = install("openai")

    assert patched == []


def test_autopatch_install_defaults_to_openai(monkeypatch):
    fake_openai = SimpleNamespace(OpenAI=object)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    assert install() == ["openai"]
    assert fake_openai.OpenAI is not object


def test_autopatch_returns_empty_without_sdk_and_unknown_provider(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)

    with pytest.warns(UserWarning, match="desconocido"):
        assert install("desconocido") == []


def test_autopatch_client_kwargs_lose_to_call_kwargs(monkeypatch):
    calls = []

    class RecordingOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    fake_openai = SimpleNamespace(OpenAI=object)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr("crupier.compat.openai.OpenAI", RecordingOpenAI)

    install("openai", base_url="a")
    fake_openai.OpenAI(base_url="b")

    assert calls == [{"base_url": "b"}]


def test_embeddings_create_returns_openai_like_list(tmp_path):
    client, adapter = make_client(tmp_path)

    response = client.embeddings.create(model="text-embedding-3-small", input=["one", "two"], dimensions=2)

    assert response.object == "list"
    assert response.model == "openai:text-embedding-3-small"
    assert response.data[0].object == "embedding"
    assert response.data[0].embedding == [0.1, 0.2]
    assert response.data[1].embedding == [0.4, 0.5]
    assert response.usage.prompt_tokens == 7
    assert adapter.calls[-1]["model"] == "text-embedding-3-small"
    assert adapter.calls[-1]["dimensions"] == 2


def test_embeddings_rejects_known_chat_model(tmp_path):
    client, _ = make_client(tmp_path)

    try:
        client.embeddings.create(model="gpt-5.5", input="hello")
    except CrupierModelUnsupportedError as exc:
        assert "embedding" in str(exc)
    else:
        raise AssertionError("chat model should not be accepted for embeddings")


def test_embeddings_can_route_to_non_openai_provider_by_model_id(tmp_path):
    client, adapter = make_specialized_client(tmp_path)

    response = client.embeddings.create(model="qwen3-embedding", input="hola")

    assert response.model == "nan:qwen3-embedding"
    assert response.data[0].embedding == [0.9, 0.8, 0.7]
    assert response.crupier.operation == "embedding"
    assert adapter.calls[-1]["model"] == "qwen3-embedding"


def test_openai_compat_images_route_through_specialized_model(tmp_path):
    client, adapter = make_specialized_client(tmp_path)

    response = client.images.generate(
        prompt="A lighthouse",
        size="1024x1024",
        response_format="url",
        seed=42,
    )

    assert response.model == "nan:flux-2-klein"
    assert response.data[0].url == "https://example.test/image.png"
    assert response.crupier.operation == "image_generation"
    assert response.crupier.route["steps"][0]["model"] == "nan:flux-2-klein"
    assert adapter.calls[0]["payload"]["seed"] == 42

    edited = client.images.edit(image=b"source", prompt="Add fog")
    assert edited.model == "nan:flux-2-klein"
    assert adapter.calls[1]["payload"]["images"] == b"source"


def test_openai_compat_audio_and_rerank_surfaces(tmp_path):
    client, _ = make_specialized_client(tmp_path)

    audio = client.audio.speech.create(input="Hola", voice="ef_dora", response_format="mp3")
    transcript = client.audio.transcriptions.create(
        file=("short.mp3", b"audio", "audio/mpeg"),
        language="es",
        response_format="verbose_json",
    )
    reranked = client.rerank.create(
        query="capital of France",
        documents=["Berlin", "Paris"],
        top_n=1,
    )

    assert audio.read() == b"audio-bytes"
    assert list(audio.iter_bytes(chunk_size=5)) == [b"audio", b"-byte", b"s"]
    assert transcript.text == "hola mundo"
    assert transcript.model == "nan:whisper"
    assert reranked.model == "nan:rerank"
    assert reranked.results[0].index == 1


def test_compat_response_edges(tmp_path):
    response = CompatObject(answer=42)
    with pytest.raises(AttributeError, match="missing"):
        _ = response.missing

    binary = CompatBinaryResponse(b"content", crupier=CompatObject())
    destination = tmp_path / "response.bin"
    binary.stream_to_file(destination)

    assert destination.read_bytes() == b"content"


@pytest.mark.parametrize("timeout", [True, 0, "slow"])
def test_compat_client_rejects_invalid_timeout(timeout):
    with pytest.raises(TypeError, match="timeout"):
        OpenAI(crupier=SimpleNamespace(), timeout=timeout)


@pytest.mark.parametrize("max_retries", [True, -1, 1.5])
def test_compat_client_rejects_invalid_max_retries(max_retries):
    with pytest.raises(TypeError, match="max_retries"):
        OpenAI(crupier=SimpleNamespace(), max_retries=max_retries)


def test_compat_client_builds_crupier_from_config_or_project(monkeypatch):
    from_config = SimpleNamespace()
    from_project = SimpleNamespace()
    monkeypatch.setattr(Crupier, "from_config", lambda config: from_config)
    monkeypatch.setattr(Crupier, "from_project", lambda project: from_project)

    assert OpenAI(config={})._crupier is from_config
    assert OpenAI(project="project-root")._crupier is from_project


def test_compat_client_can_disable_all_request_controls():
    client = OpenAI(crupier=SimpleNamespace(), allow_request_controls=False)

    with pytest.raises(TypeError, match="'constraints'.*'dry_run'.*'mode'.*'trace'"):
        client.responses.create(
            input="hello",
            constraints={"max_calls": 1},
            dry_run=True,
            mode="fast",
            trace=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"crupier": "invalid"}, "crupier options"),
        ({"constraints": "invalid"}, "constraints must"),
        ({"crupier": {"constraints": "invalid"}}, "crupier.constraints"),
        ({"metadata": "invalid"}, "metadata must"),
    ],
)
def test_compat_client_rejects_malformed_control_payloads(kwargs, message):
    client = OpenAI(crupier=SimpleNamespace())

    with pytest.raises(TypeError, match=message):
        client.responses.create(input="hello", **kwargs)


def test_compat_client_forwards_file_root_and_operation_constraints(tmp_path):
    class SpyCrupier:
        def __init__(self):
            self.kwargs = None

        def deal(self, **kwargs):
            self.kwargs = kwargs
            return CrupierResult(output_text="ok")

    spy = SpyCrupier()
    client = OpenAI(crupier=spy, file_root=tmp_path, allow_local_file_uris=False)
    client.responses.create(input="hello")

    operation_kwargs = client._operation_kwargs(
        {"max_calls": 2, "dry_run": False, "background": True}
    )
    assert spy.kwargs["constraints"]["file_root"] == str(tmp_path)
    assert spy.kwargs["constraints"]["allow_local_file_uris"] is False
    assert operation_kwargs == {"dry_run": False, "constraints": {"max_calls": 2}}


@pytest.mark.parametrize(
    ("data", "dimensions", "message"),
    [
        ("invalid", None, "invalid vector list"),
        ([], None, "no vectors"),
        ([[0.1]], 2, "did not honor dimensions"),
    ],
)
def test_compat_embeddings_reject_malformed_provider_payloads(data, dimensions, message):
    provider = SimpleNamespace(
        embed=lambda **kwargs: OperationResult(operation="embedding", model="model", data=data)
    )
    client = OpenAI(crupier=provider)

    with pytest.raises(CrupierProviderUnavailableError, match=message):
        client.embeddings.create(model="embedding-model", input="hello", dimensions=dimensions)


def test_compat_speech_rejects_non_binary_provider_payload():
    provider = SimpleNamespace(
        synthesize=lambda **kwargs: OperationResult(operation="tts", model="voice", data="not-bytes")
    )
    client = OpenAI(crupier=provider)

    with pytest.raises(CrupierProviderUnavailableError, match="non-binary"):
        client.audio.speech.create(input="hello", voice="voice")


def test_compat_helper_edges_cover_payload_and_streaming_boundaries():
    constraints = openai_compat._compat_constraints(
        model=None,
        stream=False,
        compat_mode="balanced",
        kwargs={"max_tokens": 12, "temperature": 0.2, "top_p": 0.8},
    )
    assert constraints["max_output_tokens"] == 12
    assert constraints["temperature"] == 0.2
    assert constraints["top_p"] == 0.8

    schema = {"type": "json_schema", "json_schema": {"name": "answer"}}
    assert openai_compat._response_schema_from_format(schema) is schema
    assert openai_compat._response_schema_from_format("text") == "text"

    files = []
    normalized = openai_compat._normalize_input(
        {
            "nested": {"value": 1},
            "content": [
                "plain text",
                {"type": "image_url", "image_url": "image.png"},
                {"type": "input_file", "filename": "brief.pdf"},
            ],
        },
        files,
    )
    assert normalized["content"][0] == "plain text"
    assert [item["uri"] for item in files] == ["image.png", "brief.pdf"]
    assert openai_compat._normalize_input({"nested": {"value": 1}}, []) == {
        "nested": {"value": 1}
    }
    assert openai_compat._normalize_input([{"value": 1}], []) == [{"value": 1}]

    assert openai_compat._task_from_messages([{"role": "assistant", "content": "hello"}]) == (
        "Respond to the chat messages."
    )
    assert openai_compat._text_from_content({"text": "ignored"}) == ""
    assert list(openai_compat._text_chunks("")) == []
    assert openai_compat._looks_like_embedding_model("text-embedding-3-small")
    assert not openai_compat._looks_like_embedding_model("gpt-5.5")


def test_responses_stream_includes_obfuscation_by_default(tmp_path):
    client, _ = make_client(tmp_path)

    events = list(client.responses.create(input="hello", stream=True))

    assert events[1].obfuscation == ""
