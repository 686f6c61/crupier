import sys
from types import SimpleNamespace

from crupier import Crupier, install
from crupier.adapters import AdapterResponse, EmbeddingResponse, OperationResponse
from crupier.compat.openai import OpenAI
from crupier.config import CrupierConfig
from crupier.errors import CrupierModelUnsupportedError


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
