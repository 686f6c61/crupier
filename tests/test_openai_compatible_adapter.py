from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from crupier.adapters import OpenAICompatibleAdapter
from crupier.adapters.factory import build_default_adapters
import crupier.adapters.openai_compatible as compatible
from crupier.config import CrupierConfig, ProviderSettings
from crupier.errors import (
    CrupierProviderAuthError,
    CrupierProviderRateLimitError,
    CrupierProviderUnavailableError,
)
from crupier.models import FileAsset, RequestEnvelope


class _Create:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[dict] = []

    def create(self, **payload):
        self.calls.append(payload)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _client(*, chat=None, models=None, embeddings=None):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=_Create(chat or {"choices": [{"message": {"content": "ok"}}]})),
        models=SimpleNamespace(list=lambda: models or {"data": []}),
        embeddings=_Create(embeddings or {"data": []}),
    )


def test_configurable_inference_generate_supports_schema_limits_timeout_and_images(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"image")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=[{"text": "hello"}, {"text": " world"}]))],
        usage=SimpleNamespace(total_tokens=3),
    )
    client = _client(chat=response)
    adapter = OpenAICompatibleAdapter(
        ProviderSettings(
            enabled=True,
            options={
                "thinking_control": "chat_template_kwargs",
                "extra_body": {"server_option": True},
            },
        ),
        provider="inference",
    )
    adapter._client = client
    schema = {"type": "object"}
    request = RequestEnvelope(
        task="describe",
        files=[FileAsset(kind="image", name="sample.png", uri=str(image), mime_type="image/png")],
        response_schema=schema,
        constraints={
            "max_tokens": "42",
            "temperature": 0.2,
            "top_p": 0.8,
            "timeout_seconds": 7,
            "response_schema_name": "result",
            "strict_response_schema": False,
            "disable_thinking": True,
            "extra_body": {"request_option": 1},
        },
    )

    result = adapter.generate(model="vision-model", prompt="describe", request=request)

    payload = client.chat.completions.calls[0]
    assert payload["max_tokens"] == 42
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.8
    assert payload["timeout"] == 7
    assert payload["response_format"]["json_schema"]["name"] == "result"
    assert payload["response_format"]["json_schema"]["strict"] is False
    assert payload["extra_body"] == {
        "server_option": True,
        "request_option": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert payload["messages"][0]["content"][1]["type"] == "image_url"
    assert result.text == "hello world"
    assert result.usage == {"total_tokens": 3}
    assert result.metadata["multimodal_images"] == 1


def test_configurable_inference_plain_generate_and_error_mapping() -> None:
    adapter = OpenAICompatibleAdapter(ProviderSettings(enabled=True))
    client = _client(chat={"choices": [{"message": {"content": "plain"}}]})
    adapter._client = client

    result = adapter.generate(model="chat", prompt="hello", request=RequestEnvelope(task="hello"))
    assert result.text == "plain"
    assert client.chat.completions.calls[0]["messages"] == [{"role": "user", "content": "hello"}]

    adapter._client = _client(chat=RuntimeError("offline"))
    with pytest.raises(CrupierProviderUnavailableError, match="inference: offline"):
        adapter.generate(model="chat", prompt="hello", request=RequestEnvelope(task="hello"))


def test_configurable_inference_lists_models_and_maps_discovery_errors() -> None:
    adapter = OpenAICompatibleAdapter(ProviderSettings(enabled=True), provider="edge")
    adapter._client = _client(
        models=SimpleNamespace(data=[SimpleNamespace(id="z", owner="x"), {"id": "a"}, {"name": "skip"}])
    )

    models = adapter.list_models()

    assert [item.model_ref for item in models] == ["edge:a", "edge:z"]
    assert models[1].metadata == {}

    class RateLimitError(Exception):
        pass

    adapter._client = SimpleNamespace(
        models=SimpleNamespace(list=lambda: (_ for _ in ()).throw(RateLimitError("slow")))
    )
    with pytest.raises(CrupierProviderRateLimitError):
        adapter.list_models()


def test_configurable_inference_embeddings_validate_parse_and_map_errors() -> None:
    adapter = OpenAICompatibleAdapter(ProviderSettings(enabled=True))
    with pytest.raises(CrupierProviderUnavailableError, match="positive"):
        adapter.embed(model="embed", input="x", dimensions=0)

    embeddings = SimpleNamespace(
        data=[{"embedding": [1, 2]}, SimpleNamespace(embedding=[3, 4]), {"missing": True}],
        usage=SimpleNamespace(total_tokens=2),
    )
    client = _client(embeddings=embeddings)
    adapter._client = client
    result = adapter.embed(model="embed", input=["x"], dimensions=4)
    assert result.embeddings == [[1.0, 2.0], [3.0, 4.0]]
    assert client.embeddings.calls[0]["dimensions"] == 4

    adapter._client = _client(embeddings=RuntimeError("offline"))
    with pytest.raises(CrupierProviderUnavailableError, match="offline"):
        adapter.embed(model="embed", input="x")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"ok": true, "probe": "crupier"}', True),
        ("not-json", False),
    ],
)
def test_configurable_inference_structured_probe(text: str, expected: bool) -> None:
    adapter = OpenAICompatibleAdapter(ProviderSettings(enabled=True))
    adapter._client = _client(chat={"choices": [{"message": {"content": text}}]})

    result = adapter.probe_capability(
        model="chat",
        probe="structured_output",
        request=RequestEnvelope(task="probe"),
    )

    assert result.metadata["ok"] is expected
    assert result.metadata["probe_status"] == ("verified" if expected else "failed")


def test_configurable_inference_tool_probe_supports_dict_and_object_calls() -> None:
    adapter = OpenAICompatibleAdapter(ProviderSettings(enabled=True))
    adapter._client = _client(
        chat={
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "crupier_probe_tool"}},
                            SimpleNamespace(function=SimpleNamespace(name="other")),
                        ]
                    }
                }
            ]
        }
    )

    result = adapter.probe_capability(
        model="chat",
        probe="tool_call",
        request=RequestEnvelope(task="probe", constraints={"timeout_seconds": 2}),
    )

    assert result.metadata["ok"] is True
    assert adapter._client.chat.completions.calls[0]["timeout"] == 2


def test_configurable_inference_stream_probe_and_unknown_probe() -> None:
    stream = [{"choices": [{"delta": {"content": "stream"}}]}] + [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="-ok"))])
        for _ in range(20)
    ]
    adapter = OpenAICompatibleAdapter(ProviderSettings(enabled=True))
    adapter._client = _client(chat=stream)

    result = adapter.probe_capability(
        model="chat",
        probe="streaming",
        request=RequestEnvelope(task="probe", constraints={"timeout_seconds": 3}),
    )

    assert result.metadata["ok"] is True
    assert result.metadata["event_count"] == 20
    with pytest.raises(NotImplementedError, match="no native probe"):
        adapter.probe_capability(model="chat", probe="unknown", request=RequestEnvelope(task="probe"))


def test_configurable_inference_probe_errors_are_mapped() -> None:
    adapter = OpenAICompatibleAdapter(ProviderSettings(enabled=True))
    adapter._client = _client(chat=RuntimeError("offline"))
    for probe in ("tool_call", "streaming"):
        with pytest.raises(CrupierProviderUnavailableError, match="offline"):
            adapter.probe_capability(model="chat", probe=probe, request=RequestEnvelope(task="probe"))


def test_configurable_inference_client_builds_local_and_remote_clients(monkeypatch) -> None:
    captured = []

    def fake_openai(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fake_openai))
    local = OpenAICompatibleAdapter(
        ProviderSettings(
            enabled=True,
            host="http://127.0.0.1:9000/v1",
            mode="openai_compatible",
            options={"auth": "none", "timeout_seconds": 4},
        )
    )
    local._build_client()
    assert captured[0] == {
        "api_key": "crupier-local",
        "base_url": "http://127.0.0.1:9000/v1",
        "timeout": 4.0,
    }

    remote = OpenAICompatibleAdapter(
        ProviderSettings(enabled=True, host="https://inference.example/v1", env_key="CUSTOM_KEY")
    )
    monkeypatch.setenv("CUSTOM_KEY", "secret")
    remote._build_client()
    assert captured[1]["api_key"] == "secret"


def test_configurable_inference_client_requires_remote_key_and_optional_dependency(monkeypatch) -> None:
    remote = OpenAICompatibleAdapter(ProviderSettings(enabled=True, host="https://inference.example/v1"))
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    with pytest.raises(CrupierProviderAuthError) as missing:
        remote._build_client()
    assert missing.value.env_key == "INFERENCE_API_KEY"

    local = OpenAICompatibleAdapter(
        ProviderSettings(enabled=True, options={"auth": "none"})
    )
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(CrupierProviderUnavailableError, match="inference-server"):
        local._build_client()


def test_configurable_inference_maps_auth_rate_and_generic_errors() -> None:
    adapter = OpenAICompatibleAdapter(ProviderSettings(enabled=True, env_key="KEY"), provider="edge")

    class AuthenticationError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    with pytest.raises(CrupierProviderAuthError) as auth:
        adapter._raise_mapped_error(AuthenticationError("bad key"))
    assert auth.value.provider == "edge"
    assert auth.value.env_key == "KEY"
    with pytest.raises(CrupierProviderRateLimitError):
        adapter._raise_mapped_error(RateLimitError("slow"))
    with pytest.raises(CrupierProviderUnavailableError, match="edge: offline"):
        adapter._raise_mapped_error(RuntimeError("offline"))


def test_configurable_inference_factory_supports_canonical_and_custom_provider_names() -> None:
    config = CrupierConfig.from_dict(
        {
            "providers": {
                "inference": {
                    "enabled": True,
                    "mode": "openai_compatible",
                    "host": "http://127.0.0.1:8000/v1",
                    "auth": "none",
                },
                "edge": {
                    "enabled": True,
                    "mode": "openai_compatible",
                    "host": "http://localhost:9000/v1",
                    "auth": "none",
                },
            }
        }
    )

    adapters = build_default_adapters(config)

    assert isinstance(adapters["inference"], OpenAICompatibleAdapter)
    assert adapters["inference"].provider == "inference"
    assert adapters["edge"].provider == "edge"


def test_configurable_inference_helpers_cover_empty_and_mixed_content() -> None:
    assert OpenAICompatibleAdapter.supports_file_kind(model="x", kind="image") is True
    assert OpenAICompatibleAdapter.supports_file_kind(model="x", kind="pdf") is False
    assert compatible._is_loopback_host("http://localhost:1/v1") is True
    assert compatible._is_loopback_host("https://example.test/v1") is False
    assert compatible._first_message({}) is None
    assert compatible._first_delta({}) is None
    assert compatible._message_text(None) == ""
    assert compatible._message_text({"content": [{"text": "a"}, SimpleNamespace(text="b"), {}]}) == "ab"
    assert compatible._message_text({"content": 4}) == "4"
