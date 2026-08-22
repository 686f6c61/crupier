import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

import crupier.adapters.anthropic as anthropic_module
import crupier.adapters.openai as openai_module
from crupier.adapters.anthropic import AnthropicAdapter
from crupier.adapters.nan import NaNAdapter
from crupier.adapters.ollama import OllamaAdapter
from crupier.adapters.openai import OpenAIAdapter
from crupier.adapters.openai_compatible import OpenAICompatibleAdapter
from crupier.adapters.openrouter import OpenRouterAdapter
from crupier.config import CrupierConfig, ProviderSettings
from crupier.errors import (
    CrupierConfigError,
    CrupierProviderAuthError,
    CrupierProviderRateLimitError,
    CrupierProviderUnavailableError,
)
from crupier.models import RequestEnvelope


class RecordingResponses:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or SimpleNamespace(output_text="ok", usage={"total_tokens": 1})

    def create(self, **payload):
        self.calls.append(payload)
        return self.response


def test_openai_supports_only_native_image_and_pdf():
    assert OpenAIAdapter.supports_file_kind(model="any", kind="image") is True
    assert OpenAIAdapter.supports_file_kind(model="any", kind="pdf") is True
    assert OpenAIAdapter.supports_file_kind(model="any", kind="audio") is False


def test_openai_generate_builds_prompt_and_honors_output_constraints():
    responses = RecordingResponses()
    adapter = OpenAIAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(responses=responses)
    schema = {"type": "object"}

    result = adapter.generate(
        model="gpt-test",
        prompt="",
        request=RequestEnvelope(
            task="Return data",
            input="record",
            response_schema=schema,
            constraints={"max_output_tokens": 77, "strict_response_schema": False},
        ),
    )

    payload = responses.calls[0]
    assert payload["input"] == "Task:\nReturn data\n\nInput:\nrecord"
    assert payload["max_output_tokens"] == 77
    assert payload["text"]["format"]["name"] == "crupier_response"
    assert payload["text"]["format"]["strict"] is False
    assert result.usage == {"total_tokens": 1}


def test_openai_list_models_supports_objects_skips_missing_ids_and_maps_error():
    adapter = OpenAIAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[SimpleNamespace(id="z-model", input_tokens=1), SimpleNamespace(name="missing"), {"id": "a-model"}]
            )
        )
    )

    models = adapter.list_models()

    assert [item.id for item in models] == ["a-model", "z-model"]
    assert models[1].metadata == {"input_tokens": 1}

    class RateLimitError(Exception):
        pass

    adapter._client = SimpleNamespace(models=SimpleNamespace(list=lambda: (_ for _ in ()).throw(RateLimitError("slow"))))
    with pytest.raises(CrupierProviderRateLimitError, match="slow"):
        adapter.list_models()


def test_openai_embeddings_validate_dimensions_and_parse_object_rows():
    class Embeddings:
        def __init__(self):
            self.payload = None

        def create(self, **payload):
            self.payload = payload
            return SimpleNamespace(
                data=[{"embedding": [1, 2]}, SimpleNamespace(embedding=[3.5, 4]), {"missing": True}],
                usage=SimpleNamespace(total_tokens=9),
            )

    embeddings = Embeddings()
    adapter = OpenAIAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(embeddings=embeddings)

    result = adapter.embed(model="embed-test", input=["a", "b"], dimensions=2)

    assert embeddings.payload == {"model": "embed-test", "input": ["a", "b"], "dimensions": 2}
    assert result.embeddings == [[1.0, 2.0], [3.5, 4.0]]
    assert result.usage == {"total_tokens": 9}
    with pytest.raises(CrupierProviderUnavailableError, match="must be positive"):
        adapter.embed(model="embed-test", input="x", dimensions=0)


def test_openai_embeddings_map_provider_failures():
    class AuthenticationError(Exception):
        pass

    adapter = OpenAIAdapter(ProviderSettings(enabled=True, env_key="OPENAI_TEST_KEY"))
    adapter._client = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **payload: (_ for _ in ()).throw(AuthenticationError("bad key")))
    )

    with pytest.raises(CrupierProviderAuthError) as exc_info:
        adapter.embed(model="embed-test", input="x")
    assert exc_info.value.env_key == "OPENAI_TEST_KEY"


def test_openai_probe_failures_timeouts_and_stream_limit():
    class ProbeResponses:
        def __init__(self):
            self.calls = []

        def create(self, **payload):
            self.calls.append(payload)
            if payload.get("stream"):
                return [{"text": "stream-ok"}, *({} for _ in range(25))]
            if payload.get("tools"):
                return {"output": [{"type": "function_call", "name": "other"}]}
            return {"output_text": "not-json"}

    responses = ProbeResponses()
    adapter = OpenAIAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(responses=responses)
    request = RequestEnvelope(task="probe", constraints={"timeout": 3})

    structured = adapter.probe_capability(model="gpt-test", probe="structured_output", request=request)
    tool = adapter.probe_capability(model="gpt-test", probe="tool_call", request=request)
    streaming = adapter.probe_capability(model="gpt-test", probe="streaming", request=request)

    assert structured.metadata["probe_status"] == "failed"
    assert tool.metadata["probe_status"] == "failed"
    assert streaming.metadata["probe_status"] == "verified"
    assert streaming.metadata["event_count"] == 20
    assert all(call["timeout"] == 3 for call in responses.calls)
    with pytest.raises(NotImplementedError, match="no native probe"):
        adapter.probe_capability(model="gpt-test", probe="vision", request=request)


def test_openai_probe_and_param_repair_map_errors():
    class PermissionDeniedError(Exception):
        pass

    adapter = OpenAIAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **payload: (_ for _ in ()).throw(PermissionDeniedError("denied"))
        )
    )

    for probe in ["structured_output", "tool_call", "streaming"]:
        with pytest.raises(CrupierProviderAuthError, match="denied"):
            adapter.probe_capability(model="gpt-test", probe=probe, request=RequestEnvelope(task="x"))

    class RepairFails:
        def create(self, **payload):
            if "temperature" in payload:
                raise RuntimeError("Unsupported parameter: 'temperature'")
            raise RuntimeError("still broken")

    adapter._client = SimpleNamespace(responses=RepairFails())
    with (
        pytest.warns(RuntimeWarning, match="temperature"),
        pytest.raises(CrupierProviderUnavailableError, match="still broken"),
    ):
        adapter.generate(
            model="gpt-test",
            prompt="x",
            request=RequestEnvelope(task="x", constraints={"temperature": 0}),
        )


@pytest.mark.parametrize(
    ("unsupported", "constraints"),
    [
        ("text", {"response_schema": {"type": "object"}}),
        ("max_output_tokens", {"max_output_tokens": 64}),
        ("timeout", {"timeout_seconds": 2}),
    ],
)
def test_openai_param_repair_rejects_removal_of_schema_budget_and_timeout_controls(
    unsupported, constraints
):
    class RejectControl:
        def __init__(self):
            self.payloads = []

        def create(self, **payload):
            self.payloads.append(payload)
            raise RuntimeError(f"Unsupported parameter: '{unsupported}'")

    responses = RejectControl()
    adapter = OpenAIAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(responses=responses)

    with pytest.raises(CrupierProviderUnavailableError, match=unsupported):
        adapter.generate(
            model="gpt-test",
            prompt="x",
            request=RequestEnvelope(task="x", constraints=constraints),
        )

    assert len(responses.payloads) == 1


def test_openai_build_client_supports_host_and_reports_missing_dependency(monkeypatch):
    calls = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    adapter = OpenAIAdapter(
        ProviderSettings(
            enabled=True,
            host="https://api.example.test/v1",
            options={"allow_custom_host": True},
        )
    )
    adapter._build_client()
    assert calls["base_url"] == "https://api.example.test/v1"

    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(CrupierProviderUnavailableError, match="optional dependency"):
        adapter._build_client()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (Exception("nothing relevant"), None),
        (Exception('unsupported request {"param": "temperature"}'), "temperature"),
        (Exception("not supported {'param': 'max_tokens'}"), "max_tokens"),
        (Exception("unsupported but unnamed"), None),
    ],
)
def test_openai_unsupported_parameter_patterns(error, expected):
    assert openai_module._unsupported_parameter(error) == expected


def test_openai_probe_helpers_handle_wrapped_invalid_and_object_events():
    assert openai_module._json_probe_ok('prefix {"ok":true,"probe":"crupier"} suffix') is True
    assert openai_module._json_probe_ok("no object") is False
    assert openai_module._json_probe_ok("prefix {invalid} suffix") is False
    assert openai_module._openai_has_tool_call(
        {"output": [SimpleNamespace(type="tool_call", name="target")]}, "target"
    ) is True
    assert openai_module._openai_has_tool_call({"output": None}, "target") is False
    assert openai_module._event_has_text(SimpleNamespace(delta="x")) is True
    assert openai_module._event_has_text(SimpleNamespace()) is False


def test_anthropic_supports_only_native_images():
    assert AnthropicAdapter.supports_file_kind(model="any", kind="image") is True
    assert AnthropicAdapter.supports_file_kind(model="any", kind="pdf") is False


def test_anthropic_generate_builds_prompt_and_honors_alias_constraints():
    messages = RecordingResponses(SimpleNamespace(content=[{"type": "text", "text": "ok"}], usage={}))
    adapter = AnthropicAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(messages=messages)

    result = adapter.generate(
        model="claude-test",
        prompt="",
        request=RequestEnvelope(task="Explain", input="record", constraints={"max_tokens": 33, "temperature": 0.2}),
    )

    payload = messages.calls[0]
    assert payload["max_tokens"] == 33
    assert payload["temperature"] == 0.2
    assert payload["messages"][0]["content"] == "Task:\nExplain\n\nInput:\nrecord"
    assert result.text == "ok"


def test_anthropic_list_models_supports_objects_skips_missing_and_sdk_without_catalog():
    adapter = AnthropicAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[SimpleNamespace(id="claude-z", display_name="Z"), {"display_name": "missing"}, {"id": "claude-a"}]
            )
        )
    )

    models = adapter.list_models()
    assert [item.id for item in models] == ["claude-a", "claude-z"]
    assert models[1].name == "Z"

    adapter._client = SimpleNamespace()
    with pytest.raises(CrupierProviderUnavailableError, match="does not expose"):
        adapter.list_models()


def test_anthropic_list_models_maps_provider_error():
    class RateLimitError(Exception):
        pass

    adapter = AnthropicAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(
        models=SimpleNamespace(list=lambda: (_ for _ in ()).throw(RateLimitError("slow")))
    )
    with pytest.raises(CrupierProviderRateLimitError, match="slow"):
        adapter.list_models()


def test_anthropic_probe_failures_timeouts_and_stream_limit():
    class ProbeMessages:
        def __init__(self):
            self.calls = []

        def create(self, **payload):
            self.calls.append(payload)
            if payload.get("stream"):
                return [SimpleNamespace(text="stream-ok"), *(SimpleNamespace() for _ in range(25))]
            return {"content": [{"type": "tool_use", "name": "other"}]}

    messages = ProbeMessages()
    adapter = AnthropicAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(messages=messages)
    request = RequestEnvelope(task="probe", constraints={"timeout": 4})

    structured = adapter.probe_capability(model="claude-test", probe="structured_output", request=request)
    tool = adapter.probe_capability(model="claude-test", probe="tool_call", request=request)
    stream = adapter.probe_capability(model="claude-test", probe="streaming", request=request)

    assert structured.metadata["probe_status"] == "failed"
    assert tool.metadata["probe_status"] == "failed"
    assert stream.metadata["event_count"] == 20
    assert all(call["timeout"] == 4 for call in messages.calls)
    with pytest.raises(NotImplementedError, match="no native probe"):
        adapter.probe_capability(model="claude-test", probe="vision", request=request)


def test_anthropic_probe_and_param_repair_map_errors():
    class AuthenticationError(Exception):
        pass

    adapter = AnthropicAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **payload: (_ for _ in ()).throw(AuthenticationError("bad key"))
        )
    )
    for probe in ["structured_output", "tool_call", "streaming"]:
        with pytest.raises(CrupierProviderAuthError, match="bad key"):
            adapter.probe_capability(model="claude-test", probe=probe, request=RequestEnvelope(task="x"))

    class RepairFails:
        def create(self, **payload):
            if "temperature" in payload:
                raise RuntimeError("temperature is deprecated")
            raise RuntimeError("still broken")

    adapter._client = SimpleNamespace(messages=RepairFails())
    with pytest.raises(CrupierProviderUnavailableError, match="still broken"):
        adapter.generate(
            model="claude-test",
            prompt="x",
            request=RequestEnvelope(task="x", constraints={"temperature": 0}),
        )

    adapter._client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **payload: (_ for _ in ()).throw(Exception("plain failure")))
    )
    with pytest.raises(CrupierProviderUnavailableError, match="plain failure"):
        adapter.generate(model="claude-test", prompt="x", request=RequestEnvelope(task="x"))


def test_anthropic_build_client_supports_host_timeout_and_missing_dependency(monkeypatch):
    calls = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    adapter = AnthropicAdapter(
        ProviderSettings(
            enabled=True,
            host="https://api.example.test",
            options={"timeout": 5, "allow_custom_host": True},
        )
    )
    adapter._build_client()
    assert calls == {"api_key": "test-key", "base_url": "https://api.example.test", "timeout": 5.0}

    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(CrupierProviderUnavailableError, match="optional dependency"):
        adapter._build_client()


def test_anthropic_probe_helpers_support_objects_and_empty_content():
    assert anthropic_module._is_temperature_deprecated(Exception("TEMPERATURE is DEPRECATED")) is True
    assert anthropic_module._is_temperature_deprecated(Exception("temperature invalid")) is False
    message = SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="target")])
    assert anthropic_module._anthropic_has_tool_use(message, "target") is True
    assert anthropic_module._anthropic_has_tool_use(SimpleNamespace(content=None), "target") is False
    assert anthropic_module._event_has_text({"content_block": {"type": "text"}}) is True
    assert anthropic_module._event_has_text({}) is False
    assert anthropic_module._event_has_text(SimpleNamespace(content_block="x")) is True
    assert anthropic_module._event_has_text(SimpleNamespace()) is False


def _recording_sdk(monkeypatch, module_name: str, class_name: str) -> list[dict]:
    constructed: list[dict] = []

    class FakeClient:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(**{class_name: FakeClient}))
    return constructed


@pytest.mark.parametrize(
    ("adapter_cls", "env_key", "sdk_module", "sdk_attr"),
    [
        (OpenAIAdapter, "OPENAI_API_KEY", "openai", "OpenAI"),
        (AnthropicAdapter, "ANTHROPIC_API_KEY", "anthropic", "Anthropic"),
        (OpenRouterAdapter, "OPENROUTER_API_KEY", "openai", "OpenAI"),
        (NaNAdapter, "NAN_API_KEY", "openai", "OpenAI"),
        (OllamaAdapter, "OLLAMA_API_KEY", None, None),
    ],
)
def test_canonical_provider_rejects_custom_host_with_default_credential(
    monkeypatch, adapter_cls, env_key, sdk_module, sdk_attr
):
    monkeypatch.setenv(env_key, "canonical-secret")
    settings = ProviderSettings(enabled=True, host="https://attacker.example/v1")
    adapter = adapter_cls(settings)
    outbound = []
    if sdk_module:
        constructed = _recording_sdk(monkeypatch, sdk_module, sdk_attr)
    else:

        def _blocked_urlopen(*args, **kwargs):
            outbound.append((args, kwargs))
            raise AssertionError("custom host must not receive the canonical credential")

        monkeypatch.setattr(urllib.request, "urlopen", _blocked_urlopen)
        constructed = outbound

    with pytest.raises(CrupierConfigError, match="allow_custom_host"):
        if adapter_cls is OllamaAdapter:
            adapter.generate(model="llama", prompt="ping", request=RequestEnvelope(task="ping"))
        else:
            adapter._build_client()

    assert constructed == []


def test_generic_inference_host_uses_only_its_configured_env_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "canonical-openai-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "canonical-anthropic-secret")
    monkeypatch.setenv("NAN_API_KEY", "canonical-nan-secret")
    monkeypatch.setenv("INFERENCE_API_KEY", "inference-only-secret")
    captured = _recording_sdk(monkeypatch, "openai", "OpenAI")

    adapter = OpenAICompatibleAdapter(
        ProviderSettings(enabled=True, host="https://inference.example/v1"),
        provider="inference",
    )
    adapter._build_client()

    assert captured[0]["api_key"] == "inference-only-secret"
    assert captured[0]["api_key"] not in {
        "canonical-openai-secret",
        "canonical-anthropic-secret",
        "canonical-nan-secret",
    }

    with pytest.raises(CrupierConfigError, match="canonical"):
        OpenAICompatibleAdapter(
            ProviderSettings(
                enabled=True,
                host="https://inference.example/v1",
                env_key="OPENAI_API_KEY",
            ),
            provider="inference",
        )._build_client()


def test_untrusted_project_cannot_redirect_provider_traffic_through_proxy(
    tmp_path: Path, monkeypatch
) -> None:
    attacker_proxy = "http://attacker.invalid:8080"
    (tmp_path / "crupier.toml").write_text(
        "[providers.openai]\nenabled = true\nenv_key = \"OPENAI_API_KEY\"\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        f"OPENAI_API_KEY=local-key\nHTTPS_PROXY={attacker_proxy}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("HTTPS_PROXY", raising=False)

    with pytest.warns(UserWarning, match="HTTPS_PROXY"):
        CrupierConfig.from_toml(tmp_path)

    assert all("attacker.invalid" not in value for value in urllib.request.getproxies().values())
