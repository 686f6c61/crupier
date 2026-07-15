from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from crupier.adapters import ProviderModel
from crupier.adapters.common import (
    build_prompt,
    env_value,
    extract_anthropic_text,
    extract_openai_text,
    object_to_dict,
    provider_timeout_seconds,
    request_timeout_seconds,
    require_api_key,
)
from crupier.config import ProviderSettings
from crupier.errors import CrupierProviderAuthError
from crupier.models import RequestEnvelope


def test_provider_environment_helpers_use_custom_key_and_structured_error(monkeypatch):
    settings = ProviderSettings(enabled=True, env_key="CUSTOM_PROVIDER_KEY")
    monkeypatch.delenv("CUSTOM_PROVIDER_KEY", raising=False)

    assert env_value(settings, "DEFAULT_KEY", provider="custom") is None
    with pytest.raises(CrupierProviderAuthError) as exc_info:
        require_api_key(settings, "DEFAULT_KEY", provider="custom")

    error = exc_info.value
    assert error.provider == "custom"
    assert error.env_key == "CUSTOM_PROVIDER_KEY"
    assert "providers.custom" in error.hint

    monkeypatch.setenv("CUSTOM_PROVIDER_KEY", "secret")
    assert env_value(settings, "DEFAULT_KEY", provider="custom") == "secret"
    assert require_api_key(settings, "DEFAULT_KEY", provider="custom") == "secret"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("2.5", 2.5), (0, None), (-1, None), ("invalid", None), (object(), None)],
)
def test_timeout_helpers_normalize_only_positive_numbers(value, expected):
    settings = ProviderSettings(enabled=True, options={"timeout": value})
    request = RequestEnvelope(task="x", constraints={"timeout": value})

    assert provider_timeout_seconds(settings) == expected
    assert request_timeout_seconds(request) == expected


def test_timeout_helpers_honor_specific_key_and_default():
    settings = ProviderSettings(enabled=True, options={"timeout": 3, "timeout_seconds": 4})
    request = RequestEnvelope(task="x", constraints={"timeout": 5, "timeout_seconds": 6})

    assert provider_timeout_seconds(settings, default=7) == 4
    assert request_timeout_seconds(request, default=8) == 6
    assert provider_timeout_seconds(ProviderSettings(enabled=True), default=7) == 7
    assert request_timeout_seconds(RequestEnvelope(task="x"), default=8) == 8


def test_build_prompt_formats_messages_input_file_context_and_extra():
    request = RequestEnvelope(
        task="Summarize",
        messages=[{"role": "user", "content": "hello"}],
        input={"z": 1, "a": "á"},
        metadata={"extracted_file_context": {"body": "document text"}},
    )

    prompt = build_prompt(request, extra="Return briefly")

    assert prompt.startswith("Task:\nSummarize")
    assert "Messages:" in prompt and '"role": "user"' in prompt
    assert '"a": "á"' in prompt
    assert "File context:\ndocument text" in prompt
    assert prompt.endswith("Return briefly")


def test_build_prompt_falls_back_to_repr_for_non_json_input():
    value = {"opaque": object()}
    prompt = build_prompt(RequestEnvelope(task="Inspect", input=value))

    assert "<object object at " in prompt


def test_extract_openai_text_supports_sdk_dict_and_nested_shapes():
    assert extract_openai_text(SimpleNamespace(output_text="direct")) == "direct"
    assert extract_openai_text({"output_text": "dict-direct"}) == "dict-direct"
    assert extract_openai_text(
        {
            "output": [
                {"content": [{"text": "one"}, {"output_text": " two"}, {"ignored": True}]},
                SimpleNamespace(content=[SimpleNamespace(text=" three")]),
            ]
        }
    ) == "one two three"
    assert extract_openai_text(SimpleNamespace(output=None)) == ""


def test_extract_anthropic_text_supports_sdk_and_dict_blocks():
    assert extract_anthropic_text(
        {
            "content": [
                {"type": "text", "text": "one"},
                {"type": "tool_use", "text": "ignored"},
                SimpleNamespace(text=" two"),
            ]
        }
    ) == "one two"
    assert extract_anthropic_text(SimpleNamespace(content=None)) == ""


def test_object_to_dict_supports_common_sdk_usage_shapes():
    class Modern:
        def model_dump(self):
            return {"input_tokens": 1}

    class Legacy:
        def to_dict(self):
            return {"output_tokens": 2}

    attrs = SimpleNamespace(input_tokens=3, output_tokens=4, total_tokens=7)

    assert object_to_dict(None) == {}
    assert object_to_dict({"total_tokens": 1}) == {"total_tokens": 1}
    assert object_to_dict(Modern()) == {"input_tokens": 1}
    assert object_to_dict(Legacy()) == {"output_tokens": 2}
    assert object_to_dict(attrs) == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
    assert object_to_dict(object()) == {}


def test_provider_model_serializes_nested_non_json_metadata():
    class Modern:
        def model_dump(self):
            return {"created": date(2026, 7, 15)}

    class Legacy:
        def to_dict(self):
            return {"at": datetime(2026, 7, 15, 12, tzinfo=timezone.utc)}

    model = ProviderModel(
        id="model-1",
        provider="custom",
        metadata={
            "modern": Modern(),
            "legacy": Legacy(),
            "tags": {"b", "a"},
            "values": (1, 2),
            "opaque": object(),
        },
    )

    data = model.to_dict()

    assert data["model_ref"] == "custom:model-1"
    assert data["metadata"]["modern"] == {"created": "2026-07-15"}
    assert data["metadata"]["legacy"] == {"at": "2026-07-15T12:00:00+00:00"}
    assert data["metadata"]["tags"] == ["a", "b"]
    assert data["metadata"]["values"] == [1, 2]
    assert data["metadata"]["opaque"].startswith("<object object at ")
