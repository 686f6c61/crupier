from types import SimpleNamespace

from crupier.adapters.anthropic import AnthropicAdapter
from crupier.adapters.google import GoogleAdapter
from crupier.adapters.ollama import OllamaAdapter
from crupier.adapters.openai import OpenAIAdapter
from crupier.config import ProviderSettings
from crupier.models import RequestEnvelope


class FakeOpenAIResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [{"delta": "stream-ok"}]
        if kwargs.get("tools"):
            return {"output": [{"type": "function_call", "name": "crupier_probe_tool"}]}
        return {"output_text": '{"ok": true, "probe": "crupier"}'}


def test_openai_native_probe_payloads():
    responses = FakeOpenAIResponses()
    adapter = OpenAIAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(responses=responses)
    request = RequestEnvelope(task="probe", constraints={"max_output_tokens": 64})

    structured = adapter.probe_capability(model="gpt-test", probe="structured_output", request=request)
    tool = adapter.probe_capability(model="gpt-test", probe="tool_call", request=request)
    streaming = adapter.probe_capability(model="gpt-test", probe="streaming", request=request)

    assert structured.metadata["probe_status"] == "verified"
    assert responses.calls[0]["text"]["format"]["type"] == "json_schema"
    assert tool.metadata["probe_status"] == "verified"
    assert responses.calls[1]["tools"][0]["name"] == "crupier_probe_tool"
    assert streaming.metadata["probe_status"] == "verified"
    assert responses.calls[2]["stream"] is True


class FakeAnthropicMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [{"delta": {"text": "stream-ok"}}]
        return {"content": [{"type": "tool_use", "name": "crupier_probe_tool"}]}


def test_anthropic_native_probe_payloads():
    messages = FakeAnthropicMessages()
    adapter = AnthropicAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(messages=messages)
    request = RequestEnvelope(task="probe", constraints={"max_output_tokens": 64})

    structured = adapter.probe_capability(model="claude-test", probe="structured_output", request=request)
    tool = adapter.probe_capability(model="claude-test", probe="tool_call", request=request)
    streaming = adapter.probe_capability(model="claude-test", probe="streaming", request=request)

    assert structured.metadata["probe_status"] == "verified"
    assert messages.calls[0]["tool_choice"]["name"] == "crupier_probe_tool"
    assert tool.metadata["probe_status"] == "verified"
    assert messages.calls[1]["tools"][0]["input_schema"]["required"] == ["ok", "probe"]
    assert streaming.metadata["probe_status"] == "verified"
    assert messages.calls[2]["stream"] is True


class FakeGoogleModels:
    def __init__(self):
        self.generate_calls = []
        self.stream_calls = []

    def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs)
        config = kwargs.get("config")
        if isinstance(config, dict) and config.get("response_json_schema"):
            return {"text": '{"ok": true, "probe": "crupier"}'}
        return {
            "candidates": [
                {"content": {"parts": [{"function_call": {"name": "crupier_probe_tool", "args": {"ok": True}}}]}}
            ]
        }

    def generate_content_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return [{"text": "stream-ok"}]


def test_google_native_probe_payloads():
    models = FakeGoogleModels()
    adapter = GoogleAdapter(ProviderSettings(enabled=True))
    adapter._client = SimpleNamespace(models=models)
    request = RequestEnvelope(task="probe", constraints={"max_output_tokens": 64})

    structured = adapter.probe_capability(model="gemini-test", probe="structured_output", request=request)
    tool = adapter.probe_capability(model="gemini-test", probe="tool_call", request=request)
    streaming = adapter.probe_capability(model="gemini-test", probe="streaming", request=request)

    assert structured.metadata["probe_status"] == "verified"
    structured_config = models.generate_calls[0]["config"]
    structured_mime = (
        structured_config["response_mime_type"]
        if isinstance(structured_config, dict)
        else getattr(structured_config, "response_mime_type", None)
    )
    assert structured_mime == "application/json"
    assert tool.metadata["probe_status"] == "verified"
    tool_config = models.generate_calls[1]["config"]
    automatic_function_calling = (
        tool_config["automatic_function_calling"]
        if isinstance(tool_config, dict)
        else getattr(tool_config, "automatic_function_calling", None)
    )
    thinking_config = tool_config["thinking_config"] if isinstance(tool_config, dict) else getattr(tool_config, "thinking_config", None)
    thinking_level = (
        thinking_config["thinking_level"]
        if isinstance(thinking_config, dict)
        else getattr(thinking_config, "thinking_level", None)
    )
    disabled = (
        automatic_function_calling["disable"]
        if isinstance(automatic_function_calling, dict)
        else getattr(automatic_function_calling, "disable", None)
    )
    assert disabled is True
    assert str(thinking_level).lower().endswith("minimal")
    assert streaming.metadata["probe_status"] == "verified"
    assert models.stream_calls[0]["model"] == "gemini-test"


def test_ollama_native_probe_helpers(monkeypatch):
    adapter = OllamaAdapter(ProviderSettings(enabled=True, host="http://localhost:11434"))
    request = RequestEnvelope(task="probe", constraints={"max_output_tokens": 64})

    def fake_chat_json(payload, *, timeout):
        if payload.get("tools"):
            return {"message": {"tool_calls": [{"function": {"name": "crupier_probe_tool"}}]}}
        return {"message": {"content": '{"ok": true, "probe": "crupier"}'}}

    def fake_chat_stream(payload, *, timeout):
        yield {"message": {"content": "stream-ok"}}

    monkeypatch.setattr(adapter, "_chat_json", fake_chat_json)
    monkeypatch.setattr(adapter, "_chat_stream", fake_chat_stream)

    structured = adapter.probe_capability(model="llama-test", probe="structured_output", request=request)
    tool = adapter.probe_capability(model="llama-test", probe="tool_call", request=request)
    streaming = adapter.probe_capability(model="llama-test", probe="streaming", request=request)

    assert structured.metadata["probe_status"] == "verified"
    assert tool.metadata["probe_status"] == "verified"
    assert streaming.metadata["probe_status"] == "verified"
