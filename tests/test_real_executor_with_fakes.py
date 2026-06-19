from crupier import Crupier
from crupier.adapters import AdapterResponse
from crupier.config import CrupierConfig
from crupier.errors import (
    CrupierBudgetExceededError,
    CrupierProviderRateLimitError,
    CrupierProviderUnavailableError,
    CrupierToolApprovalRequired,
)


class FakeAdapter:
    provider = "openai"

    def __init__(self, provider="openai", *, fail=False, failures=None, outputs=None):
        self.provider = provider
        self.fail = fail
        self.failures = list(failures or [])
        self.outputs = list(outputs or [])
        self.calls = []

    def generate(self, *, model, prompt, request):
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "mode": request.mode,
                "response_schema": request.response_schema,
                "constraints": dict(request.constraints),
            }
        )
        if self.failures:
            raise self.failures.pop(0)
        if self.fail:
            raise CrupierProviderUnavailableError(f"{self.provider} failed")
        text = self.outputs.pop(0) if self.outputs else f"{self.provider}:{model}: ok"
        return AdapterResponse(
            text=text,
            usage={"input_tokens": 1, "output_tokens": 2},
            metadata={"provider": self.provider, "model": model},
        )


def make_config(tmp_path, *, allow, providers=None, strategy="single"):
    provider_config = providers or {
        "openai": {"enabled": True, "env_key": "OPENAI_API_KEY"},
        "anthropic": {"enabled": True, "env_key": "ANTHROPIC_API_KEY"},
        "ollama": {"enabled": True, "host": "http://localhost:11434"},
    }
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "test", "default_profile": "agentic"},
            "providers": provider_config,
            "models": {"allow": allow},
            "routing": {
                "default_strategy": strategy,
                "allow_fusion": True,
                "allow_parallel": True,
                "retry_backoff_seconds": 0.0,
            },
            "profiles": {
                "agentic": {"prefer": ["tool_use"], "strategy": strategy},
                "research": {"prefer": ["consensus"], "strategy": "fusion"},
            },
        }
    )
    config.root = tmp_path
    return config


def test_real_single_uses_injected_adapter(tmp_path):
    adapter = FakeAdapter("openai")
    client = Crupier(
        make_config(tmp_path, allow=["openai:gpt-5.4-mini"]),
        adapters={"openai": adapter},
    )

    result = client.deal("Say hi", dry_run=False, trace="summary")

    assert result.output_text == "openai:gpt-5.4-mini: ok"
    assert result.provider_metadata["dry_run"] is False
    assert adapter.calls[0]["model"] == "gpt-5.4-mini"
    assert result.trace is not None
    assert result.trace.provider_calls[0]["provider"] == "openai"
    assert result.cost.actual_usd is not None


def test_provider_call_retries_transient_error_then_succeeds(tmp_path):
    adapter = FakeAdapter(
        "openai",
        failures=[CrupierProviderRateLimitError("temporary rate limit")],
        outputs=["recovered"],
    )
    client = Crupier(
        make_config(tmp_path, allow=["openai:gpt-5.4-mini"]),
        adapters={"openai": adapter},
    )

    result = client.deal("Say hi", dry_run=False, trace="debug")

    assert result.output_text == "recovered"
    assert len(adapter.calls) == 2
    assert result.trace is not None
    assert result.trace.provider_calls[0]["attempt"] == 2
    assert result.trace.errors[0]["phase"] == "provider_call"
    assert result.trace.errors[0]["retryable"] is True
    assert result.trace.final_quality_signals["provider_retry_errors"] == 1


def test_provider_retry_can_be_disabled_per_request(tmp_path):
    adapter = FakeAdapter(
        "openai",
        failures=[CrupierProviderRateLimitError("temporary rate limit")],
        outputs=["should not run"],
    )
    client = Crupier(
        make_config(tmp_path, allow=["openai:gpt-5.4-mini"]),
        adapters={"openai": adapter},
    )

    try:
        client.deal("Say hi", constraints={"max_provider_retries": 0}, dry_run=False, trace="debug")
    except CrupierProviderRateLimitError:
        pass
    else:
        raise AssertionError("provider retry should be disabled by request constraint")

    assert len(adapter.calls) == 1


def test_provider_retry_skips_nonretryable_setup_errors(tmp_path):
    adapter = FakeAdapter(
        "openai",
        failures=[CrupierProviderUnavailableError("missing optional dependency", retryable=False)],
        outputs=["should not run"],
    )
    client = Crupier(
        make_config(tmp_path, allow=["openai:gpt-5.4-mini"]),
        adapters={"openai": adapter},
    )

    try:
        client.deal("Say hi", dry_run=False, trace="debug")
    except CrupierProviderUnavailableError as exc:
        assert exc.retryable is False
    else:
        raise AssertionError("nonretryable provider errors should stop immediately")

    assert len(adapter.calls) == 1


def test_budget_blocks_before_provider_call(tmp_path):
    adapter = FakeAdapter("openai")
    client = Crupier(
        make_config(tmp_path, allow=["openai:gpt-5.4-mini"]),
        adapters={"openai": adapter},
    )

    try:
        client.deal("Say hi", constraints={"max_cost_usd": 0.0}, dry_run=False)
    except CrupierBudgetExceededError as exc:
        assert "exceeds max" in str(exc)
    else:
        raise AssertionError("budget should block before provider execution")

    assert adapter.calls == []


def test_real_fallback_tries_next_adapter(tmp_path):
    openai = FakeAdapter("openai", fail=True)
    anthropic = FakeAdapter("anthropic")
    client = Crupier(
        make_config(tmp_path, allow=["openai:gpt-5.5", "anthropic:claude-opus-4-8"], strategy="fallback"),
        adapters={"openai": openai, "anthropic": anthropic},
    )

    result = client.deal("Use fallback", dry_run=False, trace="debug")

    assert result.output_text == "anthropic:claude-opus-4-8: ok"
    assert len(openai.calls) == 2
    assert len(anthropic.calls) == 1
    assert result.trace is not None
    assert result.trace.fallbacks[0]["model"] == "openai:gpt-5.5"
    assert result.trace.errors[0]["max_provider_retries"] == 1


def test_real_fusion_runs_panel_judge_and_writer(tmp_path):
    openai = FakeAdapter("openai")
    anthropic = FakeAdapter("anthropic")
    client = Crupier(
        make_config(tmp_path, allow=["openai:gpt-5.5", "anthropic:claude-opus-4-8"]),
        adapters={"openai": openai, "anthropic": anthropic},
    )

    result = client.deal("Compare things", mode="research", dry_run=False, trace="summary")

    assert result.route is not None
    assert result.route.strategy == "fusion"
    assert result.output_text.startswith(result.route.steps[-1].model or "")
    assert len(openai.calls) + len(anthropic.calls) == 4


def test_real_tool_execution_runs_safe_callable(tmp_path):
    def add(a: int, b: int):
        """Add two numbers."""
        return {"sum": a + b}

    adapter = FakeAdapter(
        "openai",
        outputs=[
            '{"tool_calls":[{"name":"add","arguments":{"a":2,"b":3}}]}',
            "The sum is 5.",
        ],
    )
    client = Crupier(
        make_config(tmp_path, allow=["openai:gpt-5.4-mini"]),
        adapters={"openai": adapter},
    )

    result = client.deal("Add 2 and 3", tools=[add], dry_run=False, trace="summary")

    assert result.output_text == "The sum is 5."
    assert result.provider_metadata["tool_calls"][0]["name"] == "add"
    assert result.provider_metadata["tool_calls"][0]["status"] == "completed"
    assert result.provider_metadata["tool_calls"][0]["result"] == {"sum": 5}
    assert len(adapter.calls) == 2


def test_real_tool_execution_blocks_sensitive_tool_without_approval(tmp_path):
    called = {"value": False}

    def write_file(path: str, content: str):
        called["value"] = True
        return {"path": path, "bytes": len(content)}

    adapter = FakeAdapter(
        "openai",
        outputs=['{"tool_calls":[{"name":"write_file","arguments":{"path":"x.txt","content":"hello"}}]}'],
    )
    client = Crupier(
        make_config(tmp_path, allow=["openai:gpt-5.4-mini"]),
        adapters={"openai": adapter},
    )
    tool = {
        "name": "write_file",
        "description": "Write a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        "handler": write_file,
        "requires_approval": True,
    }

    try:
        client.deal("Write a file", tools=[tool], dry_run=False)
    except CrupierToolApprovalRequired as exc:
        assert "requires approval" in str(exc)
    else:
        raise AssertionError("sensitive tool should require approval")

    assert called["value"] is False


def test_real_tool_execution_skips_duplicate_idempotency_key(tmp_path):
    calls = {"count": 0}

    def lookup_user(name: str):
        calls["count"] += 1
        return {"name": name, "id": "usr_123"}

    adapter = FakeAdapter(
        "openai",
        outputs=[
            '{"tool_calls":['
            '{"name":"lookup_user","arguments":{"name":"Ada"}},'
            '{"name":"lookup_user","arguments":{"name":"Ada"}}'
            "]}",
            "Ada is usr_123.",
        ],
    )
    client = Crupier(
        make_config(tmp_path, allow=["openai:gpt-5.4-mini"]),
        adapters={"openai": adapter},
    )

    result = client.deal("Lookup Ada twice if needed", tools=[lookup_user], dry_run=False)

    assert result.output_text == "Ada is usr_123."
    assert calls["count"] == 1
    assert [item["status"] for item in result.provider_metadata["tool_calls"]] == [
        "completed",
        "skipped_duplicate",
    ]


def test_tool_planner_does_not_receive_final_response_schema(tmp_path):
    def lookup_user(name: str):
        return {"name": name, "id": "usr_123"}

    adapter = FakeAdapter(
        "openai",
        outputs=[
            '{"tool_calls":[{"name":"lookup_user","arguments":{"name":"Ada"}}]}',
            '{"name": "Ada", "id": "usr_123"}',
        ],
    )
    client = Crupier(
        make_config(tmp_path, allow=["openai:gpt-5.4-mini"]),
        adapters={"openai": adapter},
    )
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "id": {"type": "string"}},
        "required": ["name", "id"],
        "additionalProperties": False,
    }

    result = client.deal("Lookup Ada", tools=[lookup_user], response_schema=schema, dry_run=False)

    assert result.output_json == {"name": "Ada", "id": "usr_123"}
    assert adapter.calls[0]["response_schema"] is None
    assert adapter.calls[1]["response_schema"] == schema


def test_real_structured_output_returns_validated_json(tmp_path):
    adapter = FakeAdapter("openai", outputs=['{"name": "Ada", "total": 12.5}'])
    client = Crupier(
        make_config(tmp_path, allow=["openai:gpt-5.4-mini"]),
        adapters={"openai": adapter},
    )
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "total": {"type": "number"}},
        "required": ["name", "total"],
        "additionalProperties": False,
    }

    result = client.deal("Extract invoice data", response_schema=schema, dry_run=False, trace="summary")

    assert result.output_json == {"name": "Ada", "total": 12.5}
    assert result.output_text == '{"name": "Ada", "total": 12.5}'
    assert "JSON Schema" in adapter.calls[0]["prompt"]


def test_real_structured_output_repairs_invalid_json(tmp_path):
    adapter = FakeAdapter("openai", outputs=["name: Ada", '{"name": "Ada"}'])
    client = Crupier(
        make_config(tmp_path, allow=["openai:gpt-5.4-mini"]),
        adapters={"openai": adapter},
    )
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }

    result = client.deal("Extract user name", response_schema=schema, dry_run=False, trace="summary")

    assert result.output_json == {"name": "Ada"}
    assert len(adapter.calls) == 2
    assert "previous output was invalid" in adapter.calls[1]["prompt"]
