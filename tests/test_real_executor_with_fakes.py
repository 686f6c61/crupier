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


def test_provider_circuit_breaker_blocks_after_repeated_failures(tmp_path):
    adapter = FakeAdapter("openai", fail=True)
    config = make_config(tmp_path, allow=["openai:gpt-5.4-mini"])
    config.routing.max_provider_retries = 0
    config.routing.circuit_breaker_failure_threshold = 2
    config.routing.circuit_breaker_cooldown_seconds = 30
    client = Crupier(config, adapters={"openai": adapter})

    for _ in range(2):
        try:
            client.deal("Say hi", dry_run=False, trace="debug")
        except CrupierProviderUnavailableError:
            pass
        else:
            raise AssertionError("provider should fail")

    try:
        client.deal("Say hi", dry_run=False, trace="debug")
    except CrupierProviderUnavailableError as exc:
        assert "circuit breaker is open" in str(exc)
        assert exc.retryable is False
    else:
        raise AssertionError("open circuit should block provider call")

    assert len(adapter.calls) == 2


def test_provider_circuit_breaker_removes_degraded_provider_from_routing(tmp_path):
    openai = FakeAdapter("openai", fail=True)
    anthropic = FakeAdapter("anthropic", outputs=["healthy provider"])
    config = make_config(
        tmp_path,
        allow=["openai:gpt-5.4-mini", "anthropic:claude-opus-4-8"],
        strategy="single",
    )
    config.routing.max_provider_retries = 0
    config.routing.circuit_breaker_failure_threshold = 1
    config.routing.circuit_breaker_cooldown_seconds = 30
    client = Crupier(config, adapters={"openai": openai, "anthropic": anthropic})

    try:
        client.deal(
            "Prime circuit",
            constraints={"force_model": "openai:gpt-5.4-mini"},
            dry_run=False,
            trace="debug",
        )
    except CrupierProviderUnavailableError:
        pass
    else:
        raise AssertionError("forced degraded provider should fail")

    result = client.deal("Use a healthy provider", dry_run=False, trace="debug")

    assert result.output_text == "healthy provider"
    assert len(openai.calls) == 1
    assert len(anthropic.calls) == 1
    assert result.route is not None
    assert result.route.models == ["anthropic:claude-opus-4-8"]
    assert result.trace is not None
    assert any(item["model"] == "openai:gpt-5.4-mini" for item in result.trace.excluded_models)
    assert "provider_circuit_breaker" in result.trace.policy_filters


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


def test_real_cascade_escalates_when_primary_validation_fails(tmp_path):
    openai = FakeAdapter("openai", outputs=["Not enough information to answer."])
    anthropic = FakeAdapter("anthropic", outputs=["Escalated final answer."])
    client = Crupier(
        make_config(
            tmp_path,
            allow=["openai:gpt-5.4-mini", "anthropic:claude-opus-4-8"],
            strategy="cascade",
        ),
        adapters={"openai": openai, "anthropic": anthropic},
    )

    result = client.deal("Use cascade", dry_run=False, trace="debug")

    assert result.route is not None
    assert result.route.strategy == "cascade"
    assert result.output_text == "Escalated final answer."
    assert len(openai.calls) == 1
    assert len(anthropic.calls) == 1
    assert result.trace is not None
    assert any(item["phase"] == "cascade_validation" for item in result.trace.fallbacks)


def test_real_delegate_runs_nested_route_with_reduced_depth(tmp_path):
    adapter = FakeAdapter("openai", outputs=["delegated final answer"])
    client = Crupier(
        make_config(tmp_path, allow=["openai:gpt-5.4-mini"], strategy="delegate"),
        adapters={"openai": adapter},
    )

    result = client.deal("Plan a multi-step answer", constraints={"max_depth": 2}, dry_run=False, trace="debug")

    assert result.route is not None
    assert result.route.strategy == "delegate"
    assert result.output_text == "delegated final answer"
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["constraints"]["max_depth"] == 1
    assert result.trace is not None
    assert result.trace.provider_calls[0]["role"] == "delegate"
    assert result.trace.provider_calls[0]["nested_strategy"] == "single"
    assert result.trace.provider_calls[1]["role"] == "primary"


def test_real_delegate_blocks_when_max_depth_is_exhausted(tmp_path):
    adapter = FakeAdapter("openai")
    client = Crupier(
        make_config(tmp_path, allow=["openai:gpt-5.4-mini"], strategy="delegate"),
        adapters={"openai": adapter},
    )

    try:
        client.deal("Plan a multi-step answer", constraints={"max_depth": 0}, dry_run=False, trace="debug")
    except CrupierProviderUnavailableError as exc:
        assert "max_depth" in str(exc)
    else:
        raise AssertionError("delegate should be blocked when max_depth is exhausted")

    assert adapter.calls == []


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


def test_real_tool_execution_can_replan_for_multiple_rounds(tmp_path):
    def lookup_user(name: str):
        """Look up a user."""
        return {"user_id": "usr_123", "name": name}

    def lookup_order(user_id: str):
        """Look up a user's order."""
        return {"order_id": "ord_456", "status": "shipped", "user_id": user_id}

    adapter = FakeAdapter(
        "openai",
        outputs=[
            '{"tool_calls":[{"name":"lookup_user","arguments":{"name":"Ada"}}]}',
            '{"tool_calls":[{"name":"lookup_order","arguments":{"user_id":"usr_123"}}]}',
            "Ada order ord_456 is shipped.",
        ],
    )
    client = Crupier(
        make_config(tmp_path, allow=["openai:gpt-5.4-mini"]),
        adapters={"openai": adapter},
    )

    result = client.deal("Find Ada's order status", tools=[lookup_user, lookup_order], dry_run=False, trace="summary")

    assert result.output_text == "Ada order ord_456 is shipped."
    assert [item["name"] for item in result.provider_metadata["tool_calls"]] == ["lookup_user", "lookup_order"]
    assert len(adapter.calls) == 3


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
            '{"final":"ready for final structured answer"}',
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
    assert adapter.calls[1]["response_schema"] is None
    assert adapter.calls[2]["response_schema"] == schema


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
