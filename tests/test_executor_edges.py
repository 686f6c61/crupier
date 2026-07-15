from time import perf_counter

import pytest

import crupier.executor as executor_module
from crupier.adapters import AdapterResponse
from crupier.budgets import ExecutionBudget
from crupier.config import CrupierConfig
from crupier.errors import (
    CrupierExecutionLimitError,
    CrupierProviderRateLimitError,
    CrupierProviderUnavailableError,
    CrupierRouteValidationError,
    CrupierStructuredOutputError,
)
from crupier.executor import RouteExecutor
from crupier.models import (
    DecisionTrace,
    FileAsset,
    FileRepresentation,
    FileRoutingPlan,
    RequestEnvelope,
    RoutePlan,
    RouteStep,
)


class ScriptedAdapter:
    def __init__(self, provider="openai", outputs=None, errors=None):
        self.provider = provider
        self.outputs = list(outputs or [])
        self.errors = list(errors or [])
        self.calls = []

    def generate(self, *, model, prompt, request):
        self.calls.append({"model": model, "prompt": prompt, "request": request})
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        output = self.outputs.pop(0) if self.outputs else f"{model}:ok"
        return AdapterResponse(
            text=output,
            raw={"model": model, "text": output},
            usage={"input_tokens": 1, "output_tokens": 1},
            metadata={"provider": self.provider, "model": model},
        )


def _config(tmp_path, *, allow=None, parallel=False):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "executor-test", "default_profile": "agentic"},
            "providers": {
                "openai": {"enabled": True},
                "anthropic": {"enabled": True},
            },
            "models": {"allow": allow or ["openai:a", "openai:b", "openai:c"]},
            "routing": {
                "allow_parallel": parallel,
                "max_provider_retries": 0,
                "retry_backoff_seconds": 0,
            },
            "profiles": {"agentic": {"prefer": [], "strategy": "single"}},
        }
    )
    config.root = tmp_path
    return config


def _trace():
    return DecisionTrace(trace_id="trace", request_summary="test")


def _budget(executor, request):
    return ExecutionBudget(executor.config, request, executor._budget_cards())


def _plan(strategy="single", steps=None):
    return RoutePlan(
        strategy=strategy,
        steps=steps or [RouteStep(role="primary", model="openai:a")],
        reason="test",
    )


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [("3", 1, 3), (-2, 1, 0), ("bad", 4, 4), (None, -2, 0)],
)
def test_executor_numeric_helpers_are_defensive(value, default, expected):
    assert executor_module._non_negative_int(value, default=default) == expected


def test_executor_optional_float_and_positive_helpers_are_defensive():
    assert executor_module._positive_int(0, default=2) == 1
    assert executor_module._positive_int("bad", default=2) == 2
    assert executor_module._optional_non_negative_float(None) is None
    assert executor_module._optional_non_negative_float("2.5") == 2.5
    assert executor_module._optional_non_negative_float(-2) == 0
    assert executor_module._optional_non_negative_float("bad") is None


def test_executor_initialization_survives_registry_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(executor_module.ModelRegistry, "load", lambda self: (_ for _ in ()).throw(OSError("bad")))
    executor = RouteExecutor(_config(tmp_path))
    assert executor._cards == {}


def test_executor_dry_run_includes_structured_and_file_plan(tmp_path):
    executor = RouteExecutor(_config(tmp_path))
    file_plan = FileRoutingPlan(
        assets=[FileAsset(kind="pdf", name="report.pdf")],
        representations=[FileRepresentation("report.pdf", "pdf", "extracted_text_chunks")],
    )
    request = RequestEnvelope(
        task="read",
        mode="structured",
        constraints={"response_schema": {"type": "object"}},
        file_plan=file_plan,
    )

    result = executor.execute(request, _plan(), _trace(), dry_run=True)

    assert result.output_json["dry_run"] is True
    assert "pdf->extracted_text_chunks" in result.output_text
    assert result.trace.final_quality_signals["dry_run"] is True


def test_executor_unknown_strategy_uses_first_model_and_preserves_requested_raw_output(tmp_path):
    adapter = ScriptedAdapter(outputs=["answer"])
    executor = RouteExecutor(_config(tmp_path), adapters={"openai": adapter})
    request = RequestEnvelope(
        task="x",
        constraints={"include_raw_outputs": True},
        metadata={
            "extracted_file_context": {
                "files": [{"name": "notes.txt"}],
                "warnings": ["bounded"],
                "max_chars": 10,
            }
        },
    )

    result = executor.execute(request, _plan(strategy="custom"), _trace(), dry_run=False)

    assert result.output_text == "answer"
    assert result.raw_outputs == [{"model": "a", "text": "answer"}]
    assert "executed as first-model" in result.warnings[0]
    assert result.trace.final_quality_signals["file_context"]["max_chars"] == 10


def test_executor_tool_planner_can_finish_without_tools_and_enforces_round_limit(tmp_path):
    adapter = ScriptedAdapter(outputs=["No tool needed"])
    executor = RouteExecutor(_config(tmp_path), adapters={"openai": adapter})
    request = RequestEnvelope(task="x", tools=[lambda: "unused"])

    result = executor.execute(request, _plan(), _trace(), dry_run=False)
    assert result.output_text == "No tool needed"
    assert result.provider_metadata["tool_calls"] == []

    adapter.outputs = [
        '{"tool_calls":[{"name":"one","arguments":{}},{"name":"two","arguments":{}}]}'
    ]
    request = RequestEnvelope(
        task="x",
        tools=[{"name": "one", "handler": lambda: 1}, {"name": "two", "handler": lambda: 2}],
        constraints={"max_tool_calls_per_round": 1},
    )
    with pytest.raises(CrupierRouteValidationError, match="above max_tool_calls_per_round=1"):
        executor.execute(request, _plan(), _trace(), dry_run=False)


def test_executor_structured_output_fails_after_all_models_and_repairs(tmp_path):
    adapter = ScriptedAdapter(outputs=["bad", "still bad", "bad", "still bad"])
    executor = RouteExecutor(_config(tmp_path), adapters={"openai": adapter})
    plan = _plan(
        strategy="fallback",
        steps=[
            RouteStep(role="primary", model="openai:a"),
            RouteStep(role="fallback", model="openai:b"),
        ],
    )

    with pytest.raises(CrupierStructuredOutputError, match="failed for all route models"):
        executor.execute(
            RequestEnvelope(task="x", response_schema={"type": "object"}),
            plan,
            _trace(),
            dry_run=False,
        )


def test_executor_response_schema_request_helpers_remove_constraint_copy(tmp_path):
    executor = RouteExecutor(_config(tmp_path))
    original = RequestEnvelope(task="x", response_schema={"old": True}, constraints={"response_schema": {"x": 1}})

    without = executor._request_with_response_schema(original, None)
    with_schema = executor._request_with_response_schema(original, {"type": "object"})

    assert without.response_schema is None and "response_schema" not in without.constraints
    assert with_schema.response_schema == {"type": "object"}
    assert original.constraints == {"response_schema": {"x": 1}}


def test_executor_first_model_and_fallback_report_exhaustion(tmp_path):
    failing = ScriptedAdapter(errors=[CrupierProviderUnavailableError("a"), CrupierProviderUnavailableError("b")])
    executor = RouteExecutor(_config(tmp_path), adapters={"openai": failing})
    request = RequestEnvelope(task="x", constraints={"max_provider_retries": 0})

    with pytest.raises(CrupierProviderUnavailableError, match="no executable model"):
        executor.execute(request, RoutePlan(strategy="single", steps=[]), _trace(), dry_run=False)

    plan = _plan(
        strategy="fallback",
        steps=[RouteStep(role="primary", model="openai:a"), RouteStep(role="fallback", model="openai:b")],
    )
    trace = _trace()
    with pytest.raises(CrupierProviderUnavailableError, match="All fallback models failed"):
        executor.execute(request, plan, trace, dry_run=False)
    assert len(trace.fallbacks) == 2


def test_executor_cascade_handles_no_models_all_failures_and_last_insufficient_response(tmp_path):
    executor = RouteExecutor(_config(tmp_path), adapters={"openai": ScriptedAdapter()})
    request = RequestEnvelope(task="x", constraints={"max_provider_retries": 0})

    with pytest.raises(CrupierProviderUnavailableError, match="no executable model"):
        executor.execute(request, RoutePlan(strategy="cascade", steps=[]), _trace(), dry_run=False)

    failing = ScriptedAdapter(errors=[CrupierProviderUnavailableError("a"), CrupierProviderUnavailableError("b")])
    executor.adapters["openai"] = failing
    plan = _plan(
        strategy="cascade",
        steps=[RouteStep(role="primary", model="openai:a"), RouteStep(role="escalation", model="openai:b")],
    )
    trace = _trace()
    with pytest.raises(CrupierProviderUnavailableError, match="All cascade models failed"):
        executor.execute(request, plan, trace, dry_run=False)
    assert trace.fallbacks[0]["next_model"] == "openai:b"

    executor = RouteExecutor(
        _config(tmp_path),
        adapters={"openai": ScriptedAdapter(outputs=["I do not know"])},
    )
    with pytest.raises(CrupierProviderUnavailableError, match="without a sufficient response") as exc:
        executor.execute(request, _plan(strategy="cascade"), _trace(), dry_run=False)
    assert exc.value.retryable is False


def test_executor_cascade_model_validator_rejects_then_accepts_escalation(tmp_path):
    adapter = ScriptedAdapter(
        outputs=[
            "candidate one",
            '{"sufficient":false,"reason":"weak"}',
            "candidate two",
            '{"sufficient":true}',
        ]
    )
    executor = RouteExecutor(_config(tmp_path), adapters={"openai": adapter})
    plan = _plan(
        strategy="cascade",
        steps=[
            RouteStep(role="primary", model="openai:a"),
            RouteStep(role="validator", model="openai:c"),
            RouteStep(role="escalation", model="openai:b"),
        ],
    )

    result = executor.execute(RequestEnvelope(task="x"), plan, _trace(), dry_run=False)

    assert result.output_text == "candidate two"
    assert any(item["phase"] == "cascade_escalation" for item in result.trace.fallbacks)


def test_executor_cascade_validator_falls_back_to_heuristic_for_invalid_shapes(tmp_path):
    adapter = ScriptedAdapter(outputs=["solid answer", "not-json", "solid answer", "[]"])
    executor = RouteExecutor(_config(tmp_path), adapters={"openai": adapter})
    plan = _plan(
        strategy="cascade",
        steps=[RouteStep(role="primary", model="openai:a"), RouteStep(role="validator", model="openai:c")],
    )

    first = executor.execute(RequestEnvelope(task="x"), plan, _trace(), dry_run=False)
    second = executor.execute(RequestEnvelope(task="x"), plan, _trace(), dry_run=False)

    assert first.output_text == "solid answer"
    assert second.output_text == "solid answer"


@pytest.mark.parametrize(
    ("text", "constraints", "ok", "reason"),
    [
        ("short", {"cascade_min_output_chars": 10}, False, "shorter"),
        ("short", {"cascade_min_output_chars": "bad"}, True, "passed"),
        ("I don't know", {}, False, "uncertainty"),
        ("complete answer", {}, True, "passed"),
    ],
)
def test_executor_cascade_heuristic(text, constraints, ok, reason, tmp_path):
    executor = RouteExecutor(_config(tmp_path))
    result = executor._heuristic_validate_cascade_response(
        RequestEnvelope(task="x", constraints=constraints), AdapterResponse(text=text)
    )
    assert result[0] is ok and reason in result[1]


def test_executor_panel_sequential_keeps_successes_and_reports_total_failure(tmp_path):
    openai = ScriptedAdapter(errors=[CrupierProviderUnavailableError("first")], outputs=["second"])
    executor = RouteExecutor(_config(tmp_path, parallel=False), adapters={"openai": openai})
    plan = _plan(strategy="panel", steps=[RouteStep(role="panel", models=["openai:a", "openai:b"])])
    request = RequestEnvelope(task="x", constraints={"max_provider_retries": 0})

    result = executor.execute(request, plan, _trace(), dry_run=False)
    assert result.output_text == "## openai:b\nsecond"
    assert any(item["phase"] == "panel" for item in result.trace.errors)

    failing = ScriptedAdapter(
        errors=[CrupierProviderUnavailableError("a"), CrupierProviderUnavailableError("b")]
    )
    executor.adapters["openai"] = failing
    with pytest.raises(CrupierProviderUnavailableError, match="All panel models failed"):
        executor.execute(request, plan, _trace(), dry_run=False)


def test_executor_panel_parallel_keeps_order_and_records_failure(tmp_path):
    openai = ScriptedAdapter(errors=[CrupierProviderUnavailableError("one"), None], outputs=["two"])
    executor = RouteExecutor(_config(tmp_path, parallel=True), adapters={"openai": openai})
    plan = _plan(strategy="panel", steps=[RouteStep(role="panel", models=["openai:a", "openai:b"])])

    result = executor.execute(
        RequestEnvelope(task="x", constraints={"max_provider_retries": 0, "max_parallel_models": 2}),
        plan,
        _trace(),
        dry_run=False,
    )

    assert "two" in result.output_text
    assert any(item["phase"] == "panel" for item in result.trace.errors)


def test_executor_fusion_rejects_empty_panel_and_uses_role_fallbacks(tmp_path):
    executor = RouteExecutor(_config(tmp_path), adapters={"openai": ScriptedAdapter()})
    with pytest.raises(CrupierProviderUnavailableError, match="Fusion panel failed"):
        executor.execute(
            RequestEnvelope(task="x"),
            RoutePlan(strategy="fusion", steps=[]),
            _trace(),
            dry_run=False,
        )

    adapter = ScriptedAdapter(outputs=["panel", "judge", "final"])
    executor.adapters["openai"] = adapter
    plan = _plan(strategy="fusion", steps=[RouteStep(role="panel", models=["openai:a"])])
    result = executor.execute(RequestEnvelope(task="x"), plan, _trace(), dry_run=False)
    assert result.output_text == "final"


def test_executor_fusion_rejects_degraded_single_output(tmp_path):
    adapter = ScriptedAdapter(outputs=["", "survivor"])
    executor = RouteExecutor(_config(tmp_path, parallel=False), adapters={"openai": adapter})
    plan = _plan(
        strategy="fusion",
        steps=[
            RouteStep(role="panel", models=["openai:a", "openai:b"]),
            RouteStep(role="judge", model="openai:c"),
            RouteStep(role="final_writer", model="openai:a"),
        ],
    )

    with pytest.raises(CrupierProviderUnavailableError, match="at least 2 non-empty panel outputs"):
        executor.execute(RequestEnvelope(task="x"), plan, _trace(), dry_run=False)


def test_executor_critique_repair_runs_complete_sequence(tmp_path):
    adapter = ScriptedAdapter(
        outputs=[
            "draft",
            "critique",
            "```xml\n<final_answer>\nrepaired\n</final_answer>\n```\n## Independent Critic Challenge that must not leak",
        ]
    )
    executor = RouteExecutor(_config(tmp_path), adapters={"openai": adapter})
    plan = _plan(
        strategy="critique_repair",
        steps=[
            RouteStep(role="generator", model="openai:a"),
            RouteStep(role="critic", model="openai:b"),
            RouteStep(role="repair", model="openai:c"),
        ],
    )

    result = executor.execute(RequestEnvelope(task="x"), plan, _trace(), dry_run=False)

    assert result.output_text == "repaired"
    assert [call["role"] for call in result.trace.provider_calls] == ["generator", "critic", "repair"]
    assert "Prompt-Version: executor.critique.v1" in adapter.calls[1]["prompt"]
    assert "Prompt-Version: executor.repair.v1" in adapter.calls[2]["prompt"]
    assert "Independent Critic Challenge" not in result.output_text


def test_executor_critique_repair_falls_back_between_validated_role_models(tmp_path):
    openai = ScriptedAdapter(provider="openai", outputs=[""])
    anthropic = ScriptedAdapter(
        provider="anthropic",
        outputs=["fallback draft", "independent critique", "repaired final"],
    )
    config = _config(tmp_path, allow=["openai:a", "anthropic:b", "anthropic:c"])
    executor = RouteExecutor(config, adapters={"openai": openai, "anthropic": anthropic})
    plan = RoutePlan(
        strategy="critique_repair",
        steps=[
            RouteStep(role="generator", model="openai:a"),
            RouteStep(role="critic", model="anthropic:b"),
            RouteStep(role="repair", model="anthropic:c"),
        ],
    )

    result = executor.execute(
        RequestEnvelope(task="x", constraints={"max_provider_retries": 0}),
        plan,
        _trace(),
        dry_run=False,
    )

    assert result.output_text == "repaired final"
    assert [call["status"] for call in result.trace.provider_calls] == ["failed", "success", "success", "success"]
    assert result.trace.fallbacks[0] == {
        "phase": "role_fallback",
        "role": "generator",
        "plan_role": "generator",
        "model": "openai:a",
        "error": "Provider 'openai' model 'openai:a' returned an empty text response.",
        "next_model": "anthropic:b",
    }


def test_executor_tools_honor_critique_repair_roles(tmp_path):
    def lookup_case(case_id: str):
        return {"case_id": case_id, "refund_status": "not_started"}

    adapter = ScriptedAdapter(
        outputs=[
            '{"tool_calls":[{"name":"lookup_case","arguments":{"case_id":"C-1"}}]}',
            '{"tool_calls":[],"final":"draft from tool"}',
            "unsupported-claim critique",
            "<final_answer>\nrepaired tool-grounded answer\n</final_answer>\nInternal verification note that must not leak.",
        ]
    )
    executor = RouteExecutor(_config(tmp_path), adapters={"openai": adapter})
    plan = _plan(
        strategy="critique_repair",
        steps=[
            RouteStep(role="generator", model="openai:a"),
            RouteStep(role="critic", model="openai:b"),
            RouteStep(role="repair", model="openai:c"),
        ],
    )

    result = executor.execute(
        RequestEnvelope(task="check case", tools=[lookup_case], constraints={"max_tool_rounds": 2}),
        plan,
        _trace(),
        dry_run=False,
    )

    assert result.output_text == "repaired tool-grounded answer"
    assert result.provider_metadata["tool_calls"][0]["status"] == "completed"
    assert [call["role"] for call in result.trace.provider_calls] == [
        "tool_planner_round_1",
        "tool_planner_round_2",
        "tool_critic",
        "tool_repair",
    ]
    assert '"refund_status": "not_started"' in adapter.calls[2]["prompt"]
    assert '"refund_status": "not_started"' in adapter.calls[3]["prompt"]
    assert "Prompt-Version: executor.tool_critique.v1" in adapter.calls[2]["prompt"]
    assert "Prompt-Version: executor.tool_repair.v1" in adapter.calls[3]["prompt"]
    assert "Return exactly one <final_answer>" in adapter.calls[3]["prompt"]
    assert "Internal verification note" not in result.output_text


def test_executor_delegate_requires_step(tmp_path):
    executor = RouteExecutor(_config(tmp_path))
    with pytest.raises(CrupierProviderUnavailableError, match="no delegate step"):
        executor.execute(
            RequestEnvelope(task="x"),
            RoutePlan(strategy="delegate", steps=[]),
            _trace(),
            dry_run=False,
        )


def test_executor_call_model_requires_adapter_and_records_open_circuit(tmp_path):
    executor = RouteExecutor(_config(tmp_path), adapters={})
    request = RequestEnvelope(task="x")
    trace = _trace()
    with pytest.raises(CrupierProviderUnavailableError, match="No adapter configured"):
        executor._call_model(
            "openai:a", "x", request, trace, [], role="primary", budget=_budget(executor, request)
        )

    executor.adapters["openai"] = ScriptedAdapter()
    executor._provider_circuit_open_until["openai"] = perf_counter() + 10
    with pytest.raises(CrupierProviderUnavailableError, match="circuit breaker is open"):
        executor._call_model(
            "openai:a", "x", request, trace, [], role="primary", budget=_budget(executor, request)
        )
    assert trace.errors[-1]["circuit_open"] is True

    executor._provider_circuit_open_until["openai"] = perf_counter() - 1
    executor._provider_failure_counts["openai"] = 2
    assert executor.provider_circuit_open_reason("openai") is None
    assert "openai" not in executor._provider_failure_counts


def test_executor_retry_backoff_runtime_policy_and_success_reset(tmp_path, monkeypatch):
    adapter = ScriptedAdapter(
        errors=[CrupierProviderRateLimitError("slow"), None],
        outputs=["recovered"],
    )
    config = _config(tmp_path)
    config.routing.max_provider_retries = 1
    config.routing.retry_backoff_seconds = 0.01
    config.routing.retry_jitter_seconds = 0.02
    executor = RouteExecutor(config, adapters={"openai": adapter})
    sleeps = []
    monkeypatch.setattr(executor_module, "sleep", sleeps.append)
    monkeypatch.setattr(executor_module.random, "uniform", lambda start, end: end)

    result = executor.execute(RequestEnvelope(task="x"), _plan(), _trace(), dry_run=False)

    assert result.output_text == "recovered"
    assert sleeps == [0.03]
    assert executor._provider_failure_counts == {}


def test_executor_configuration_helpers_fall_back_for_invalid_values(tmp_path):
    executor = RouteExecutor(_config(tmp_path))
    request = RequestEnvelope(
        task="x",
        constraints={
            "max_provider_retries": "bad",
            "retry_backoff_seconds": "bad",
            "retry_jitter_seconds": "bad",
            "max_tool_rounds": "bad",
            "max_depth": "bad",
            "max_parallel_models": "bad",
        },
    )

    assert executor._provider_retry_budget(request) == executor.config.routing.max_provider_retries
    assert executor._provider_retry_backoff_seconds(request) == executor.config.routing.retry_backoff_seconds
    assert executor._provider_retry_jitter_seconds(request) == executor.config.routing.retry_jitter_seconds
    assert executor._max_tool_rounds(request) == executor.config.routing.max_tool_rounds
    assert executor._remaining_delegate_depth(request) == executor.config.routing.max_depth
    assert executor._max_parallel_models(request, 3) == 3
    assert executor._coerce_non_negative_int("bad", 2) == 2
    assert executor._provider_error_retryable(CrupierProviderRateLimitError("x")) is True
    assert executor._provider_error_retryable(CrupierProviderUnavailableError("x", retryable=False)) is False


def test_executor_circuit_threshold_cost_and_model_helpers(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.routing.circuit_breaker_failure_threshold = 0
    executor = RouteExecutor(config)
    executor._record_provider_failure("openai")
    assert executor._provider_failure_counts == {}

    config.routing.circuit_breaker_failure_threshold = 1
    config.routing.circuit_breaker_cooldown_seconds = 0
    executor._record_provider_failure("openai")
    assert executor._provider_circuit_open_until == {}

    monkeypatch.setattr(
        executor_module.ModelRegistry,
        "list",
        lambda self, allowed_only=False: (_ for _ in ()).throw(OSError("registry")),
    )
    assert executor._budget_cards() == []
    assert executor._actual_cost([]) is None
    assert executor._usage_estimated_cost([]) is None

    plan = RoutePlan(
        strategy="custom",
        steps=[
            RouteStep(role="other", model="openai:a", models=["openai:b"]),
            RouteStep(role="other", model="openai:a", models=["openai:c"]),
        ],
    )
    assert executor._models_in_execution_order(plan) == ["openai:a", "openai:b", "openai:c"]
    assert executor._cascade_models(plan) == ["openai:a", "openai:b", "openai:c"]
    assert executor._model_for_role(plan, "missing") is None


def test_executor_parallel_panel_propagates_execution_limit(tmp_path):
    adapter = ScriptedAdapter(outputs=["one", "two"])
    executor = RouteExecutor(_config(tmp_path, parallel=True), adapters={"openai": adapter})
    plan = _plan(strategy="panel", steps=[RouteStep(role="panel", models=["openai:a", "openai:b"])])

    with pytest.raises(CrupierExecutionLimitError, match="max_calls=1"):
        executor.execute(
            RequestEnvelope(task="x", constraints={"max_calls": 1}),
            plan,
            _trace(),
            dry_run=False,
        )
