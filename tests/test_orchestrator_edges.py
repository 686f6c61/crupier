import json

import pytest

import crupier.orchestrator as orchestrator_module
from crupier.adapters import AdapterResponse
from crupier.config import CrupierConfig, ProfileSettings
from crupier.errors import CrupierBudgetExceededError, CrupierRouteValidationError
from crupier.models import CapabilityCard, FileAsset, ModelRef, PlanningContext, RequestEnvelope
from crupier.orchestrator import (
    DeterministicOrchestrator,
    ModelOrchestrator,
    _candidate_summary,
    _compact_deterministic_scores,
    _extract_json_object,
    _json_text,
    _normalize_common_route_shape,
    _redact_planning_text,
    _request_content,
    _request_shape,
    _tool_name,
)


def make_config(tmp_path) -> CrupierConfig:
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "orchestrator-edges", "default_profile": "agentic"},
            "providers": {
                "openai": {"enabled": True, "env_key": "OPENAI_API_KEY"},
                "anthropic": {"enabled": True, "env_key": "ANTHROPIC_API_KEY"},
                "ollama": {"enabled": True, "host": "http://localhost:11434"},
            },
            "models": {"allow": ["openai:a", "anthropic:b", "ollama:c"]},
            "routing": {
                "default_strategy": "orchestrated",
                "allow_fusion": True,
                "max_calls": 20,
                "max_depth": 4,
                "require_operational_providers": False,
            },
            "orchestrator": {
                "model": "openai:planner",
                "fallback": "deterministic",
                "max_repairs": 0,
                "candidate_limit": 3,
            },
            "profiles": {
                name: {"strategy": "orchestrated"}
                for name in ("agentic", "private", "research", "structured", "cheap", "fast", "quality")
            },
        }
    )
    config.root = tmp_path
    return config


def card(
    key: str,
    *,
    cost: str = "medium",
    latency: str = "medium",
    quality: str = "strong",
) -> CapabilityCard:
    return CapabilityCard(
        model_ref=ModelRef.parse(key),
        last_updated="test",
        supports_tools=True,
        supports_structured_output=True,
        cost_tier=cost,
        latency_tier=latency,
        quality_tier=quality,
        pricing={"input_per_million_usd": 1.0, "confidence": "provider"},
        skill_scores={"coding": 8.0},
    )


def context(request: RequestEnvelope | None = None) -> PlanningContext:
    return PlanningContext(
        request=request or RequestEnvelope(task="route this", mode="agentic"),
        candidates=[
            card("openai:a", cost="low", latency="fast"),
            card("anthropic:b", quality="frontier"),
            card("ollama:c", cost="low", latency="slow"),
        ],
        filters_applied=["allowlist:3"],
    )


class StaticAdapter:
    provider = "openai"

    def __init__(self, text: str = "{}", *, error: Exception | None = None):
        self.text = text
        self.error = error

    def generate(self, *, model, prompt, request):
        if self.error:
            raise self.error
        return AdapterResponse(text=self.text, usage={"input_tokens": 4}, metadata={"model": model})


@pytest.mark.parametrize(
    ("route_request", "expected"),
    [
        (RequestEnvelope(task="x", strategy="panel"), "panel"),
        (RequestEnvelope(task="x", strategy="not-real"), "single"),
        (RequestEnvelope(task="x", mode="private"), "local_first"),
        (RequestEnvelope(task="x", mode="research"), "fusion"),
        (RequestEnvelope(task="x", mode="structured"), "cascade"),
        (RequestEnvelope(task="x", mode="cheap"), "cascade"),
        (RequestEnvelope(task="x", mode="fast"), "single"),
        (RequestEnvelope(task="x", mode="quality"), "single"),
        (RequestEnvelope(task="x", mode="agentic", tools=[object()]), "critique_repair"),
        (RequestEnvelope(task="x", mode="missing"), "single"),
    ],
)
def test_deterministic_strategy_matrix(tmp_path, route_request, expected):
    plan = DeterministicOrchestrator(make_config(tmp_path)).plan(context(route_request))

    assert plan.strategy == expected
    assert plan.selection_scores
    if route_request.strategy == "not-real":
        assert "fell back to single" in plan.reason


def test_orchestrated_modes_fall_back_to_single_with_one_candidate_or_disabled_fusion(tmp_path):
    config = make_config(tmp_path)
    router = DeterministicOrchestrator(config)
    one = [card("openai:a")]

    assert router._orchestrate_strategy(RequestEnvelope(task="x", mode="research"), one) == "single"
    assert router._orchestrate_strategy(RequestEnvelope(task="x", mode="structured"), one) == "single"
    assert router._orchestrate_strategy(RequestEnvelope(task="x", mode="cheap"), one) == "single"
    assert (
        router._orchestrate_strategy(RequestEnvelope(task="x", mode="agentic", tools=[object()]), one)
        == "single"
    )
    config.routing.allow_fusion = False
    assert router._orchestrate_strategy(RequestEnvelope(task="x", mode="research"), context().candidates) == "single"


@pytest.mark.parametrize(
    "when",
    [
        {"tools": True},
        {"structured": True},
        {"risk_level": "high"},
        {"min_tools": 1},
        {"max_tools": -1},
        {"min_candidates": 4},
        {"max_candidates": 2},
        {"file_kind": "pdf"},
    ],
)
def test_strategy_rule_mismatch_conditions(tmp_path, when):
    router = DeterministicOrchestrator(make_config(tmp_path))

    assert router._strategy_rule_matches(RequestEnvelope(task="x"), context().candidates, when) is False


def test_strategy_rules_ignore_invalid_entries_and_return_none(tmp_path):
    config = make_config(tmp_path)
    profile = ProfileSettings(
        name="custom",
        strategy="orchestrated",
        options={"strategy_rules": ["bad", {"strategy": "invalid"}, {"strategy": "single", "when": {"tools": True}}]},
    )
    config.profiles["custom"] = profile
    router = DeterministicOrchestrator(config)

    assert router._strategy_from_rules(RequestEnvelope(task="x"), context().candidates, None) is None
    assert router._strategy_from_rules(RequestEnvelope(task="x"), context().candidates, profile) is None
    profile.options["strategy_rules"] = "bad"
    assert router._strategy_from_rules(RequestEnvelope(task="x"), context().candidates, profile) is None


def test_deterministic_limits_and_risk_helpers(tmp_path):
    config = make_config(tmp_path)
    router = DeterministicOrchestrator(config)

    assert router._latency_estimate([]) == 0
    assert router._panel_size(RequestEnvelope(task="x", constraints={"max_panel_size": "bad"}), context().candidates) == 3
    assert router._max_depth(RequestEnvelope(task="x", constraints={"max_depth": "bad"})) == 4
    assert router._risk_level(RequestEnvelope(task="x", constraints={"risk_level": "critical"}), "single") == "critical"
    assert router._risk_level(RequestEnvelope(task="x", mode="fast"), "single") == "low"
    assert router._risk_level(RequestEnvelope(task="x"), "single") == "medium"


def test_panel_and_fusion_prefer_provider_diversity(tmp_path):
    config = make_config(tmp_path)
    router = DeterministicOrchestrator(config)
    candidates = [
        card("openai:a", quality="frontier"),
        card("openai:d", quality="strong"),
        card("anthropic:b", quality="strong"),
        card("ollama:c", quality="strong"),
    ]

    panel = router.plan(
        PlanningContext(RequestEnvelope(task="x", strategy="panel"), candidates)
    )
    fusion = router.plan(
        PlanningContext(RequestEnvelope(task="x", strategy="fusion"), candidates)
    )

    assert len({model.split(":", 1)[0] for model in panel.steps[0].models}) == 3
    assert len({model.split(":", 1)[0] for model in fusion.steps[0].models}) == 3
    assert router._latency_estimate([card("openai:a", latency="fast")]) == 5000
    assert fusion.estimated_latency_ms == 36000


def test_model_orchestrator_survives_registry_failure(tmp_path, monkeypatch):
    class BrokenRegistry:
        def __init__(self, config):
            raise OSError("registry unavailable")

    monkeypatch.setattr(orchestrator_module, "ModelRegistry", BrokenRegistry)
    planner = ModelOrchestrator(make_config(tmp_path), adapters={})

    assert planner._cards == {}


def test_model_orchestrator_without_model_uses_deterministic_fallback(tmp_path):
    config = make_config(tmp_path)
    config.orchestrator.model = None

    plan = ModelOrchestrator(config, adapters={}).plan(context())

    assert "no orchestrator model is configured" in plan.reason


def test_orchestrated_request_is_not_treated_as_a_required_final_strategy(tmp_path):
    planner = ModelOrchestrator(make_config(tmp_path), adapters={})
    ctx = context(RequestEnvelope(task="nested task", mode="agentic", strategy="orchestrated"))

    payload = planner._planning_payload(ctx)

    assert planner._profile_strategy(ctx) is None
    assert payload["required_strategy"] is None
    assert "single" in payload["allowed_strategies"]


def test_model_orchestrator_error_mode_reports_missing_adapter(tmp_path):
    config = make_config(tmp_path)
    config.orchestrator.fallback = "error"

    with pytest.raises(CrupierRouteValidationError, match="No adapter is configured"):
        ModelOrchestrator(config, adapters={}).plan(context())


def test_model_orchestrator_does_not_swallow_budget_errors(tmp_path, monkeypatch):
    planner = ModelOrchestrator(make_config(tmp_path), adapters={})

    def fail(*args, **kwargs):
        raise CrupierBudgetExceededError("budget")

    monkeypatch.setattr(planner, "_plan_with_model", fail)
    with pytest.raises(CrupierBudgetExceededError, match="budget"):
        planner.plan(context())


def test_orchestrator_adapter_failures_are_recorded(tmp_path):
    planner = ModelOrchestrator(
        make_config(tmp_path),
        adapters={"openai": StaticAdapter(error=RuntimeError("provider down"))},
    )
    ctx = context()

    with pytest.raises(RuntimeError, match="provider down"):
        planner._call_orchestrator("openai:planner", "plan", ctx)

    record = ctx.request.metadata["_crupier_orchestrator_calls"][0]
    assert record["error_type"] == "RuntimeError"
    assert record["provider"] == "openai"


def test_planning_candidate_limit_and_nested_route_plan(tmp_path):
    planner = ModelOrchestrator(make_config(tmp_path), adapters={})
    ctx = context(RequestEnvelope(task="x", constraints={"orchestrator_candidate_limit": "bad"}))

    assert len(planner._planning_candidates(ctx)) == 3
    plan = planner._plan_from_text(
        json.dumps(
            {
                "route_plan": {
                    "strategy": "single",
                    "steps": [{"role": "primary", "model": "openai:a"}],
                    "reason": "nested",
                }
            }
        )
    )
    assert plan.models == ["openai:a"]


def test_candidate_summary_and_compact_scores_keep_only_safe_bounded_data():
    rich = card("openai:a")
    rich.modalities_input = ["text", "image"]
    rich.supports_embeddings = True
    rich.supports_file_input = True
    rich.known_edge_cases = ["one", "two", "three"]
    rich.routing_hints = {"routing_status": "recommended", "strategy_bias": ["single"]}
    rich.natural_profile = {"summary": "general", "best_for": ["a"], "avoid_for": ["b"]}
    rich.capability_status = {
        "tools": {"status": "verified"},
        "audio": {"status": "failed"},
    }

    summary = _candidate_summary(rich)
    compact = _compact_deterministic_scores(
        [
            {"model": "other:x", "score": 1, "terms": []},
            {
                "model": "openai:a",
                "score": 9,
                "terms": [
                    {"name": "small", "value": 1},
                    {"name": "large", "value": -8},
                    "ignored",
                ],
            },
        ],
        {"openai:a"},
    )

    assert summary["capabilities"] == ["embeddings", "file_input", "streaming", "structured_output", "tools"]
    assert summary["verified_capabilities"] == ["tools"]
    assert summary["failed_capabilities"] == ["audio"]
    assert compact[0]["top_terms"][0] == {"name": "large", "value": -8}


def test_request_prompt_helpers_bound_redact_and_describe_content():
    def local_tool():
        return None

    request = RequestEnvelope(
        task="x",
        input={"a_token": "sk-abcdefghijklmno", "z_extra": "x" * 80},
        messages=[{"role": "user", "content": "Bearer abcdefghijklmnop"}],
        files=[FileAsset(kind="pdf", name="x.pdf")],
        tools=[{"type": "function", "function": {"name": "lookup"}}, local_tool],
        constraints={"response_schema": {"type": "object"}},
    )

    shape = _request_shape(request)
    content = _request_content(request, limit=30)

    assert shape["file_kinds"] == ["pdf"]
    assert shape["tool_names"] == ["lookup", "local_tool"]
    assert shape["has_response_schema"] is True
    assert "[redacted]" in content["input"]
    assert content["input_truncated"] is True
    assert "messages" not in content
    assert _request_content(RequestEnvelope(task="x"), limit=-1) == {}
    assert _tool_name({"name": "direct"}) == "direct"
    assert "[redacted]" in _redact_planning_text("SERVICE_API_KEY=secret-value")


def test_json_text_falls_back_for_circular_values():
    circular = []
    circular.append(circular)

    assert _json_text(circular) == "[[...]]"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('```json\n{"ok": true}\n   ```', {"ok": True}),
        ('prefix {"ok": true} suffix', {"ok": True}),
    ],
)
def test_extract_json_object_accepts_fences_and_embedded_objects(text, expected):
    assert _extract_json_object(text) == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("no json", "did not contain"),
        ("prefix {bad} suffix", "invalid JSON"),
        ("[]", "must be a JSON object"),
    ],
)
def test_extract_json_object_rejects_invalid_responses(text, message):
    with pytest.raises(CrupierRouteValidationError, match=message):
        _extract_json_object(text)


@pytest.mark.parametrize(
    ("strategy", "step", "expected_role"),
    [
        ("cascade", {"role": "fallback", "models": ["openai:a", "anthropic:b"]}, "primary"),
        ("cascade", {"role": "fallback", "model": "openai:a"}, "primary"),
        ("single", {"role": "fallback", "models": ["openai:a"]}, "primary"),
        ("single", {"role": "fallback", "model": "openai:a"}, "primary"),
        ("fallback", {"role": "primary", "models": ["openai:a"]}, "fallback"),
        ("cascade", {"role": "escalation", "models": ["anthropic:b"]}, "escalation"),
        ("local_first", {"role": "primary", "models": ["ollama:c"]}, "primary"),
    ],
)
def test_normalize_common_route_shapes(strategy, step, expected_role):
    normalized = _normalize_common_route_shape({"strategy": strategy, "steps": [step], "reason": "x"})

    assert normalized["steps"][0]["role"] == expected_role
    assert "normalized common route role shape" in normalized["reason"]


def test_normalize_cascade_converts_duplicate_primary_and_preserves_non_dict_steps():
    normalized = _normalize_common_route_shape(
        {
            "strategy": "cascade",
            "steps": [
                "marker",
                {"role": "primary", "model": "openai:a"},
                {"role": "primary", "model": "anthropic:b"},
            ],
        }
    )

    assert normalized["steps"][0] == "marker"
    assert normalized["steps"][2]["role"] == "escalation"
    assert _normalize_common_route_shape({"strategy": "single", "steps": []})["steps"] == []
