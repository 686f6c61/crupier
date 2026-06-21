from crupier import Crupier
from crupier.adapters import AdapterResponse, ProviderModel
from crupier.config import CrupierConfig, PolicyRule
from crupier.errors import CrupierPolicyError, CrupierProviderAuthError, CrupierRouteValidationError
from crupier.models import CapabilityCard, ModelRef, RequestEnvelope
from crupier.orchestrator import DeterministicOrchestrator, ModelOrchestrator
from crupier.planner import RoutePlanner
from crupier.policy import PolicyEngine
from crupier.registry import ModelRegistry


def make_config(tmp_path, *, allow=None):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "test", "default_profile": "agentic"},
            "providers": {
                "openai": {"enabled": True, "env_key": "OPENAI_API_KEY"},
                "anthropic": {"enabled": True, "env_key": "ANTHROPIC_API_KEY"},
                "google": {"enabled": True, "env_key": "GOOGLE_API_KEY"},
                "ollama": {"enabled": True, "host": "http://localhost:11434"},
                "openrouter": {"enabled": False, "mode": "byok"},
            },
            "models": {
                "allow": allow
                or [
                    "openai:gpt-5.5",
                    "openai:gpt-5.4-mini",
                    "anthropic:claude-opus-4-8",
                    "google:gemini-3.5-flash",
                    "ollama:qwen3.5:122b",
                ]
            },
            "routing": {
                "default_strategy": "orchestrated",
                "allow_fusion": True,
                "allow_parallel": True,
                "allow_latest_aliases": False,
                "allow_preview_models": False,
                "max_calls": 40,
                "require_operational_providers": False,
            },
            "profiles": {
                "agentic": {"prefer": ["tool_use", "coding", "long_horizon", "reliability"], "strategy": "orchestrated"},
                "private": {"prefer": ["local"], "strategy": "local_first"},
                "research": {"prefer": ["consensus", "critique"], "strategy": "fusion"},
                "structured": {"prefer": ["structured_output"], "strategy": "cascade"},
            },
            "orchestrator": {"model": "openai:gpt-5.4-mini"},
        }
    )
    config.root = tmp_path
    return config


class FakeOrchestratorAdapter:
    provider = "openai"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, *, model, prompt, request):
        self.prompts.append(prompt)
        return AdapterResponse(text=self.responses.pop(0), metadata={"model": model})


class FakeVisibleModelsAdapter:
    provider = "openai"

    def __init__(self, models):
        self.models = models

    def list_models(self):
        return [ProviderModel(id=model, provider="openai") for model in self.models]

    def generate(self, *, model, prompt, request):
        return AdapterResponse(text="ok", metadata={"model": model})


class FakeBrokenListAdapter(FakeVisibleModelsAdapter):
    def list_models(self):
        raise CrupierProviderAuthError("bad key", provider="openai", env_key="OPENAI_API_KEY")


def test_policy_filters_latest_aliases(tmp_path):
    config = make_config(tmp_path, allow=["openai:gpt-latest"])
    registry = ModelRegistry(config)
    policy = PolicyEngine(config)

    try:
        policy.filter_candidates(RequestEnvelope(task="x"), registry.allowed_cards())
    except CrupierPolicyError as exc:
        assert "latest aliases are disabled" in str(exc)
    else:
        raise AssertionError("latest alias should have been rejected")


def test_client_filters_models_not_visible_to_operational_api_key(tmp_path):
    config = make_config(tmp_path, allow=["openai:gpt-5.5", "openai:gpt-5.4-mini"])
    config.routing.require_operational_providers = True
    client = Crupier(config, adapters={"openai": FakeVisibleModelsAdapter(["gpt-5.4-mini"])})

    result = client.deal(
        task="Clasifica esto rapido",
        mode="fast",
        constraints={"require_operational_providers": True},
        trace="summary",
    )

    assert result.route is not None
    assert result.route.models == ["openai:gpt-5.4-mini"]
    assert result.trace is not None
    assert any(
        item["model"] == "openai:gpt-5.5" and "not visible" in item["reason"]
        for item in result.trace.excluded_models
    )


def test_client_blocks_provider_with_non_operational_api_key(tmp_path):
    config = make_config(tmp_path, allow=["openai:gpt-5.5"])
    config.routing.require_operational_providers = True
    client = Crupier(config, adapters={"openai": FakeBrokenListAdapter(["gpt-5.5"])})

    try:
        client.deal(task="x", constraints={"require_operational_providers": True})
    except CrupierPolicyError as exc:
        assert "not operational" in str(exc)
        assert "bad key" in str(exc)
    else:
        raise AssertionError("provider with invalid API key should be blocked before routing")


def test_policy_rejects_deprecated_models_by_default(tmp_path):
    config = make_config(tmp_path, allow=["openai:gpt-5.2-chat-latest"])
    registry = ModelRegistry(config)
    policy = PolicyEngine(config)

    try:
        policy.filter_candidates(RequestEnvelope(task="x"), registry.allowed_cards())
    except CrupierPolicyError as exc:
        assert "deprecated or shut down" in str(exc)
    else:
        raise AssertionError("deprecated model should have been rejected")


def test_private_mode_prefers_ollama_local_first(tmp_path):
    client = Crupier(make_config(tmp_path))

    result = client.deal(task="Route private work", input={"x": 1}, mode="private", trace="summary")

    assert result.route is not None
    assert result.route.strategy == "local_first"
    assert result.route.steps[0].model == "ollama:qwen3.5:122b"
    assert result.trace is not None


def test_structured_request_filters_models_without_structured_support(tmp_path):
    client = Crupier(make_config(tmp_path, allow=["openai:gpt-5.4-mini", "ollama:qwen3.5:122b"]))

    result = client.deal(task="Extract data", input="hello", mode="structured", response_schema=object, trace="summary")

    assert result.route is not None
    assert "ollama:qwen3.5:122b" not in result.route.models
    assert result.trace is not None
    assert any(item["model"] == "ollama:qwen3.5:122b" for item in result.trace.excluded_models)


def test_policy_rejects_failed_capability_probe_for_required_structured_output(tmp_path):
    config = make_config(tmp_path, allow=[])
    policy = PolicyEngine(config)
    failed = CapabilityCard(
        model_ref=ModelRef.parse("openai:gpt-5.5"),
        last_updated="test",
        supports_structured_output=True,
        capability_status={"structured_output": {"status": "failed", "source": "probe:structured_output"}},
    )

    try:
        policy.filter_candidates(RequestEnvelope(task="Extract", response_schema=object), [failed])
    except CrupierPolicyError as exc:
        assert "structured_output support is failed" in str(exc)
    else:
        raise AssertionError("failed structured_output probe should be rejected")


def test_policy_can_require_verified_capabilities(tmp_path):
    config = make_config(tmp_path, allow=[])
    policy = PolicyEngine(config)
    inferred = CapabilityCard(
        model_ref=ModelRef.parse("openai:gpt-5.5"),
        last_updated="test",
        supports_tools=True,
    )
    verified = CapabilityCard(
        model_ref=ModelRef.parse("openai:gpt-5.4-mini"),
        last_updated="test",
        supports_tools=True,
        capability_status={"tool_call": {"status": "verified", "source": "probe:tool_call"}},
    )

    result = policy.filter_candidates(
        RequestEnvelope(
            task="Use a tool",
            tools=[object()],
            constraints={"require_verified_capabilities": True},
        ),
        [inferred, verified],
    )

    assert [card.model_ref.key for card in result.allowed] == ["openai:gpt-5.4-mini"]
    assert result.excluded[0].model == "openai:gpt-5.5"


def test_declarative_policy_rule_can_deny_provider_by_mode(tmp_path):
    config = make_config(tmp_path, allow=["openai:gpt-5.5", "anthropic:claude-opus-4-8"])
    config.policy.rules = [
        PolicyRule(
            name="no_openai_agentic",
            effect="deny",
            modes=["agentic"],
            providers=["openai"],
            reason="agentic routes must use non-OpenAI provider in this project",
        )
    ]
    client = Crupier(config)

    result = client.deal("Plan agent", mode="agentic", trace="summary")

    assert result.route.models == ["anthropic:claude-opus-4-8"]
    assert result.trace is not None
    assert any(item["model"] == "openai:gpt-5.5" and "agentic routes" in item["reason"] for item in result.trace.excluded_models)


def test_declarative_policy_rule_can_require_verified_capability(tmp_path):
    config = make_config(tmp_path, allow=[])
    config.policy.rules = [
        PolicyRule(
            name="verified_tools_only",
            effect="require_verified_capability",
            capabilities=["tool_call"],
            reason="tools require verified capability",
        )
    ]
    policy = PolicyEngine(config)
    inferred = CapabilityCard(
        model_ref=ModelRef.parse("openai:gpt-5.5"),
        last_updated="test",
        supports_tools=True,
    )
    verified = CapabilityCard(
        model_ref=ModelRef.parse("openai:gpt-5.4-mini"),
        last_updated="test",
        supports_tools=True,
        capability_status={"tool_call": {"status": "verified", "source": "probe:tool_call"}},
    )

    result = policy.filter_candidates(RequestEnvelope(task="Use tool", tools=[object()]), [inferred, verified])

    assert [card.model_ref.key for card in result.allowed] == ["openai:gpt-5.4-mini"]
    assert result.excluded[0].model == "openai:gpt-5.5"
    assert "tools require verified capability" in result.excluded[0].reason


def test_research_mode_uses_fusion_when_multiple_models_allowed(tmp_path):
    client = Crupier(make_config(tmp_path, allow=["openai:gpt-5.5", "anthropic:claude-opus-4-8"]))

    result = client.deal(task="Compare architectures", mode="research")

    assert result.route is not None
    assert result.route.strategy == "fusion"
    assert len(result.route.models) == 2


def test_profile_strategy_rules_override_orchestrated_strategy(tmp_path):
    config = make_config(tmp_path, allow=["openai:gpt-5.5", "anthropic:claude-opus-4-8"])
    config.profiles["agentic"].options["strategy_rules"] = [
        {"when": {"tools": True, "max_tools": 1}, "strategy": "single"},
        {"when": {"tools": True, "min_tools": 2}, "strategy": "critique_repair"},
    ]
    client = Crupier(config)

    short = client.deal("Use one tool", mode="agentic", tools=[object()])
    long = client.deal("Use two tools", mode="agentic", tools=[object(), object()])

    assert short.route.strategy == "single"
    assert long.route.strategy == "critique_repair"


def test_force_model_uses_exact_allowed_model(tmp_path):
    client = Crupier(make_config(tmp_path, allow=["openai:gpt-5.5", "openai:gpt-5.4-mini"]))

    result = client.deal("Force cheap model", constraints={"force_model": "openai:gpt-5.4-mini"})

    assert result.route is not None
    assert result.route.models == ["openai:gpt-5.4-mini"]
    assert result.route.selection_scores


def test_force_model_rejects_model_outside_allowlist(tmp_path):
    client = Crupier(make_config(tmp_path, allow=["openai:gpt-5.4-mini"]))

    try:
        client.deal("Force disallowed model", constraints={"force_model": "anthropic:claude-opus-4-8"})
    except CrupierRouteValidationError as exc:
        assert "not allowed" in str(exc)
    else:
        raise AssertionError("forced disallowed model should fail")


def test_route_planner_builds_orchestrator_context(tmp_path):
    config = make_config(tmp_path, allow=["openai:gpt-5.5", "anthropic:claude-opus-4-8"])
    request = RequestEnvelope(
        task="Compare model routing approaches",
        mode="research",
        constraints={"selection_trace_limit": 1},
    )
    candidates = ModelRegistry(config).allowed_cards()
    planner = RoutePlanner(config)

    context = planner.build_context(request, candidates, ["allowlist:2"])

    assert context.request is request
    assert context.candidate_models == ["openai:gpt-5.5", "anthropic:claude-opus-4-8"]
    assert context.filters_applied == ["allowlist:2"]
    assert len(context.deterministic_scores) == 1
    assert context.orchestrator_mode == "deterministic"
    assert context.metadata["configured_orchestrator_model"] == "openai:gpt-5.4-mini"
    assert context.metadata["configured_orchestrator_fallback_model"] is None


def test_deterministic_orchestrator_matches_route_planner_facade(tmp_path):
    config = make_config(tmp_path, allow=["openai:gpt-5.5", "anthropic:claude-opus-4-8"])
    request = RequestEnvelope(task="Compare architectures", mode="research")
    candidates = ModelRegistry(config).allowed_cards()
    planner = RoutePlanner(config)
    context = planner.build_context(request, candidates, ["allowlist:2"])

    direct_plan = DeterministicOrchestrator(config, selector=planner.selector).plan(context)
    facade_plan = planner.plan(request, candidates, ["allowlist:2"])

    assert facade_plan.to_dict() == direct_plan.to_dict()
    assert facade_plan.strategy == "fusion"


def test_model_orchestrator_accepts_valid_validated_plan(tmp_path):
    config = make_config(tmp_path, allow=["openai:gpt-5.5", "anthropic:claude-opus-4-8"])
    config.orchestrator.mode = "model"
    request = RequestEnvelope(task="Choose a robust model route", mode="agentic")
    candidates = ModelRegistry(config).allowed_cards()
    adapter = FakeOrchestratorAdapter(
        """
        {
          "strategy": "single",
          "steps": [{"role": "primary", "model": "anthropic:claude-opus-4-8"}],
          "estimated_cost": {"estimated_usd": 0.0},
          "estimated_latency_ms": 6000,
          "reason": "Best fit for the requested robustness profile.",
          "risk_level": "medium",
          "summary": "Single Claude route."
        }
        """
    )
    planner = RoutePlanner(config)
    context = planner.build_context(request, candidates, ["allowlist:2"])

    plan = ModelOrchestrator(config, adapters={"openai": adapter}).plan(context)

    assert plan.strategy == "single"
    assert plan.models == ["anthropic:claude-opus-4-8"]
    assert plan.policy_filters_applied == ["allowlist:2"]
    assert plan.selection_scores
    assert "Model orchestrator proposed and validated" in plan.reason
    assert "candidate_cards" in adapter.prompts[0]
    assert "routing_status" in adapter.prompts[0]
    assert "best_skills" in adapter.prompts[0]
    assert "Prompt-Version: orchestrator.route_plan.v1" in adapter.prompts[0]


def test_model_orchestrator_rejects_illegal_model_and_falls_back(tmp_path):
    config = make_config(tmp_path, allow=["openai:gpt-5.5", "anthropic:claude-opus-4-8"])
    config.orchestrator.mode = "model"
    config.orchestrator.max_repairs = 0
    request = RequestEnvelope(task="Compare architectures", mode="research")
    candidates = ModelRegistry(config).allowed_cards()
    adapter = FakeOrchestratorAdapter(
        """
        {
          "strategy": "single",
          "steps": [{"role": "primary", "model": "openai:not-allowed"}],
          "estimated_cost": {"estimated_usd": 0.0},
          "reason": "Invalid on purpose.",
          "risk_level": "medium",
          "summary": "Invalid route."
        }
        """
    )
    context = RoutePlanner(config).build_context(request, candidates, ["allowlist:2"])

    plan = ModelOrchestrator(config, adapters={"openai": adapter}).plan(context)

    assert "openai:not-allowed" not in plan.models
    assert plan.strategy == "fusion"
    assert "deterministic fallback" in plan.reason


def test_model_orchestrator_uses_fallback_model_before_deterministic(tmp_path):
    config = make_config(tmp_path, allow=["openai:gpt-5.5", "anthropic:claude-opus-4-8"])
    config.orchestrator.mode = "model"
    config.orchestrator.fallback_model = "anthropic:claude-opus-4-8"
    config.orchestrator.max_repairs = 0
    request = RequestEnvelope(task="Choose a robust model route", mode="agentic")
    candidates = ModelRegistry(config).allowed_cards()
    primary = FakeOrchestratorAdapter(
        """
        {
          "strategy": "single",
          "steps": [{"role": "primary", "model": "openai:not-allowed"}],
          "estimated_cost": {"estimated_usd": 0.0},
          "reason": "Invalid on purpose.",
          "risk_level": "medium",
          "summary": "Invalid route."
        }
        """
    )
    fallback = FakeOrchestratorAdapter(
        """
        {
          "strategy": "single",
          "steps": [{"role": "primary", "model": "anthropic:claude-opus-4-8"}],
          "estimated_cost": {"estimated_usd": 0.0},
          "estimated_latency_ms": 6000,
          "reason": "Fallback orchestrator selected the robust model.",
          "risk_level": "medium",
          "summary": "Single Claude route."
        }
        """
    )
    context = RoutePlanner(config).build_context(request, candidates, ["allowlist:2"])

    plan = ModelOrchestrator(config, adapters={"openai": primary, "anthropic": fallback}).plan(context)

    assert plan.strategy == "single"
    assert plan.models == ["anthropic:claude-opus-4-8"]
    assert "Fallback orchestrator anthropic:claude-opus-4-8 was used" in plan.reason
    assert "deterministic fallback" not in plan.reason
    assert len(primary.prompts) == 1
    assert len(fallback.prompts) == 1


def test_model_orchestrator_normalizes_common_cascade_role_shape(tmp_path):
    config = make_config(tmp_path, allow=["openai:gpt-5.4-mini", "anthropic:claude-opus-4-8"])
    config.orchestrator.mode = "model"
    request = RequestEnvelope(task="Extract JSON fields", mode="structured")
    candidates = ModelRegistry(config).allowed_cards()
    adapter = FakeOrchestratorAdapter(
        """
        {
          "strategy": "cascade",
          "steps": [{"role": "fallback", "models": ["openai:gpt-5.4-mini", "anthropic:claude-opus-4-8"]}],
          "estimated_cost": {"estimated_usd": 0.0},
          "estimated_latency_ms": 6000,
          "reason": "Use a cheap first model and escalate.",
          "risk_level": "medium",
          "summary": "Cascade route."
        }
        """
    )
    context = RoutePlanner(config).build_context(request, candidates, ["allowlist:2"])

    plan = ModelOrchestrator(config, adapters={"openai": adapter}).plan(context)

    assert plan.strategy == "cascade"
    assert [step.role for step in plan.steps] == ["primary", "escalation"]
    assert plan.models == ["openai:gpt-5.4-mini", "anthropic:claude-opus-4-8"]
    assert "normalized common route role shape" in plan.reason


def test_model_orchestrator_respects_explicit_profile_strategy(tmp_path):
    config = make_config(tmp_path, allow=["openai:gpt-5.5", "ollama:qwen3.5:122b"])
    config.orchestrator.mode = "model"
    request = RequestEnvelope(task="Private routing", mode="private")
    candidates = ModelRegistry(config).allowed_cards()
    adapter = FakeOrchestratorAdapter(
        """
        {
          "strategy": "single",
          "steps": [{"role": "primary", "model": "ollama:qwen3.5:122b"}],
          "estimated_cost": {"estimated_usd": 0.0},
          "reason": "Invalid because private profile requires local_first.",
          "risk_level": "medium",
          "summary": "Invalid private route."
        }
        """,
        """
        {
          "strategy": "single",
          "steps": [{"role": "primary", "model": "ollama:qwen3.5:122b"}],
          "estimated_cost": {"estimated_usd": 0.0},
          "reason": "Still invalid.",
          "risk_level": "medium",
          "summary": "Still invalid."
        }
        """,
    )
    context = RoutePlanner(config).build_context(request, candidates, ["allowlist:2"])

    plan = ModelOrchestrator(config, adapters={"openai": adapter}).plan(context)

    assert plan.strategy == "local_first"
    assert plan.models[0] == "ollama:qwen3.5:122b"
    assert "deterministic fallback" in plan.reason


def test_model_orchestrator_rejects_agentic_high_risk_strategy_downgrade(tmp_path):
    config = make_config(tmp_path, allow=["openai:gpt-5.5", "anthropic:claude-opus-4-8"])
    config.orchestrator.mode = "model"
    config.orchestrator.max_repairs = 0
    request = RequestEnvelope(
        task="Plan a code-changing agent step with rollback risks.",
        mode="agentic",
        constraints={"risk_level": "high"},
    )
    candidates = ModelRegistry(config).allowed_cards()
    adapter = FakeOrchestratorAdapter(
        """
        {
          "strategy": "cascade",
          "steps": [{"role": "primary", "model": "anthropic:claude-opus-4-8"}],
          "estimated_cost": {"estimated_usd": 0.0},
          "reason": "Invalid downgrade for a high-risk agentic route.",
          "risk_level": "high",
          "summary": "Invalid cascade route."
        }
        """
    )
    context = RoutePlanner(config).build_context(request, candidates, ["allowlist:2"])

    plan = ModelOrchestrator(config, adapters={"openai": adapter}).plan(context)

    assert plan.strategy == "critique_repair"
    assert "deterministic fallback" in plan.reason
    assert "strategy_guardrail" in adapter.prompts[0]
