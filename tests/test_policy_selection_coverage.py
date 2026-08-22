import pytest

from crupier.config import CrupierConfig, PolicyRule
from crupier.errors import CrupierConfigError, CrupierRouteValidationError
from crupier.models import (
    CapabilityCard,
    FileRoutingPlan,
    ModelRef,
    RequestEnvelope,
    RoutePlan,
    RouteStep,
)
from crupier.policy import (
    PolicyEngine,
    PolicyResult,
    _declared_capability,
    _declared_file_capability,
    _string_set,
)
from crupier.route_schema import (
    _validate_panel_cardinality,
    _validate_required_roles,
    validate_route_plan_shape,
)
from crupier.runtime_policy import apply_runtime_policy
from crupier.selector import ModelSelector
from crupier.selector import (
    _declared_file_capability as selector_declared_file_capability,
)


def _config() -> CrupierConfig:
    return CrupierConfig.from_dict(
        {
            "providers": {
                "openai": {"enabled": True, "env_key": "OPENAI_API_KEY"},
                "anthropic": {"enabled": True, "env_key": "ANTHROPIC_API_KEY"},
            },
            "routing": {"allow_parallel": False, "max_calls": 8},
        }
    )


def _card(model: str = "openai:test", **kwargs: object) -> CapabilityCard:
    return CapabilityCard(model_ref=ModelRef.parse(model), last_updated="test", **kwargs)


def test_policy_rejects_non_array_rules_before_any_model_is_allowed() -> None:
    config = _config()
    config.policy.rules = None  # type: ignore[assignment]

    with pytest.raises(CrupierConfigError, match="must be an array"):
        PolicyEngine(config).filter_candidates(RequestEnvelope(task="route"), [_card()])


def test_policy_marks_panel_sequential_when_parallel_execution_is_disabled() -> None:
    config = _config()
    plan = RoutePlan(
        strategy="panel",
        steps=[RouteStep(role="panel", models=["openai:a", "anthropic:b"])],
    )
    allowed = PolicyResult(allowed=[_card("openai:a"), _card("anthropic:b")])

    PolicyEngine(config).validate_route(plan, allowed, RequestEnvelope(task="compare"))

    assert plan.policy_filters_applied == ["sequential_panel_execution"]


def test_policy_panel_size_defenses_allow_absent_panel_and_reject_oversize() -> None:
    no_panel = RoutePlan(
        strategy="fusion",
        steps=[
            RouteStep(role="judge", model="openai:a"),
            RouteStep(role="final_writer", model="anthropic:b"),
        ],
    )
    PolicyEngine._validate_panel_size_constraints(no_panel, RequestEnvelope(task="compare"))
    with pytest.raises(CrupierRouteValidationError, match="missing roles: panel"):
        validate_route_plan_shape(no_panel)

    oversized = RoutePlan(
        strategy="panel",
        steps=[RouteStep(role="panel", models=["openai:a", "anthropic:b", "openai:c"])],
    )
    with pytest.raises(CrupierRouteValidationError, match="above max_panel_size=2"):
        PolicyEngine._validate_panel_size_constraints(
            oversized,
            RequestEnvelope(task="compare", constraints={"max_panel_size": 2}),
        )


def test_policy_denies_unknown_required_capability_and_ignores_nonmatching_mode() -> None:
    config = _config()
    rule = PolicyRule(
        name="known_capability_only",
        effect="require_capability",
        modes=["agentic"],
        models=["openai:test"],
        capabilities=["quantum_input"],
        reason="capability must be declared",
    )
    config.policy.rules = [rule]
    result = PolicyEngine(config).filter_candidates(
        RequestEnvelope(task="route", mode="agentic"),
        [_card(), _card("anthropic:safe")],
    )

    assert [card.model_ref.key for card in result.allowed] == ["anthropic:safe"]
    assert result.excluded[0].model == "openai:test"
    assert "quantum_input support is unknown" in result.excluded[0].reason
    assert PolicyEngine._rule_matches(rule, _card(), RequestEnvelope(task="route", mode="fast")) is False


def test_policy_file_capability_decisions_cover_absent_failed_and_declared_inputs() -> None:
    card = _card(modalities_input=["text", "video"])
    request_without_files = RequestEnvelope(task="route")
    assert PolicyEngine._file_input_rejection_reason(card, request_without_files, False) is None

    request_with_audio = RequestEnvelope(
        task="route",
        file_plan=FileRoutingPlan(required_model_capabilities=["audio_input"]),
    )
    rejection = PolicyEngine._file_input_rejection_reason(card, request_with_audio, False)
    assert rejection is not None
    assert "requires audio_input" in rejection
    assert _declared_file_capability(card, "video_input") is True
    assert _declared_file_capability(card, "quantum_input") is False


@pytest.mark.parametrize(
    ("capability", "card"),
    [
        ("structured_output", _card(supports_structured_output=True)),
        ("streaming", _card(supports_streaming=True)),
        ("embeddings", _card(supports_embeddings=True)),
        ("vision_input", _card(modalities_input=["text", "image"])),
        ("unknown", _card()),
    ],
)
def test_policy_declared_capability_returns_the_security_decision(capability: str, card: CapabilityCard) -> None:
    expected = capability != "unknown"
    assert _declared_capability(card, capability) is expected


def test_policy_allowed_scope_parser_accepts_one_string_and_rejects_mapping() -> None:
    assert _string_set("openai:test") == {"openai:test"}
    with pytest.raises(CrupierRouteValidationError, match="string or list of strings"):
        _string_set({"openai": "test"})


@pytest.mark.parametrize(
    ("mode", "max_output_tokens", "expected_effort", "expected_reason"),
    [
        ("cheap", 1000, "low", "cheap_profile"),
        ("agentic", 1000, "medium", "routine_request"),
    ],
)
def test_runtime_policy_selects_profile_effort_and_thinking_decision(
    mode: str,
    max_output_tokens: int,
    expected_effort: str,
    expected_reason: str,
) -> None:
    card = _card(
        routing_hints={
            "reasoning": "enabled_by_default",
            "reasoning_effort": ["low", "medium", "high"],
        }
    )
    updated, policy = apply_runtime_policy(
        card.model_ref.key,
        RequestEnvelope(
            task="Classify this request",
            mode=mode,
            constraints={"max_output_tokens": max_output_tokens},
        ),
        card,
    )

    assert updated.constraints["reasoning_effort"] == expected_effort
    assert updated.constraints["enable_thinking"] is False
    assert policy["reason"] == expected_reason


def test_runtime_policy_warns_when_always_on_reasoning_exceeds_output_budget() -> None:
    card = _card(routing_hints={"reasoning": "always_enabled"})

    updated, policy = apply_runtime_policy(
        card.model_ref.key,
        RequestEnvelope(task="Answer", constraints={"max_output_tokens": 299}),
        card,
    )

    assert updated.constraints == {"max_output_tokens": 299}
    assert policy["thinking_enabled"] is True
    assert "300-token reasoning floor" in policy["warning"]


def test_selector_applies_lifecycle_penalties_and_skips_invalid_feedback() -> None:
    selector = ModelSelector(_config())
    card = _card(
        "openai:test-preview",
        deprecation={"reason": "retired"},
        local_eval_scores={"agentic": "invalid", "human:agentic": object()},
    )

    score = selector.score(RequestEnvelope(task="route", mode="agentic"), card)
    terms = {term.name: term for term in score.terms}

    assert terms["deprecation_penalty"].value == selector.scoring.deprecation_penalty
    assert terms["stability_penalty"].value == selector.scoring.preview_stability_penalty
    assert "local_eval" not in terms
    assert "human_feedback" not in terms


def test_selector_budget_decisions_cover_invalid_over_and_within_budget() -> None:
    selector = ModelSelector(_config())
    card = _card(pricing={"input_per_million_usd": 0, "output_per_million_usd": 1})

    invalid = selector._budget_fit_score(
        RequestEnvelope(task="route", constraints={"max_cost_usd": "invalid"}), card
    )
    over = selector._budget_fit_score(
        RequestEnvelope(task="route", constraints={"max_cost_usd": 0.0005, "max_output_tokens": 1000}), card
    )
    within = selector._budget_fit_score(
        RequestEnvelope(task="route", constraints={"max_cost_usd": 0.0015, "max_output_tokens": 1000}), card
    )

    assert invalid is None
    assert over is not None and over[0] == "budget_fit_penalty"
    assert within is not None and within[0] == "budget_fit" and "within budget" in within[2]
    assert selector._tier_weight("unsupported", "unknown") == 0


def test_selector_file_capability_decisions_cover_video_and_unknown() -> None:
    card = _card(modalities_input=["text", "video"])

    assert selector_declared_file_capability(card, "video_input") is True
    assert selector_declared_file_capability(card, "unknown_input") is False


def test_route_schema_rejects_empty_step_model_with_specific_decision() -> None:
    plan = RoutePlan(
        strategy="fallback",
        steps=[
            RouteStep(role="primary", model="openai:a"),
            RouteStep(role="fallback"),
        ],
    )

    with pytest.raises(CrupierRouteValidationError, match="has no model"):
        validate_route_plan_shape(plan)


@pytest.mark.parametrize(
    ("strategy", "seen_roles", "message"),
    [
        ("panel", set(), "requires a panel step"),
        ("delegate", set(), "requires a delegate step"),
    ],
)
def test_route_schema_required_role_defenses_reject_missing_role(
    strategy: str, seen_roles: set[str], message: str
) -> None:
    with pytest.raises(CrupierRouteValidationError, match=message):
        _validate_required_roles(strategy, seen_roles)


def test_route_schema_panel_cardinality_defense_allows_absent_panel_step() -> None:
    plan = RoutePlan(strategy="panel", steps=[RouteStep(role="panel", models=["openai:a", "anthropic:b"])])
    plan.steps[0].role = "fallback"

    _validate_panel_cardinality(plan)
    with pytest.raises(CrupierRouteValidationError, match="not valid for strategy 'panel'"):
        validate_route_plan_shape(plan)
