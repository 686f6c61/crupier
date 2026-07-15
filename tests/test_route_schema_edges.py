import pytest

from crupier.errors import CrupierRouteValidationError
from crupier.models import CostEstimate, RoutePlan, RouteStep
from crupier.route_schema import planned_call_count, validate_route_plan_shape


def _plan(strategy="single", steps=None, **kwargs):
    return RoutePlan(
        strategy=strategy,
        steps=steps if steps is not None else [RouteStep(role="primary", model="openai:test")],
        **kwargs,
    )


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (_plan(strategy="unknown"), "Unsupported route strategy"),
        (_plan(steps=[]), "at least one step"),
        (_plan(steps=[RouteStep(role="primary")]), "at least one model"),
        (_plan(risk_level="critical"), "risk_level"),
        (_plan(estimated_cost=CostEstimate(actual_usd=-1)), "actual cost cannot be negative"),
        (_plan(estimated_latency_ms=-1), "latency cannot be negative"),
        (_plan(steps=[RouteStep(role="", model="openai:test")]), "missing a role"),
        (
            _plan(
                strategy="fallback",
                steps=[
                    RouteStep(role="fallback", model="openai:a"),
                    RouteStep(role="primary", model="openai:b"),
                    RouteStep(role="primary", model="openai:c"),
                ],
            ),
            "appears more than once",
        ),
        (
            _plan(steps=[RouteStep(role="primary", model="openai:a", models=["openai:b"])]),
            "cannot set both model and models",
        ),
        (_plan(steps=[RouteStep(role="primary")]), "at least one model"),
        (_plan(steps=[RouteStep(role="primary", model="missing-provider")]), "provider:model form"),
        (_plan(steps=[RouteStep(role="primary", model="openai:a", timeout_ms=0)]), "timeout must be positive"),
    ],
)
def test_route_shape_rejects_invalid_invariants(plan, message):
    with pytest.raises(CrupierRouteValidationError, match=message):
        validate_route_plan_shape(plan)


@pytest.mark.parametrize(
    ("strategy", "steps", "message"),
    [
        ("single", [RouteStep(role="primary", model="openai:a")], None),
        ("cascade", [RouteStep(role="validator", model="openai:a")], "requires a primary"),
        ("local_first", [RouteStep(role="fallback", model="openai:a")], "requires a primary"),
        ("fallback", [RouteStep(role="primary", model="openai:a")], "requires a fallback"),
        ("panel", [RouteStep(role="panel", models=["openai:a", "anthropic:b"])], None),
        ("fusion", [RouteStep(role="panel", models=["openai:a"])], "missing roles: final_writer, judge"),
        (
            "critique_repair",
            [RouteStep(role="generator", model="openai:a")],
            "missing roles: critic, repair",
        ),
        ("delegate", [RouteStep(role="delegate", model="openai:a")], None),
    ],
)
def test_route_shape_enforces_strategy_roles(strategy, steps, message):
    plan = _plan(strategy=strategy, steps=steps)
    if message is None:
        validate_route_plan_shape(plan)
    else:
        with pytest.raises(CrupierRouteValidationError, match=message):
            validate_route_plan_shape(plan)


def test_route_shape_counts_panel_models_and_repeated_fallbacks():
    plan = _plan(
        strategy="fallback",
        steps=[
            RouteStep(role="primary", model="openai:a"),
            RouteStep(role="fallback", model="anthropic:b"),
            RouteStep(role="fallback", models=["google:c", "ollama:d"]),
        ],
    )

    assert planned_call_count(plan) == 4
    validate_route_plan_shape(plan, max_calls=4)
    with pytest.raises(CrupierRouteValidationError, match="above max_calls=3"):
        validate_route_plan_shape(plan, max_calls=3)


@pytest.mark.parametrize("strategy", ["panel", "fusion"])
def test_route_shape_requires_two_panel_models(strategy):
    steps = [RouteStep(role="panel", models=["openai:a"])]
    if strategy == "fusion":
        steps.extend(
            [
                RouteStep(role="judge", model="openai:b"),
                RouteStep(role="final_writer", model="openai:c"),
            ]
        )
    with pytest.raises(CrupierRouteValidationError, match="at least two panel models"):
        validate_route_plan_shape(_plan(strategy=strategy, steps=steps))
