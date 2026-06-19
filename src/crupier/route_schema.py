"""Strict RoutePlan shape validation.

This module deliberately stays dependency-free. It is the hard schema layer
between free-form planning sources, such as an LLM orchestrator, and provider
execution.
"""

from __future__ import annotations

from .errors import CrupierRouteValidationError
from .models import RoutePlan


ALLOWED_STRATEGIES = {
    "single",
    "fallback",
    "cascade",
    "panel",
    "fusion",
    "critique_repair",
    "local_first",
}
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}
ALLOWED_ROLES_BY_STRATEGY = {
    "single": {"primary"},
    "fallback": {"fallback", "primary"},
    "cascade": {"primary", "escalation", "validator"},
    "panel": {"panel"},
    "fusion": {"panel", "judge", "final_writer"},
    "critique_repair": {"generator", "critic", "repair"},
    "local_first": {"primary", "fallback"},
}


def validate_route_plan_shape(plan: RoutePlan, *, max_calls: int | None = None) -> None:
    """Validate structural RoutePlan invariants before policy/provider checks."""

    if plan.strategy not in ALLOWED_STRATEGIES:
        raise CrupierRouteValidationError(f"Unsupported route strategy {plan.strategy!r}.")
    if not plan.steps:
        raise CrupierRouteValidationError("Route plan must include at least one step.")
    if not plan.models:
        raise CrupierRouteValidationError("Route plan must include at least one model.")
    if plan.risk_level not in ALLOWED_RISK_LEVELS:
        raise CrupierRouteValidationError(f"Route risk_level {plan.risk_level!r} is not supported.")
    if plan.estimated_cost.estimated_usd < 0:
        raise CrupierRouteValidationError("Route estimated cost cannot be negative.")
    if plan.estimated_cost.actual_usd is not None and plan.estimated_cost.actual_usd < 0:
        raise CrupierRouteValidationError("Route actual cost cannot be negative.")
    if plan.estimated_latency_ms is not None and plan.estimated_latency_ms < 0:
        raise CrupierRouteValidationError("Route estimated latency cannot be negative.")

    allowed_roles = ALLOWED_ROLES_BY_STRATEGY[plan.strategy]
    seen_roles: set[str] = set()
    for index, step in enumerate(plan.steps):
        if not step.role:
            raise CrupierRouteValidationError(f"Route step {index} is missing a role.")
        if step.role not in allowed_roles:
            raise CrupierRouteValidationError(
                f"Route step role {step.role!r} is not valid for strategy {plan.strategy!r}."
            )
        if step.role in seen_roles and step.role not in {"fallback", "escalation"}:
            raise CrupierRouteValidationError(f"Route role {step.role!r} appears more than once.")
        seen_roles.add(step.role)
        if step.model and step.models:
            raise CrupierRouteValidationError(f"Route step {step.role!r} cannot set both model and models.")
        if not step.model and not step.models:
            raise CrupierRouteValidationError(f"Route step {step.role!r} has no model.")
        for model in [step.model, *step.models]:
            if model and ":" not in model:
                raise CrupierRouteValidationError(f"Route model {model!r} must use provider:model form.")
        if step.timeout_ms is not None and step.timeout_ms <= 0:
            raise CrupierRouteValidationError(f"Route step {step.role!r} timeout must be positive.")

    _validate_required_roles(plan.strategy, seen_roles)
    if max_calls is not None:
        planned_calls = planned_call_count(plan)
        if planned_calls > max_calls:
            raise CrupierRouteValidationError(f"Route plans {planned_calls} calls, above max_calls={max_calls}.")


def planned_call_count(plan: RoutePlan) -> int:
    return sum(max(1, len(step.models) if step.models else 1) for step in plan.steps)


def _validate_required_roles(strategy: str, seen_roles: set[str]) -> None:
    if strategy in {"single", "cascade", "local_first"} and "primary" not in seen_roles:
        raise CrupierRouteValidationError(f"Route strategy {strategy!r} requires a primary step.")
    if strategy == "fallback" and "fallback" not in seen_roles:
        raise CrupierRouteValidationError("Route strategy 'fallback' requires a fallback step.")
    if strategy == "panel" and "panel" not in seen_roles:
        raise CrupierRouteValidationError("Route strategy 'panel' requires a panel step.")
    if strategy == "fusion":
        missing = {"panel", "judge", "final_writer"} - seen_roles
        if missing:
            raise CrupierRouteValidationError("Route strategy 'fusion' is missing roles: " + ", ".join(sorted(missing)))
    if strategy == "critique_repair":
        missing = {"generator", "critic", "repair"} - seen_roles
        if missing:
            raise CrupierRouteValidationError(
                "Route strategy 'critique_repair' is missing roles: " + ", ".join(sorted(missing))
            )
