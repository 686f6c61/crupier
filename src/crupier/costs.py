"""Cost estimation helpers.

Costs are estimates until providers return billable usage and pricing refresh is
implemented. The important product invariant is that budgets are checked before
provider execution.
"""

from __future__ import annotations

from math import ceil
from typing import Any

from .adapters.common import build_prompt
from .models import CapabilityCard, CostEstimate, RequestEnvelope, RoutePlan


DEFAULT_PRICE_BY_COST_TIER = {
    "low": {"input_per_million_usd": 0.20, "output_per_million_usd": 0.80},
    "medium": {"input_per_million_usd": 3.00, "output_per_million_usd": 12.00},
    "high": {"input_per_million_usd": 15.00, "output_per_million_usd": 75.00},
    "unknown": {"input_per_million_usd": 1.00, "output_per_million_usd": 5.00},
}


def estimate_route_cost(plan: RoutePlan, request: RequestEnvelope, cards: list[CapabilityCard]) -> CostEstimate:
    cards_by_key = {card.model_ref.key: card for card in cards}
    input_tokens = estimate_tokens(build_prompt(request))
    output_tokens = int(request.constraints.get("max_output_tokens", request.constraints.get("max_tokens", 1024)))
    total = 0.0
    for model in _planned_model_calls(plan):
        card = cards_by_key.get(model)
        total += estimate_model_cost(card, input_tokens=input_tokens, output_tokens=output_tokens)
    return CostEstimate(estimated_usd=round(total, 8))


def actual_cost_from_calls(calls: list[dict[str, Any]], cards: list[CapabilityCard]) -> float | None:
    cards_by_key = {card.model_ref.key: card for card in cards}
    total = 0.0
    any_usage = False
    for call in calls:
        model = call.get("model")
        usage = call.get("usage") or {}
        if not model or not usage:
            continue
        input_tokens = _usage_value(usage, "input_tokens", "prompt_tokens", "prompt_eval_count")
        output_tokens = _usage_value(usage, "output_tokens", "completion_tokens", "eval_count")
        if input_tokens is None and output_tokens is None:
            continue
        any_usage = True
        total += estimate_model_cost(
            cards_by_key.get(model),
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
        )
    return round(total, 8) if any_usage else None


def estimate_model_cost(card: CapabilityCard | None, *, input_tokens: int, output_tokens: int) -> float:
    pricing = _pricing(card)
    input_rate = float(pricing["input_per_million_usd"])
    output_rate = float(pricing["output_per_million_usd"])
    return (input_tokens / 1_000_000 * input_rate) + (output_tokens / 1_000_000 * output_rate)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, ceil(len(text) / 4))


def _pricing(card: CapabilityCard | None) -> dict[str, float]:
    if card is None:
        return DEFAULT_PRICE_BY_COST_TIER["unknown"]
    if card.pricing:
        input_rate = _first_number(
            card.pricing,
            "input_per_million_usd",
            "input_usd_per_million",
            "prompt_per_million_usd",
            "prompt_usd_per_million",
        )
        output_rate = _first_number(
            card.pricing,
            "output_per_million_usd",
            "output_usd_per_million",
            "completion_per_million_usd",
            "completion_usd_per_million",
        )
        if input_rate is not None and output_rate is not None:
            return {"input_per_million_usd": input_rate, "output_per_million_usd": output_rate}
    return DEFAULT_PRICE_BY_COST_TIER.get(card.cost_tier, DEFAULT_PRICE_BY_COST_TIER["unknown"])


def _planned_model_calls(plan: RoutePlan) -> list[str]:
    calls: list[str] = []
    for step in plan.steps:
        if step.model:
            calls.append(step.model)
        calls.extend(step.models)
    return calls


def _first_number(data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _usage_value(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int | float):
            return int(value)
    return None
