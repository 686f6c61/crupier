"""Card-driven runtime controls applied immediately before provider calls."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .model_profiles import classify_task_signal_weights
from .models import CapabilityCard, RequestEnvelope

_COMPLEX_SIGNALS = {"reasoning", "math", "coding", "research", "critique", "long_context"}
_EXPLICIT_REASONING_KEYS = {"enable_thinking", "disable_thinking", "reasoning_effort"}


def apply_runtime_policy(
    model_key: str,
    request: RequestEnvelope,
    card: CapabilityCard | None,
) -> tuple[RequestEnvelope, dict[str, Any]]:
    """Derive model-family controls without overriding explicit caller choices."""

    if card is None:
        return request, {}
    constraints = dict(request.constraints)
    behavior = card.routing_hints.get("reasoning")
    effort_options = card.routing_hints.get("reasoning_effort")
    explicit = sorted(_EXPLICIT_REASONING_KEYS.intersection(constraints))
    policy: dict[str, Any] = {"model": model_key}
    if explicit:
        policy.update({"source": "request", "explicit_controls": explicit})
        return request, policy

    signal_request = replace(request, mode=None)
    signals = classify_task_signal_weights(signal_request)
    complex_values = [signals.get(name, 0.0) for name in _COMPLEX_SIGNALS]
    strong_signal_count = sum(value >= 0.6 for value in complex_values)
    complexity = min(1.0, max(complex_values, default=0.0) + max(0, strong_signal_count - 1) * 0.1)
    high_reasoning_need = complexity >= 0.6 or bool(request.tools) or request.constraints.get("risk_level") == "high"
    max_output = _positive_int(constraints.get("max_output_tokens", constraints.get("max_tokens")))

    if isinstance(effort_options, list) and effort_options:
        if request.mode in {"cheap", "fast"} or (max_output is not None and max_output <= 256):
            effort = "low"
        elif high_reasoning_need and complexity >= 0.8:
            effort = "high"
        else:
            effort = "medium"
        if effort in effort_options:
            constraints["reasoning_effort"] = effort
            policy.update(
                {
                    "source": "capability_card",
                    "reasoning_effort": effort,
                    "complexity": round(complexity, 3),
                }
            )

    if behavior in {"enabled_by_default", "opt_in"}:
        enabled = high_reasoning_need
        reason = "complex_request" if enabled else "routine_request"
        if request.mode in {"cheap", "fast"}:
            enabled = False
            reason = f"{request.mode}_profile"
        if max_output is not None and max_output <= 256:
            enabled = False
            reason = "tight_output_budget"
        constraints["enable_thinking"] = enabled
        policy.update(
            {
                "source": "capability_card",
                "thinking_enabled": enabled,
                "reason": reason,
                "complexity": round(complexity, 3),
            }
        )
    elif behavior == "always_enabled":
        policy.update(
            {
                "source": "capability_card",
                "thinking_enabled": True,
                "reason": "model_always_reasons",
                "complexity": round(complexity, 3),
            }
        )
        if max_output is not None and max_output < 300:
            policy["warning"] = "output budget below the model's documented 300-token reasoning floor"

    if constraints == request.constraints:
        return request, policy if len(policy) > 1 else {}
    return replace(request, constraints=constraints), policy


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
