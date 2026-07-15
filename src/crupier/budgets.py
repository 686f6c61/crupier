"""Shared request budget accounting for planning and route execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock
from time import perf_counter

from .config import CrupierConfig
from .costs import estimate_model_cost, estimate_tokens
from .errors import CrupierBudgetExceededError, CrupierExecutionLimitError
from .models import CapabilityCard, RequestEnvelope


@dataclass(slots=True)
class BudgetReservation:
    estimated_usd: float
    timeout_seconds: float | None


class ExecutionBudget:
    """Thread-safe call, cost, and latency budget for a complete ``deal``."""

    def __init__(
        self,
        config: CrupierConfig,
        request: RequestEnvelope,
        cards: list[CapabilityCard],
        *,
        started_at: float | None = None,
    ):
        self._cards = {card.model_ref.key: card for card in cards}
        self._lock = RLock()
        self._started = started_at if started_at is not None else perf_counter()
        self.max_calls = _non_negative_int(
            request.constraints.get("max_calls", config.routing.max_calls),
            default=config.routing.max_calls,
        )
        self.max_cost_usd = _optional_non_negative_float(
            request.constraints.get("max_cost_usd", config.routing.max_cost_per_request_usd)
        )
        self.max_latency_ms = _optional_non_negative_float(
            request.constraints.get("max_latency_ms", config.routing.max_latency_ms)
        )
        self.calls_started = 0
        self.estimated_cost_reserved_usd = 0.0

    def reserve(self, *, model: str, prompt: str, request: RequestEnvelope) -> BudgetReservation:
        output_tokens = _positive_int(
            request.constraints.get("max_output_tokens", request.constraints.get("max_tokens", 1024)),
            default=1024,
        )
        estimate = estimate_model_cost(
            self._cards.get(model),
            input_tokens=estimate_tokens(prompt),
            output_tokens=output_tokens,
        )
        return self.reserve_call(estimated_usd=estimate)

    def reserve_call(self, *, estimated_usd: float = 0.0) -> BudgetReservation:
        """Reserve one provider call when token-based estimation does not apply."""

        estimate = max(0.0, float(estimated_usd))
        with self._lock:
            remaining = self._remaining_seconds_unlocked()
            if remaining is not None and remaining <= 0:
                raise CrupierExecutionLimitError(
                    f"Route exceeded max_latency_ms={self.max_latency_ms:g} before the next provider call."
                )
            if self.calls_started >= self.max_calls:
                raise CrupierExecutionLimitError(
                    f"Route exhausted max_calls={self.max_calls}; no further provider calls are allowed."
                )
            projected_cost = self.estimated_cost_reserved_usd + estimate
            if self.max_cost_usd is not None and projected_cost > self.max_cost_usd:
                raise CrupierBudgetExceededError(
                    f"Next provider call would reserve ${projected_cost:.4f}, above max ${self.max_cost_usd:.4f}."
                )
            self.calls_started += 1
            self.estimated_cost_reserved_usd = projected_cost
            return BudgetReservation(estimated_usd=estimate, timeout_seconds=remaining)

    def ensure_deadline(self) -> None:
        with self._lock:
            remaining = self._remaining_seconds_unlocked()
            if remaining is not None and remaining < 0:
                raise CrupierExecutionLimitError(
                    f"Route exceeded max_latency_ms={self.max_latency_ms:g} during a provider call."
                )

    def remaining_calls(self) -> int:
        with self._lock:
            return max(0, self.max_calls - self.calls_started)

    def remaining_cost_usd(self) -> float | None:
        with self._lock:
            if self.max_cost_usd is None:
                return None
            return max(0.0, self.max_cost_usd - self.estimated_cost_reserved_usd)

    def remaining_latency_ms(self) -> int | None:
        with self._lock:
            remaining = self._remaining_seconds_unlocked()
            return None if remaining is None else max(0, int(remaining * 1000))

    def absorb(self, snapshot: dict[str, object]) -> None:
        with self._lock:
            self.calls_started += _non_negative_int(snapshot.get("calls_started"), default=0)
            self.estimated_cost_reserved_usd += _optional_non_negative_float(
                snapshot.get("estimated_cost_reserved_usd")
            ) or 0.0

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "calls_started": self.calls_started,
                "max_calls": self.max_calls,
                "estimated_cost_reserved_usd": round(self.estimated_cost_reserved_usd, 8),
                "max_cost_usd": self.max_cost_usd,
                "max_latency_ms": self.max_latency_ms,
                "elapsed_ms": int((perf_counter() - self._started) * 1000),
            }

    def _remaining_seconds_unlocked(self) -> float | None:
        if self.max_latency_ms is None:
            return None
        return (self.max_latency_ms / 1000.0) - (perf_counter() - self._started)


def request_with_timeout(request: RequestEnvelope, remaining_seconds: float | None) -> RequestEnvelope:
    if remaining_seconds is None:
        return request
    constraints = dict(request.constraints)
    configured = constraints.get("timeout_seconds", constraints.get("timeout"))
    try:
        configured_seconds = float(configured) if configured is not None else None
    except (TypeError, ValueError):
        configured_seconds = None
    timeout_seconds = max(0.001, remaining_seconds)
    if configured_seconds is not None and configured_seconds > 0:
        timeout_seconds = min(timeout_seconds, configured_seconds)
    constraints["timeout_seconds"] = timeout_seconds
    return replace(request, constraints=constraints)


def _non_negative_int(value: object, *, default: int) -> int:
    try:
        return max(0, int(value))  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return max(0, int(default))


def _positive_int(value: object, *, default: int) -> int:
    return max(1, _non_negative_int(value, default=default))


def _optional_non_negative_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
