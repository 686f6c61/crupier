"""Request-constraint contract and validation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import CrupierConfig
from .errors import CrupierRouteValidationError
from .models import RequestEnvelope

SUPPORTED_CONSTRAINTS = frozenset(
    {
        "allow_deprecated_models",
        "allow_latest_aliases",
        "allow_local_file_uris",
        "allow_parallel",
        "allow_preview_models",
        "allowed_models",
        "allowed_providers",
        "allowed_tools",
        "approval_ttl_seconds",
        "approve_tool_calls",
        "approved_tools",
        "cascade_min_output_chars",
        "cascade_validator_model",
        "compat",
        "compat_mode",
        "disable_thinking",
        "dry_run",
        "enable_thinking",
        "extra_body",
        "file_root",
        "file_strategy",
        "force_model",
        "human_approval_granted",
        "include_raw_outputs",
        "include_tool_error_detail",
        "logging_mode",
        "max_calls",
        "max_cell_chars",
        "max_cost_usd",
        "max_depth",
        "max_file_bytes",
        "max_file_context_chars",
        "max_latency_ms",
        "max_native_file_bytes",
        "max_output_tokens",
        "max_panel_size",
        "max_parallel_models",
        "max_pdf_pages",
        "max_provider_retries",
        "max_document_tables",
        "max_table_sheets",
        "max_tokens",
        "max_tool_calls_per_round",
        "max_tool_result_chars",
        "max_tool_rounds",
        "max_table_columns",
        "max_table_rows",
        "min_panel_size",
        "model_kind",
        "ocr_timeout_seconds",
        "orchestrator_candidate_limit",
        "reasoning_effort",
        "requested_model",
        "require_approval_for",
        "require_native_file_input",
        "require_operational_providers",
        "require_streaming",
        "require_verified_capabilities",
        "require_zdr",
        "requires_human_approval",
        "requires_tools",
        "response_schema",
        "response_schema_name",
        "retry_backoff_seconds",
        "retry_jitter_seconds",
        "risk_level",
        "selection_trace_limit",
        "store_prompt",
        "store_response",
        "store_trace",
        "stream",
        "strict_constraints",
        "strict_response_schema",
        "temperature",
        "thinking_budget",
        "thinking_config",
        "thinking_level",
        "timeout",
        "timeout_seconds",
        "top_p",
    }
)

BOOLEAN_CONSTRAINTS = frozenset(
    {
        "allow_deprecated_models",
        "allow_latest_aliases",
        "allow_local_file_uris",
        "allow_parallel",
        "allow_preview_models",
        "approve_tool_calls",
        "disable_thinking",
        "enable_thinking",
        "human_approval_granted",
        "include_raw_outputs",
        "include_tool_error_detail",
        "require_native_file_input",
        "require_operational_providers",
        "require_streaming",
        "require_verified_capabilities",
        "require_zdr",
        "requires_human_approval",
        "requires_tools",
        "store_prompt",
        "store_response",
        "store_trace",
        "stream",
        "strict_constraints",
        "strict_response_schema",
    }
)

@dataclass(frozen=True, slots=True)
class _NumericConstraint:
    integer: bool
    minimum: int | float
    maximum: int | float | None = None
    minimum_inclusive: bool = True


_POSITIVE_INTEGER = _NumericConstraint(integer=True, minimum=1)
_NON_NEGATIVE_INTEGER = _NumericConstraint(integer=True, minimum=0)
_POSITIVE_NUMBER = _NumericConstraint(integer=False, minimum=0, minimum_inclusive=False)
_NON_NEGATIVE_NUMBER = _NumericConstraint(integer=False, minimum=0)

NUMERIC_CONSTRAINTS: dict[str, _NumericConstraint] = {
    "approval_ttl_seconds": _POSITIVE_INTEGER,
    "cascade_min_output_chars": _NON_NEGATIVE_INTEGER,
    "max_calls": _POSITIVE_INTEGER,
    "max_cell_chars": _POSITIVE_INTEGER,
    "max_cost_usd": _NON_NEGATIVE_NUMBER,
    "max_depth": _NON_NEGATIVE_INTEGER,
    "max_document_tables": _POSITIVE_INTEGER,
    "max_file_bytes": _POSITIVE_INTEGER,
    "max_file_context_chars": _POSITIVE_INTEGER,
    "max_latency_ms": _POSITIVE_INTEGER,
    "max_native_file_bytes": _POSITIVE_INTEGER,
    "max_output_tokens": _POSITIVE_INTEGER,
    "max_panel_size": _POSITIVE_INTEGER,
    "max_parallel_models": _POSITIVE_INTEGER,
    "max_pdf_pages": _POSITIVE_INTEGER,
    "max_provider_retries": _NON_NEGATIVE_INTEGER,
    "max_table_columns": _POSITIVE_INTEGER,
    "max_table_rows": _POSITIVE_INTEGER,
    "max_table_sheets": _POSITIVE_INTEGER,
    "max_tokens": _POSITIVE_INTEGER,
    "max_tool_calls_per_round": _POSITIVE_INTEGER,
    "max_tool_result_chars": _POSITIVE_INTEGER,
    "max_tool_rounds": _POSITIVE_INTEGER,
    "min_panel_size": _POSITIVE_INTEGER,
    "ocr_timeout_seconds": _POSITIVE_NUMBER,
    "orchestrator_candidate_limit": _POSITIVE_INTEGER,
    "retry_backoff_seconds": _NON_NEGATIVE_NUMBER,
    "retry_jitter_seconds": _NON_NEGATIVE_NUMBER,
    "selection_trace_limit": _POSITIVE_INTEGER,
    "temperature": _NumericConstraint(integer=False, minimum=0, maximum=2),
    "thinking_budget": _NON_NEGATIVE_INTEGER,
    "timeout": _POSITIVE_NUMBER,
    "timeout_seconds": _POSITIVE_NUMBER,
    "top_p": _NumericConstraint(integer=False, minimum=0, maximum=1),
}


def validate_request_constraints(
    constraints: dict[str, Any],
    *,
    has_tools: bool,
) -> list[str]:
    """Validate core semantics and return non-fatal compatibility warnings."""

    for key in sorted(BOOLEAN_CONSTRAINTS.intersection(constraints)):
        if not isinstance(constraints[key], bool):
            raise CrupierRouteValidationError(f"Constraint {key!r} must be a boolean.")

    for key in sorted(NUMERIC_CONSTRAINTS.keys() & constraints.keys()):
        constraints[key] = _validate_numeric_constraint(key, constraints[key])

    if constraints.get("requires_tools") and not has_tools:
        raise CrupierRouteValidationError(
            "requires_tools=True requires at least one tool definition in tools=[...]."
        )

    unknown = sorted(set(constraints).difference(SUPPORTED_CONSTRAINTS))
    if not unknown:
        return []

    names = ", ".join(repr(key) for key in unknown)
    message = (
        f"Unknown request constraint(s) {names}; Crupier core does not apply them. "
        "Use metadata for application-only values or strict_constraints=True to reject them."
    )
    if constraints.get("strict_constraints"):
        raise CrupierRouteValidationError(message)
    return [message]


def _validate_numeric_constraint(key: str, value: Any) -> int | float:
    spec = NUMERIC_CONSTRAINTS[key]
    if isinstance(value, bool) or not isinstance(value, int if spec.integer else int | float):
        expected = "an integer" if spec.integer else "a number"
        raise CrupierRouteValidationError(f"Constraint {key!r} must be {expected}.")

    normalized = value if spec.integer else float(value)
    if not spec.integer and not math.isfinite(normalized):
        raise CrupierRouteValidationError(f"Constraint {key!r} must be finite.")
    below_minimum = (
        normalized < spec.minimum
        if spec.minimum_inclusive
        else normalized <= spec.minimum
    )
    if below_minimum or spec.maximum is not None and normalized > spec.maximum:
        interval_start = "[" if spec.minimum_inclusive else "("
        interval_end = "]" if spec.maximum is not None else ")"
        maximum = str(spec.maximum) if spec.maximum is not None else "infinity"
        raise CrupierRouteValidationError(
            f"Constraint {key!r} must be within {interval_start}{spec.minimum}, {maximum}{interval_end}."
        )
    return normalized


def request_allows_parallel(config: CrupierConfig, request: RequestEnvelope) -> bool:
    """Resolve request preference without bypassing the project-level limit."""

    return bool(config.routing.allow_parallel) and bool(
        request.constraints.get("allow_parallel", True)
    )


def request_requires_human_approval(request: RequestEnvelope) -> bool:
    return bool(request.constraints.get("requires_human_approval"))


def human_approval_granted(request: RequestEnvelope) -> bool:
    approval = request.metadata.get("_crupier_approval")
    return isinstance(approval, dict) and bool(approval.get("approval_id"))
