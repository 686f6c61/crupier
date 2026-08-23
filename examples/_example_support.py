"""Shared helpers for offline Crupier examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crupier import Crupier, CrupierResult, OperationResult
from crupier.config import CrupierConfig

EXAMPLE_PROFILES = {
    "agentic": {
        "prefer": ["tool_use", "coding", "long_horizon", "reliability"],
        "strategy": "orchestrated",
    },
    "cheap": {"prefer": ["low_cost"], "strategy": "orchestrated"},
    "fast": {"prefer": ["low_latency"], "strategy": "orchestrated"},
    "private": {
        "prefer": ["local", "zdr", "no_prompt_logging"],
        "strategy": "local_first",
    },
    "quality": {"prefer": ["reasoning", "reliability"], "strategy": "orchestrated"},
    "research": {"prefer": ["consensus", "critique"], "strategy": "orchestrated"},
    "structured": {
        "prefer": ["structured_output", "schema_validity"],
        "strategy": "orchestrated",
    },
}


def offline_client(
    *,
    project: str,
    allow: list[str],
    profile: str = "agentic",
    root: str | Path | None = None,
    experiments: dict[str, Any] | None = None,
    policy_rules: list[dict[str, Any]] | None = None,
) -> Crupier:
    """Build a dry-run friendly client without requiring provider keys.

    ``policy_rules`` mirrors the ``[[policy.rules]]`` tables of a real
    ``crupier.toml``. A malformed rule raises ``CrupierConfigError`` instead of
    degrading to an allow-all policy.
    """

    config = CrupierConfig.from_dict(
        {
            "project": {"name": project, "default_profile": profile},
            "providers": {
                "openai": {"enabled": True, "env_key": "OPENAI_API_KEY"},
                "anthropic": {"enabled": True, "env_key": "ANTHROPIC_API_KEY"},
                "google": {"enabled": True, "env_key": "GOOGLE_API_KEY"},
                "ollama": {"enabled": True, "host": "https://ollama.com/api", "env_key": "OLLAMA_API_KEY"},
                # OpenRouter es BYOK opcional y viene deshabilitado igual que en el
                # crupier.toml por defecto: cualquier modelo suyo en la allowlist se
                # excluye con el filtro openrouter_byok en vez de enrutarse.
                "openrouter": {
                    "enabled": False,
                    "mode": "byok",
                    "host": "https://openrouter.ai/api/v1",
                    "env_key": "OPENROUTER_API_KEY",
                },
                "nan": {"enabled": True, "env_key": "NAN_API_KEY"},
            },
            "models": {"allow": allow},
            "profiles": EXAMPLE_PROFILES,
            "routing": {
                "default_strategy": "orchestrated",
                "allow_fusion": True,
                "allow_parallel": True,
                "allow_latest_aliases": False,
                "allow_preview_models": False,
                "max_provider_retries": 1,
                "retry_backoff_seconds": 0.2,
            },
            "logging": {
                "persist_traces": False,
                "store_prompts": False,
                "store_responses": False,
                "redact_secrets": True,
            },
            "experiments": dict(experiments or {}),
            "policy": {"rules": list(policy_rules or [])},
        }
    )
    if root is not None:
        config.root = Path(root).resolve()
    return Crupier(config)


def print_route(
    title: str,
    result: CrupierResult | OperationResult,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Print routing evidence without exposing prompts or provider responses."""

    route = result.route
    if route is None:
        raise RuntimeError(f"{title} did not produce a route plan")

    print(f"== {title} ==")
    print(f"strategy={route.strategy}")
    print(f"models={','.join(route.models)}")
    print(f"roles={_format_roles(route.steps)}")
    print(f"risk={route.risk_level}")
    print(f"reason={route.reason}")
    print(f"estimated_cost_usd={route.estimated_cost.estimated_usd}")
    print(f"estimated_latency_ms={route.estimated_latency_ms}")
    print(f"filters={','.join(route.policy_filters_applied) or 'none'}")
    print(f"human_approval_required={route.requires_user_confirmation}")

    if route.selection_scores:
        leader = route.selection_scores[0]
        print(f"score_leader={leader.get('model')}:{_number(leader.get('score'))}")
        print(f"score_terms={_format_score_terms(leader.get('terms', []))}")

    file_plan = route.input_plan.get("files") if route.input_plan else None
    if isinstance(file_plan, dict):
        print(f"file_representations={_format_file_representations(file_plan)}")
        print(f"file_warnings={','.join(file_plan.get('warnings', [])) or 'none'}")

    if result.trace is not None:
        print(f"trace_id={result.trace.trace_id}")
        print(f"candidates={len(result.trace.candidate_models)}")
        print(f"excluded={len(result.trace.excluded_models)}")
        print(f"orchestrator_model={result.trace.orchestrator_model or 'deterministic-dry-run'}")
        print(
            "planned_provider_calls="
            f"{_planned_provider_call_count(result.trace.provider_calls)}"
        )
        print(
            "real_provider_calls="
            f"{sum(1 for item in result.trace.provider_calls if item.get('dry_run') is not True)}"
        )
        print(f"trace_errors={_format_errors(result.trace.errors)}")
    else:
        print("trace=disabled")

    if isinstance(result, OperationResult):
        print(f"operation={result.operation}")

    print(f"warnings={_format_values(result.warnings)}")

    for key, value in (extra or {}).items():
        print(f"{key}={value}")


def _format_roles(steps: list[Any]) -> str:
    values: list[str] = []
    for step in steps:
        assigned = step.model or ",".join(step.models) or "unassigned"
        values.append(f"{step.role}:{assigned}")
    return ";".join(values) or "none"


def _format_score_terms(terms: list[dict[str, Any]]) -> str:
    return ";".join(
        f"{term.get('name')}={_number(term.get('value'))}"
        for term in terms
        if term.get("name") is not None
    ) or "none"


def _format_file_representations(file_plan: dict[str, Any]) -> str:
    values = [
        f"{item.get('asset_name', 'file')}:{item.get('representation', 'unknown')}"
        for item in file_plan.get("representations", [])
        if isinstance(item, dict)
    ]
    return ";".join(values) or "none"


def _number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_values(values: list[Any]) -> str:
    return ";".join(_safe_text(value) for value in values) or "none"


def _format_errors(errors: list[dict[str, Any]]) -> str:
    values = []
    for error in errors:
        phase = error.get("phase") or "unknown"
        model = error.get("model") or error.get("provider") or "unknown"
        message = error.get("error") or error.get("message") or error.get("error_type") or "error"
        values.append(f"{phase}:{model}:{_safe_text(message)}")
    return ";".join(values) or "none"


def _planned_provider_call_count(calls: list[dict[str, Any]]) -> int:
    total = 0
    for call in calls:
        if call.get("dry_run") is not True:
            continue
        total += int(bool(call.get("model")))
        models = call.get("models")
        if isinstance(models, list):
            total += len(models)
    return total


def _safe_text(value: Any) -> str:
    return " ".join(str(value).split())
