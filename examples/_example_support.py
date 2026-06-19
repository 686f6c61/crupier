"""Shared helpers for offline Crupier examples."""

from __future__ import annotations

from typing import Any

from crupier import Crupier, CrupierResult


def offline_client(*, project: str, allow: list[str], profile: str = "agentic") -> Crupier:
    """Build a dry-run friendly client without requiring provider keys."""

    return Crupier.from_config(
        {
            "project": {"name": project, "default_profile": profile},
            "providers": {
                "openai": {"enabled": True, "env_key": "OPENAI_API_KEY"},
                "anthropic": {"enabled": True, "env_key": "ANTHROPIC_API_KEY"},
                "google": {"enabled": True, "env_key": "GOOGLE_API_KEY"},
                "ollama": {"enabled": True, "host": "https://ollama.com/api", "env_key": "OLLAMA_API_KEY"},
            },
            "models": {"allow": allow},
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
        }
    )


def print_route(title: str, result: CrupierResult, *, extra: dict[str, Any] | None = None) -> None:
    route = result.route
    print(f"== {title} ==")
    print(f"strategy={route.strategy}")
    print(f"models={','.join(route.models)}")
    print(f"risk={route.risk_level}")
    print(f"reason={route.reason}")
    if route.input_plan:
        print(f"input_plan={route.input_plan}")
    for key, value in (extra or {}).items():
        print(f"{key}={value}")
