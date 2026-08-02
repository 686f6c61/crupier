"""Compare route shapes and model scores under different project constraints.

This example performs no provider calls:

    python examples/routing_tradeoffs.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from _example_support import offline_client, print_route


@dataclass(frozen=True)
class Scenario:
    name: str
    task: str
    mode: str
    constraints: dict[str, Any] = field(default_factory=dict)
    strategy: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)


def main() -> None:
    crupier = offline_client(
        project="routing-tradeoffs",
        profile="agentic",
        allow=[
            "openai:gpt-5.5",
            "openai:gpt-5.4-mini",
            "anthropic:claude-sonnet-4-6",
            "anthropic:claude-opus-4-8",
            "google:gemini-3.1-flash-lite",
            "ollama:gpt-oss:120b",
        ],
    )

    scenarios = [
        Scenario(
            name="latency_first",
            task="Answer a short account question in one paragraph.",
            mode="fast",
            constraints={"max_cost_usd": 0.01, "max_latency_ms": 6000},
        ),
        Scenario(
            name="schema_first",
            task="Extract contract parties, dates, renewal terms, and notice periods as strict JSON.",
            mode="structured",
            constraints={"strict_response_schema": True, "max_cost_usd": 0.08},
        ),
        Scenario(
            name="evidence_first",
            task=(
                "Compare three agent architectures, reconcile conflicting evidence, "
                "and identify blind spots before recommending one."
            ),
            mode="research",
            constraints={"max_cost_usd": 0.50},
        ),
        Scenario(
            name="critique_before_action",
            task=(
                "Plan a payments migration with repository tools, rollback controls, "
                "and an independent critique before execution."
            ),
            mode="agentic",
            constraints={"risk_level": "high", "requires_tools": True, "max_cost_usd": 0.40},
            tools=[
                {"name": "read_repository", "description": "Read approved files from the checkout."},
                {"name": "run_tests", "description": "Run an approved targeted test command."},
            ],
        ),
        Scenario(
            name="delegated_incident_review",
            task=(
                "Investigate an intermittent payment outage, delegate evidence collection, "
                "synthesize root causes, critique the mitigation, and produce a rollback plan."
            ),
            mode="agentic",
            strategy="delegate",
            constraints={
                "risk_level": "high",
                "requires_tools": True,
                "max_cost_usd": 0.50,
                "max_calls": 8,
                "max_depth": 3,
            },
            tools=[
                {"name": "query_metrics", "description": "Read approved service metrics."},
                {"name": "read_deploy_log", "description": "Read the deployment log."},
            ],
        ),
    ]

    for scenario in scenarios:
        result = crupier.deal(
            task=scenario.task,
            mode=scenario.mode,
            strategy=scenario.strategy,
            constraints=scenario.constraints,
            tools=scenario.tools,
            dry_run=True,
            trace="summary",
        )
        print_route(
            scenario.name,
            result,
            extra={
                "mode": scenario.mode,
                "budget_usd": scenario.constraints.get("max_cost_usd"),
                "dry_run": True,
            },
        )


if __name__ == "__main__":
    main()
