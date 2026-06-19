"""Use Crupier as the AI boundary for an existing app or agent stack.

This example does not call providers. It shows the shape a production app can
use before turning on real execution:

    python examples/drop_in_agent_boundary.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from _example_support import offline_client, print_route
from crupier import Crupier, CrupierResult


@dataclass(frozen=True)
class WorkItem:
    """A task the existing application already knows how to describe."""

    name: str
    mode: str
    task: str
    payload: dict[str, Any]
    constraints: dict[str, Any]


class ExistingAIBoundary:
    """Thin wrapper around the app's single AI call site."""

    def __init__(self, crupier: Crupier) -> None:
        self._crupier = crupier

    def plan(self, item: WorkItem) -> CrupierResult:
        return self._crupier.deal(
            task=item.task,
            input=item.payload,
            mode=item.mode,
            constraints=item.constraints,
            trace="summary",
            dry_run=True,
        )


def main() -> None:
    crupier = offline_client(
        project="existing-agent-platform",
        profile="agentic",
        allow=[
            "openai:gpt-5.4-mini",
            "anthropic:claude-sonnet-4-6",
            "anthropic:claude-opus-4-8",
            "google:gemini-3.5-flash",
            "ollama:gpt-oss:120b",
        ],
    )
    ai = ExistingAIBoundary(crupier)

    queue = [
        WorkItem(
            name="support_reply",
            mode="fast",
            task="Draft a short reply for a billing support ticket.",
            payload={
                "ticket_id": "SUP-21102",
                "message": "My annual invoice shows the wrong tax ID.",
                "account_segment": "startup",
            },
            constraints={
                "max_latency_ms": 2500,
                "max_cost_usd": 0.01,
            },
        ),
        WorkItem(
            name="contract_extraction",
            mode="structured",
            task="Extract renewal terms and flag clauses that need legal review.",
            payload={
                "document_type": "msa_addendum",
                "jurisdiction": "EU",
                "pages": 9,
            },
            constraints={
                "strict_response_schema": True,
                "max_cost_usd": 0.08,
                "requires_human_approval": True,
            },
        ),
        WorkItem(
            name="release_agent",
            mode="agentic",
            task=(
                "Plan a release readiness review for a backend change touching "
                "payments, retries, telemetry, and rollback behavior."
            ),
            payload={
                "repo": "commerce-api",
                "changed_files": [
                    "src/payments/retry_policy.py",
                    "src/telemetry/events.py",
                    "tests/test_retry_policy.py",
                ],
                "deployment_window": "2026-06-24T22:00:00Z",
            },
            constraints={
                "risk_level": "high",
                "requires_tools": True,
                "requires_human_approval": True,
                "max_cost_usd": 0.35,
            },
        ),
        WorkItem(
            name="cost_sensitive_batch",
            mode="cheap",
            task="Classify low-risk inbound leads into three routing buckets.",
            payload={
                "batch_size": 500,
                "labels": ["sales", "support", "ignore"],
                "latency_sla": "next_hour",
            },
            constraints={
                "max_cost_usd": 0.05,
                "allow_parallel": False,
            },
        ),
    ]

    for item in queue:
        result = ai.plan(item)
        print_route(
            item.name,
            result,
            extra={
                "mode": item.mode,
                "dry_run": True,
                "handoff": "switch dry_run=False at this boundary after provider verification",
            },
        )


if __name__ == "__main__":
    main()
