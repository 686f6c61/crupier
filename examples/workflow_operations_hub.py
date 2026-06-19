"""Plan routes for a realistic multi-workflow AI operations hub.

This is still offline and safe to run without provider keys:

    python examples/workflow_operations_hub.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from _example_support import offline_client, print_route


@dataclass(frozen=True)
class WorkflowRequest:
    name: str
    mode: str
    task: str
    payload: dict[str, Any]
    constraints: dict[str, Any] = field(default_factory=dict)
    files: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    response_schema: dict[str, Any] | None = None


def _risk_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "risk": {"type": "string", "enum": ["low", "medium", "high"]},
            "reason": {"type": "string"},
            "next_action": {"type": "string"},
        },
        "required": ["risk", "reason", "next_action"],
    }


def main() -> None:
    crupier = offline_client(
        project="workflow-operations-hub",
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

    workflows = [
        WorkflowRequest(
            name="billing_dispute",
            mode="fast",
            task="Classify a billing dispute and draft a concise first response.",
            payload={
                "ticket_id": "SUP-33120",
                "segment": "enterprise",
                "message": "The invoice doubled after we removed seats. We need this fixed today.",
                "contract_value_usd": 84_000,
            },
            constraints={"max_latency_ms": 2500, "max_cost_usd": 0.02},
            response_schema=_risk_schema(),
        ),
        WorkflowRequest(
            name="claim_package_review",
            mode="structured",
            task="Extract evidence from a claim package and decide if human review is required.",
            payload={"claim_id": "CLM-88210", "line_of_business": "property"},
            files=[
                {"kind": "image", "name": "roof_damage.jpg", "mime_type": "image/jpeg", "size_bytes": 912_000},
                {"kind": "pdf", "name": "contractor_estimate.pdf", "mime_type": "application/pdf", "page_count": 7},
            ],
            constraints={"strict_response_schema": True, "max_file_context_chars": 50_000},
            response_schema=_risk_schema(),
        ),
        WorkflowRequest(
            name="release_readiness",
            mode="agentic",
            task="Plan a release-readiness review for a payments service change.",
            payload={
                "repo": "payments-api",
                "changed_files": [
                    "src/settlement/retry_policy.py",
                    "src/provider_callbacks.py",
                    "tests/test_settlement_retries.py",
                ],
                "deployment_window": "2026-06-25T21:30:00Z",
            },
            constraints={"risk_level": "high", "requires_tools": True, "requires_human_approval": True},
            tools=[
                {"name": "read_changed_file", "description": "Read a file from the local checkout."},
                {"name": "run_targeted_tests", "description": "Run tests selected by the review plan."},
            ],
        ),
        WorkflowRequest(
            name="private_policy_summary",
            mode="private",
            task="Summarize a sensitive internal policy without leaving the configured private route when possible.",
            payload={"document_class": "internal_policy", "contains_pii": True, "pages": 12},
            constraints={"max_cost_usd": 0.10, "requires_human_approval": True},
        ),
    ]

    for workflow in workflows:
        result = crupier.deal(
            task=workflow.task,
            input=workflow.payload,
            mode=workflow.mode,
            constraints=workflow.constraints,
            files=workflow.files,
            tools=workflow.tools,
            response_schema=workflow.response_schema,
            dry_run=True,
            trace="summary",
            metadata={"tenant_id": "acme", "workflow": workflow.name},
        )
        print_route(
            workflow.name,
            result,
            extra={
                "mode": workflow.mode,
                "files": len(workflow.files),
                "tools": len(workflow.tools),
                "human_review": bool(workflow.constraints.get("requires_human_approval")),
            },
        )


if __name__ == "__main__":
    main()
