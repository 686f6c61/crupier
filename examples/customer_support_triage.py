"""Route a realistic customer-support task without calling providers.

Run:

    python examples/customer_support_triage.py
"""

from __future__ import annotations

from _example_support import offline_client, print_route


def main() -> None:
    crupier = offline_client(
        project="support-desk",
        profile="fast",
        allow=[
            "openai:gpt-5.4-mini",
            "anthropic:claude-opus-4-8",
            "google:gemini-3.5-flash",
        ],
    )

    result = crupier.deal(
        task=(
            "Draft a concise support reply, classify escalation risk, and keep the "
            "answer under 90 words."
        ),
        input={
            "ticket_id": "SUP-18429",
            "plan": "Team",
            "message": "We were charged twice after upgrading seats yesterday.",
            "account_age_days": 420,
            "priority": "normal",
        },
        mode="fast",
        constraints={
            "max_latency_ms": 2500,
            "max_cost_usd": 0.01,
            "response_schema_name": "support_reply",
        },
        response_schema={
            "type": "object",
            "properties": {
                "reply": {"type": "string"},
                "escalation": {"type": "string", "enum": ["none", "billing", "technical", "legal"]},
                "confidence": {"type": "number"},
            },
            "required": ["reply", "escalation", "confidence"],
        },
        dry_run=True,
    )

    print_route("support_triage", result, extra={"ticket_id": "SUP-18429"})


if __name__ == "__main__":
    main()
