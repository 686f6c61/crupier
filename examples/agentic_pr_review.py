"""Plan a high-risk agentic code-review route without executing tools.

Run:

    python examples/agentic_pr_review.py
"""

from __future__ import annotations

from _example_support import offline_client, print_route


def main() -> None:
    crupier = offline_client(
        project="repo-review-agent",
        profile="agentic",
        allow=[
            "openai:gpt-5.5",
            "openai:gpt-5.4-mini",
            "anthropic:claude-opus-4-8",
            "google:gemini-3.5-flash",
        ],
    )

    result = crupier.deal(
        task=(
            "Review a pull request that changes payment retry behavior. Identify "
            "production risks, missing tests, rollback concerns, and whether a "
            "second-model critique is justified before merge."
        ),
        input={
            "repo": "billing-worker",
            "changed_files": [
                "src/retries.py",
                "src/provider_webhooks.py",
                "tests/test_retries.py",
            ],
            "risk_notes": [
                "touches idempotency keys",
                "changes provider timeout behavior",
                "affects recurring invoices",
            ],
        },
        mode="agentic",
        constraints={
            "risk_level": "high",
            "requires_tools": True,
            "requires_human_approval": True,
            "max_cost_usd": 0.30,
        },
        tools=[
            {
                "name": "read_changed_file",
                "description": "Read a changed source file from the repository checkout.",
                "requires_approval": True,
            },
            {
                "name": "run_targeted_tests",
                "description": "Run a focused test command selected by the reviewer.",
                "requires_approval": True,
            },
        ],
        dry_run=True,
    )

    print_route("agentic_pr_review", result, extra={"changed_files": 3})


if __name__ == "__main__":
    main()
