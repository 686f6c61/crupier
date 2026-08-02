"""Run a Crupier route decision without provider SDKs or API keys.

This example is safe to run right after installing the base package:

    python examples/sdk_dry_run.py
"""

from _example_support import offline_client, print_route


def main() -> None:
    crupier = offline_client(
        project="sdk-dry-run",
        profile="fast",
        allow=[
            "openai:gpt-5.4-mini",
            "anthropic:claude-sonnet-4-6",
            "google:gemini-3.1-flash-lite",
            "ollama:gpt-oss:120b",
        ],
    )

    result = crupier.deal(
        task="Choose a model route for a short support reply.",
        input={"priority": "normal", "message": "Where is my invoice?"},
        mode="fast",
        constraints={"max_cost_usd": 0.01, "max_latency_ms": 6000},
        dry_run=True,
        trace="summary",
    )

    print_route("sdk_dry_run", result, extra={"dry_run": True})


if __name__ == "__main__":
    main()
