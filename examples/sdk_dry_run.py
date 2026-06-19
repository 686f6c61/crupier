"""Run a Crupier route decision without provider SDKs or API keys.

This example is safe to run right after installing the base package:

    python examples/sdk_dry_run.py
"""

from crupier import Crupier


def main() -> None:
    crupier = Crupier.from_config(
        {
            "project": {"name": "example", "default_profile": "agentic"},
            "models": {
                "allow": [
                    "openai:gpt-5.4-mini",
                    "anthropic:claude-opus-4-8",
                ]
            },
            "routing": {
                "default_strategy": "orchestrated",
                "allow_fusion": True,
                "allow_parallel": True,
                "allow_latest_aliases": False,
                "allow_preview_models": False,
            },
            "logging": {
                "store_prompts": False,
                "store_responses": False,
                "redact_secrets": True,
            },
        }
    )

    result = crupier.deal(
        task="Choose a model route for a short support reply.",
        input={"priority": "normal", "message": "Where is my invoice?"},
        mode="agentic",
        dry_run=True,
        trace="summary",
    )

    print(f"strategy={result.route.strategy}")
    print(f"models={','.join(result.route.models)}")
    print(f"summary={result.route.model_summary}")


if __name__ == "__main__":
    main()
