"""Compare route variants and turn human review into a project-local signal.

The feedback store lives in a temporary directory and is deleted on exit:

    python examples/eval_feedback_loop.py
"""

from __future__ import annotations

from tempfile import TemporaryDirectory

from _example_support import offline_client, print_route

from crupier import CompareVariant, HumanFeedbackStore


def main() -> None:
    crupier = offline_client(
        project="release-routing-eval",
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
    task = (
        "Assess a payment retry change, identify idempotency risks, and recommend "
        "whether it is safe to deploy."
    )
    payload = {
        "service": "billing-worker",
        "changed_files": ["src/retries.py", "src/provider_webhooks.py"],
    }
    variants = [
        CompareVariant(
            name="latency_first",
            mode="fast",
            strategy="single",
            constraints={"max_cost_usd": 0.02, "max_latency_ms": 5000},
        ),
        CompareVariant(
            name="review_depth",
            mode="agentic",
            strategy="critique_repair",
            constraints={"max_cost_usd": 0.35, "risk_level": "high"},
        ),
    ]

    baseline = crupier.deal(
        task=task,
        input=payload,
        mode="agentic",
        constraints={"risk_level": "high", "max_cost_usd": 0.35},
        dry_run=True,
        trace="summary",
    )
    print_route("eval_feedback_loop", baseline, extra={"variants": len(variants)})

    report = crupier.evals.compare(
        task=task,
        input=payload,
        variants=variants,
        dry_run=True,
    )
    print(f"automated_winner={report.winner}")
    print(f"recommendation={report.recommendation}")
    print(f"report_ok={report.ok}")
    print(f"report_status={'pass' if report.ok else 'fail'}")
    for variant in report.variants:
        print(
            "variant="
            f"{variant.name};strategy={variant.strategy};models={','.join(variant.models)};"
            f"cost={variant.estimated_cost_usd};latency_ms={variant.estimated_latency_ms};"
            f"ok={variant.ok};status={variant.status};"
            f"failed_checks={','.join(variant.failed_checks) or 'none'};"
            f"error={variant.error or 'none'}"
        )
    if not report.ok:
        raise RuntimeError("One or more route variants failed deterministic eval checks")

    by_name = {variant.name: variant for variant in report.variants}
    latency_first = by_name["latency_first"]
    review_depth = by_name["review_depth"]
    with TemporaryDirectory(prefix="crupier-feedback-") as root:
        store = HumanFeedbackStore(root)
        store.record(
            project="release-routing-eval",
            rating=2,
            verdict="needs_work",
            models=latency_first.models,
            mode=latency_first.mode,
            strategy=latency_first.strategy,
            tags=["dry_run_source", "missing_independent_critique"],
            note="Example reviewer requires independent critique for payment retry changes.",
        )
        store.record(
            project="release-routing-eval",
            rating=5,
            verdict="accept",
            models=review_depth.models,
            mode=review_depth.mode,
            strategy=review_depth.strategy,
            tags=["dry_run_source", "independent_critique"],
            note="Example reviewer accepts the deeper route shape for this risk class.",
        )
        summary = store.summary()
        for group in summary["groups"]:
            print(
                "feedback="
                f"{group['model']};mode={group['mode']};rating={group['avg_rating']};"
                f"score_delta={group['score_delta']};status={group['status']}"
            )
        print("feedback_persistence=temporary")


if __name__ == "__main__":
    main()
