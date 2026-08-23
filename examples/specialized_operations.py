"""Plan chat-adjacent operations against operation-capable model cards.

The inputs are summarized for planning and no provider is called:

    python examples/specialized_operations.py

Unlike the chat examples, every route below reports `planned_provider_calls=0`.
Operation planning does not stage dry-run calls in the trace, so the planned work
is the route itself: one `primary` step bound to one operation-capable model.
`real_provider_calls=0` is the line that proves nothing was sent to a provider.
"""

from __future__ import annotations

from _example_support import offline_client, print_route


def main() -> None:
    crupier = offline_client(
        project="specialized-operations",
        profile="fast",
        allow=[
            "openai:gpt-5.4-mini",
            "openai:text-embedding-3-small",
            "google:gemini-embedding-001",
            "nan:qwen3-embedding",
            "nan:rerank",
            "nan:whisper",
            "nan:kokoro",
            "nan:flux-2-klein",
        ],
    )

    results = [
        (
            "auto_classified_embeddings",
            crupier.run(
                "Create embeddings for semantic product-catalog search.",
                ["annual billing", "payment retry", "seat provisioning"],
                operation="auto",
                dry_run=True,
            ),
        ),
        (
            "rerank_retrieval_results",
            crupier.rerank(
                query="payment retry idempotency",
                documents=[
                    "Brand color usage in the admin console",
                    "Idempotency keys for payment-provider retries",
                    "Exporting payroll records as CSV",
                ],
                top_n=2,
                dry_run=True,
            ),
        ),
        (
            "transcribe_support_call",
            crupier.transcribe(
                file=("support-call.wav", b"RIFF-example-placeholder", "audio/wav"),
                dry_run=True,
            ),
        ),
        (
            "synthesize_status_update",
            crupier.synthesize(
                input="The deployment completed and all health checks passed.",
                voice="af_heart",
                dry_run=True,
            ),
        ),
        (
            "generate_release_diagram",
            crupier.generate_image(
                prompt="A precise technical diagram of a multi-provider model routing system.",
                dry_run=True,
            ),
        ),
    ]

    for name, result in results:
        print_route(name, result, extra={"dry_run": True})

    # La columna «planned» del chat no aplica a operaciones: el planificador de
    # operaciones no registra llamadas en la traza, por eso siempre vale 0.
    print(
        "planned_provider_calls_note=operations plan one primary step per request; "
        "the trace records no staged calls, so planned stays 0 while real proves the dry run"
    )


if __name__ == "__main__":
    main()
