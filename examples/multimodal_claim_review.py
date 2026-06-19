"""Plan a multimodal insurance-claim route without provider calls.

Run:

    python examples/multimodal_claim_review.py
"""

from __future__ import annotations

from _example_support import offline_client, print_route


def main() -> None:
    crupier = offline_client(
        project="claims-review",
        profile="structured",
        allow=[
            "openai:gpt-5.4-mini",
            "google:gemini-3.1-flash-lite",
            "anthropic:claude-opus-4-8",
        ],
    )

    result = crupier.deal(
        task=(
            "Review an insurance claim package. Extract claimant, incident date, "
            "estimated repair total, missing evidence, and whether a human adjuster "
            "must inspect the file before approval."
        ),
        input={"claim_id": "CLM-2026-10442", "line_of_business": "auto"},
        files=[
            {"kind": "image", "name": "damage_photo_front.png", "mime_type": "image/png", "size_bytes": 640_000},
            {"kind": "pdf", "name": "repair_estimate.pdf", "mime_type": "application/pdf", "page_count": 4},
            {"kind": "spreadsheet", "name": "parts_quote.csv", "mime_type": "text/csv", "size_bytes": 18_400},
        ],
        mode="structured",
        constraints={
            "risk_level": "medium",
            "strict_response_schema": True,
            "max_file_context_chars": 40_000,
        },
        response_schema={
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "incident_date": {"type": "string"},
                "repair_total": {"type": "number"},
                "missing_evidence": {"type": "array", "items": {"type": "string"}},
                "human_adjuster_required": {"type": "boolean"},
            },
            "required": ["claim_id", "missing_evidence", "human_adjuster_required"],
        },
        dry_run=True,
    )

    file_plan = result.route.input_plan.get("files", {})
    print_route(
        "multimodal_claim_review",
        result,
        extra={
            "required_modalities": file_plan.get("required_model_modalities"),
            "extraction_required": file_plan.get("extraction_required"),
        },
    )


if __name__ == "__main__":
    main()
