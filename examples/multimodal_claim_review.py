"""Plan a multimodal insurance-claim route without provider calls.

Run:

    python examples/multimodal_claim_review.py
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _example_support import offline_client, print_route

from crupier.multimodal import (
    normalize_file,
    plan_file_representations,
    prepare_extracted_file_context,
)


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

    with TemporaryDirectory(prefix="crupier-claim-example-") as temporary:
        quote = Path(temporary) / "parts_quote.csv"
        quote.write_text(
            "part,quantity,unit_price\nbumper,1,420.00\nsensor,2,89.50\n",
            encoding="utf-8",
        )
        quote_asset = normalize_file(quote)
        quote_plan = plan_file_representations([quote_asset])
        extracted = prepare_extracted_file_context([quote_asset], quote_plan)

        result = crupier.deal(
            task=(
                "Review an insurance claim package. Extract claimant, incident date, "
                "estimated repair total, missing evidence, and whether a human adjuster "
                "must inspect the file before approval."
            ),
            input={"claim_id": "CLM-2026-10442", "line_of_business": "auto"},
            files=[
                {
                    "kind": "image",
                    "name": "damage_photo_front.png",
                    "mime_type": "image/png",
                    "size_bytes": 640_000,
                },
                {
                    "kind": "pdf",
                    "name": "repair_estimate.pdf",
                    "mime_type": "application/pdf",
                    "page_count": 4,
                },
                quote,
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
            trace="summary",
        )

        route = result.route
        if route is None:
            raise RuntimeError("multimodal_claim_review did not produce a route plan")
        file_plan = route.input_plan.get("files", {})
        tables = extracted.get("tables", [])
        if not tables or len(tables[0].get("rows", [])) != 2:
            raise RuntimeError("CSV extraction did not produce the expected bounded rows")
        print_route(
            "multimodal_claim_review",
            result,
            extra={
                "required_modalities": file_plan.get("required_model_modalities"),
                "extraction_required": file_plan.get("extraction_required"),
                "csv_rows_extracted": len(tables[0]["rows"]),
                "execution_boundary": (
                    "CSV, TSV, XLSX and DOCX extract locally; PDF tables, audio and video "
                    "still fail or warn explicitly"
                ),
            },
        )


if __name__ == "__main__":
    main()
