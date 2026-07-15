"""Validate autonomous routing and multi-model execution with real providers.

Offline preview:

    python examples/live_routing_validation.py

Real validation from a configured project:

    python examples/live_routing_validation.py --real --project . --write-report
"""

from __future__ import annotations

import argparse
import binascii
import json
import struct
import tempfile
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from _example_support import offline_client, print_route
from crupier import Crupier, CrupierResult


CASE_NAMES = (
    "fast",
    "structured",
    "research",
    "agentic",
    "tools",
    "delegate",
    "image",
    "pdf",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="Execute configured provider calls")
    parser.add_argument("--project", default=".", help="Directory containing crupier.toml")
    parser.add_argument("--case", action="append", choices=CASE_NAMES, help="Run only selected cases")
    parser.add_argument(
        "--write-report",
        nargs="?",
        const=".crupier/evals/live-routing-validation.json",
        help="Write the sanitized JSON report",
    )
    args = parser.parse_args()
    if not args.real:
        _offline_preview()
        return

    configured_client = Crupier.from_project(args.project)
    selected = args.case or list(CASE_NAMES)
    cases = [_run_case(name, Crupier(configured_client.config)) for name in selected]
    report = {
        "schema_version": 1,
        "project": configured_client.config.project.name,
        "real_provider_calls": True,
        "summary": {
            "passed": sum(case["status"] == "pass" for case in cases),
            "failed": sum(case["status"] == "fail" for case in cases),
            "total": len(cases),
        },
        "cases": cases,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.write_report:
        path = Path(args.project) / args.write_report
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"report={path}")
    if report["summary"]["failed"]:
        raise SystemExit(1)


def _offline_preview() -> None:
    client = offline_client(
        project="live-routing-validation",
        profile="research",
        allow=[
            "openai:gpt-5.5",
            "anthropic:claude-opus-4-8",
            "google:gemini-3.5-flash",
        ],
    )
    result = client.deal(
        task="Compare two production agent architectures and recommend one.",
        mode="research",
        dry_run=True,
    )
    print_route("live_routing_validation", result, extra={"validation": "offline-preview"})


def _run_case(name: str, client: Crupier) -> dict[str, Any]:
    runners: dict[str, Callable[[Crupier], tuple[str, CrupierResult, dict[str, bool]]]] = {
        "fast": _fast_case,
        "structured": _structured_case,
        "research": _research_case,
        "agentic": _agentic_case,
        "tools": _tools_case,
        "delegate": _delegate_case,
        "image": _image_case,
        "pdf": _pdf_case,
    }
    try:
        task, result, checks = runners[name](client)
        return _observation(name, task, result, checks)
    except Exception as exc:  # noqa: BLE001 - validation reports every case
        return {
            "id": name,
            "status": "fail",
            "error_type": exc.__class__.__name__,
            "error": str(exc)[:1000],
        }


def _fast_case(client: Crupier) -> tuple[str, CrupierResult, dict[str, bool]]:
    task = "Summarize the deployment event in exactly one sentence and include the incident id."
    result = client.deal(
        task=task,
        input={
            "incident_id": "INC-42",
            "event": "A canary raised errors from 0.2% to 2.1% and rolled back automatically.",
        },
        mode="fast",
        constraints=_constraints(output_tokens=100, cost=0.20),
        trace="debug",
        dry_run=False,
    )
    checks = _common_checks(result) | {
        "single_route": _strategy(result) == "single",
        "incident_preserved": "inc-42" in result.output_text.lower(),
    }
    return task, result, checks


def _structured_case(client: Crupier) -> tuple[str, CrupierResult, dict[str, bool]]:
    task = (
        "Extract claim fields from the supplied text; do not infer missing facts. Use a primary extraction, "
        "validate it against the response schema, and reserve a separate escalation model only if validation fails."
    )
    expected = {
        "claim_id": "CLM-2048",
        "opened_date": "2026-07-12",
        "repair_total_eur": 1840.5,
        "missing_evidence": ["police report"],
        "human_adjuster_required": True,
    }
    schema = {
        "type": "object",
        "properties": {
            "claim_id": {"type": "string"},
            "opened_date": {"type": "string"},
            "repair_total_eur": {"type": "number"},
            "missing_evidence": {"type": "array", "items": {"type": "string"}},
            "human_adjuster_required": {"type": "boolean"},
        },
        "required": list(expected),
        "additionalProperties": False,
    }
    result = client.deal(
        task=task,
        input=(
            "Claim CLM-2048 opened on 2026-07-12. Repair estimate: EUR 1840.50. "
            "A police report is missing, so a human adjuster must review it."
        ),
        response_schema=schema,
        mode="structured",
        constraints=_constraints(output_tokens=300, cost=0.25),
        trace="debug",
        dry_run=False,
    )
    checks = _common_checks(result) | {
        "cascade_route": _strategy(result) == "cascade",
        "schema_exact": result.output_json == expected,
    }
    return task, result, checks


def _research_case(client: Crupier) -> tuple[str, CrupierResult, dict[str, bool]]:
    task = (
        "Compare a single frontier model with provider fallback against a capability-aware "
        "multi-provider router for 100000 support tickets per month. Evaluate reliability, latency, "
        "cost, observability, failure modes, and migration risk. Obtain three independent provider-diverse "
        "analyses, have a separate judge reconcile consensus and disagreements, then use a final writer to recommend one."
    )
    result = client.deal(
        task=task,
        input={"availability_target": "99.9%", "team_size": 6, "regulated_data": False},
        mode="research",
        constraints=_constraints(output_tokens=900, cost=0.50)
        | {"min_panel_size": 3, "max_panel_size": 3},
        trace="debug",
        dry_run=False,
    )
    roles = _call_roles(result)
    panel_models = _role_models(result, "panel")
    checks = _common_checks(result) | {
        "fusion_route": _strategy(result) == "fusion",
        "panel_provider_diversity": len({_provider(model) for model in panel_models}) >= 2,
        "panel_executed": roles.count("panel") >= 2,
        "fusion_quorum": bool(
            result.trace
            and result.trace.final_quality_signals.get("fusion_panel_quorum") is True
            and result.trace.final_quality_signals.get("fusion_panel_successful", 0) >= 2
        ),
        "judge_executed": "judge" in roles,
        "writer_executed": "final_writer" in roles,
    }
    return task, result, checks


def _agentic_case(client: Crupier) -> tuple[str, CrupierResult, dict[str, bool]]:
    task = (
        "Review a production payment-retry change. Produce a merge decision, identify rollback "
        "risks, challenge the first draft with an independent critic, and repair the recommendation."
    )
    result = client.deal(
        task=task,
        input={
            "change": "Retry HTTP 429 and 5xx up to five times with exponential backoff.",
            "current_behavior": "Two retries; idempotency key reused per invoice attempt.",
            "known_gaps": ["No Retry-After test", "No concurrent worker test"],
        },
        mode="agentic",
        constraints=_constraints(output_tokens=900, cost=0.50) | {"risk_level": "high", "max_calls": 10},
        trace="debug",
        dry_run=False,
    )
    roles = _call_roles(result)
    checks = _common_checks(result) | {
        "critique_repair_route": _strategy(result) == "critique_repair",
        "generator_executed": "generator" in roles,
        "critic_executed": "critic" in roles,
        "repair_executed": "repair" in roles,
        "no_internal_review_material": not any(
            marker in result.output_text.lower()
            for marker in (
                "first-draft recommendation",
                "first draft recommendation",
                "preserved for audit",
                "independent critic challenge",
                "critic verification",
                "internal verification",
            )
        ),
        "final_envelope_removed": "<final_answer>" not in result.output_text.lower()
        and '"final":' not in result.output_text.lower(),
    }
    return task, result, checks


def _tools_case(client: Crupier) -> tuple[str, CrupierResult, dict[str, bool]]:
    def lookup_billing_case(ticket_id: str) -> dict[str, object]:
        """Return authoritative billing-case state for a support reply."""

        return {
            "ticket_id": ticket_id,
            "duplicate_charge_confirmed": True,
            "refund_status": "not_started",
            "next_action": "billing specialist review",
            "review_eta_business_days": 2,
        }

    task = (
        "Use lookup_billing_case for ticket SUP-LIVE-TOOL-1, then draft a concise reply. "
        "Do not claim a refund started or completed unless the tool says so. State the next action and ETA. "
        "Have an independent critic verify the draft against the tool result, then repair any unsupported claim."
    )
    result = client.deal(
        task=task,
        input={"ticket_id": "SUP-LIVE-TOOL-1", "customer_message": "I was charged twice."},
        tools=[lookup_billing_case],
        mode="agentic",
        constraints=_constraints(output_tokens=300, cost=0.40)
        | {"risk_level": "high", "max_calls": 10, "max_tool_rounds": 2},
        trace="debug",
        dry_run=False,
    )
    output = result.output_text.lower()
    tool_calls = result.provider_metadata.get("tool_calls", [])
    roles = _call_roles(result)
    checks = _common_checks(result) | {
        "tool_completed": bool(tool_calls) and tool_calls[0].get("status") == "completed",
        "tool_critic_executed": "tool_critic" in roles,
        "tool_repair_executed": "tool_repair" in roles,
        "refund_not_started": "refund" in output
        and any(
            marker in output
            for marker in (
                "not started",
                "not yet started",
                "not initiated",
                "not yet initiated",
                "not been initiated",
                "not yet been initiated",
                "has not been initiated",
                "hasn't been initiated",
                "has not been started",
                "hasn't been started",
                "not been started",
            )
        ),
        "review_and_eta_preserved": "review" in output and "2 business days" in output,
        "no_false_missing_tool_claim": "don't have" not in output and "do not have" not in output,
        "no_internal_review_material": not any(
            marker in output
            for marker in (
                "critic verification",
                "verification note",
                "internal note",
                "tool field",
                "tool ledger",
                '"tool_calls"',
            )
        ),
        "final_envelope_removed": "<final_answer>" not in output and '"final":' not in output,
    }
    return task, result, checks


def _delegate_case(client: Crupier) -> tuple[str, CrupierResult, dict[str, bool]]:
    task = (
        "Use delegate exactly once. Hand off this bounded subtask: identify three failure modes "
        "in a capability-aware router and one mitigation for each. Execute the subtask as single-model analysis."
    )
    result = client.deal(
        task=task,
        input={"system": "AI routing layer", "constraints": ["BYOK", "shared budget"]},
        mode="agentic",
        strategy="delegate",
        constraints=_constraints(output_tokens=350, cost=0.40) | {"max_depth": 2, "max_calls": 8},
        trace="debug",
        dry_run=False,
    )
    calls = _trace_calls(result)
    delegate_call = next((call for call in calls if call.get("role") == "delegate"), {})
    checks = _common_checks(result) | {
        "delegate_route": _strategy(result) == "delegate",
        "nested_route_recorded": bool(delegate_call.get("nested_strategy")),
        "depth_reduced": delegate_call.get("max_depth_remaining") == 1,
        "nested_provider_executed": any(call.get("role") == "primary" for call in calls),
    }
    return task, result, checks


def _image_case(client: Crupier) -> tuple[str, CrupierResult, dict[str, bool]]:
    task = "Inspect the attached solid-color image and reply with exactly one lowercase color word."
    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        handle.write(_solid_png((255, 0, 0)))
        handle.flush()
        result = client.deal(
            task=task,
            files=[handle.name],
            mode="fast",
            constraints=_constraints(output_tokens=20, cost=0.20) | {"file_strategy": "auto"},
            trace="debug",
            dry_run=False,
        )
    native_images = sum(
        int((call.get("metadata") or {}).get("multimodal_images", 0) or 0)
        for call in _trace_calls(result)
    )
    checks = _common_checks(result) | {
        "native_image_sent": native_images == 1,
        "color_exact": result.output_text.strip().lower().strip(". ") == "red",
    }
    return task, result, checks


def _pdf_case(client: Crupier) -> tuple[str, CrupierResult, dict[str, bool]]:
    task = "Read the attached PDF and reply with only the audit passphrase."
    with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
        handle.write(_text_pdf("The audit passphrase is zircon."))
        handle.flush()
        result = client.deal(
            task=task,
            files=[handle.name],
            mode="fast",
            constraints=_constraints(output_tokens=100, cost=0.20) | {"file_strategy": "native"},
            trace="debug",
            dry_run=False,
        )
    native_files = sum(
        int((call.get("metadata") or {}).get("native_files", 0) or 0)
        for call in _trace_calls(result)
    )
    checks = _common_checks(result) | {
        "native_pdf_sent": native_files == 1,
        "passphrase_exact": result.output_text.strip().lower().strip(". ") == "zircon",
    }
    return task, result, checks


def _observation(
    name: str,
    task: str,
    result: CrupierResult,
    checks: dict[str, bool],
) -> dict[str, Any]:
    route = result.route
    trace = result.trace
    calls = _trace_calls(result)
    return {
        "id": name,
        "status": "pass" if all(checks.values()) else "fail",
        "task": task,
        "checks": checks,
        "route": {
            "strategy": route.strategy if route else None,
            "steps": [step.to_dict() for step in route.steps] if route else [],
            "reason": route.reason if route else None,
            "estimated_cost": route.estimated_cost.to_dict() if route else None,
            "estimated_latency_ms": route.estimated_latency_ms if route else None,
            "input_plan": route.input_plan if route else {},
        },
        "trace": {
            "orchestrator_model": trace.orchestrator_model if trace else None,
            "calls": [_sanitize_call(call) for call in calls],
            "errors": trace.errors if trace else [],
            "fallbacks": trace.fallbacks if trace else [],
            "quality": trace.final_quality_signals if trace else {},
        },
        "output_preview": result.output_text[:1000],
        "output_json": result.output_json,
        "cost": result.cost.to_dict(),
        "latency_ms": result.latency_ms,
        "warnings": result.warnings,
    }


def _common_checks(result: CrupierResult) -> dict[str, bool]:
    trace = result.trace
    quality = trace.final_quality_signals if trace else {}
    errors = trace.errors if trace else []
    errors_recovered = _trace_errors_recovered(result)
    return {
        "has_route": result.route is not None,
        "real_provider_calls": quality.get("real_provider_calls") is True,
        "model_plan_validated": quality.get("orchestrator_outcome") == "validated_model_plan",
        "no_unrecovered_trace_errors": bool(trace is not None and (not errors or errors_recovered)),
        "nonempty_output": bool(result.output_text.strip() or result.output_json is not None),
    }


def _constraints(*, output_tokens: int, cost: float) -> dict[str, Any]:
    return {
        "max_output_tokens": output_tokens,
        "max_cost_usd": cost,
        "max_latency_ms": 120000,
    }


def _trace_calls(result: CrupierResult) -> list[dict[str, Any]]:
    return list(result.trace.provider_calls) if result.trace else []


def _call_roles(result: CrupierResult) -> list[str]:
    return [str(call.get("role")) for call in _successful_calls(result)]


def _strategy(result: CrupierResult) -> str | None:
    return result.route.strategy if result.route else None


def _role_models(result: CrupierResult, role: str) -> list[str]:
    return [
        str(call["model"])
        for call in _successful_calls(result)
        if call.get("role") == role and call.get("model")
    ]


def _successful_calls(result: CrupierResult) -> list[dict[str, Any]]:
    return [
        call
        for call in _trace_calls(result)
        if call.get("status", "success") != "failed"
        and not bool((call.get("metadata") or {}).get("empty_response"))
    ]


def _trace_errors_recovered(result: CrupierResult) -> bool:
    trace = result.trace
    if trace is None:
        return False
    quality = trace.final_quality_signals
    successful_calls = _successful_calls(result)
    fusion_quorum = quality.get("fusion_panel_quorum") is True
    for error in trace.errors:
        phase = error.get("phase")
        if phase in {"orchestrator_call", "orchestrator_validation"}:
            if quality.get("orchestrator_outcome") != "validated_model_plan":
                return False
            continue
        role_fallback = next(
            (
                fallback
                for fallback in trace.fallbacks
                if fallback.get("phase") == "role_fallback"
                and fallback.get("model") == error.get("model")
                and fallback.get("role") == error.get("role")
                and fallback.get("next_model")
            ),
            None,
        )
        if role_fallback is not None and any(
            call.get("role") == error.get("role")
            and call.get("model") == role_fallback.get("next_model")
            for call in successful_calls
        ):
            continue
        if fusion_quorum and (
            phase == "panel" or (phase == "provider_call" and error.get("role") == "panel")
        ):
            continue
        if phase != "provider_call" or not error.get("retryable"):
            return False
        if not any(
            call.get("role") == error.get("role")
            and call.get("model") == error.get("model")
            and int(call.get("attempt", 0) or 0) > int(error.get("attempt", 0) or 0)
            for call in successful_calls
        ):
            return False
    return True


def _provider(model: str) -> str:
    return model.split(":", 1)[0]


def _sanitize_call(call: dict[str, Any]) -> dict[str, Any]:
    metadata = call.get("metadata") if isinstance(call.get("metadata"), dict) else {}
    return {
        key: value
        for key, value in {
            "role": call.get("role"),
            "provider": call.get("provider"),
            "model": call.get("model"),
            "latency_ms": call.get("latency_ms"),
            "attempt": call.get("attempt"),
            "status": call.get("status"),
            "plan_status": call.get("plan_status"),
            "repair_attempt": call.get("repair_attempt"),
            "strategy": call.get("strategy"),
            "validation_error": call.get("validation_error"),
            "nested_strategy": call.get("nested_strategy"),
            "nested_models": call.get("nested_models"),
            "max_depth_remaining": call.get("max_depth_remaining"),
            "multimodal_images": metadata.get("multimodal_images"),
            "native_files": metadata.get("native_files"),
            "empty_response": metadata.get("empty_response"),
        }.items()
        if value is not None
    }


def _solid_png(rgb: tuple[int, int, int], *, width: int = 64, height: int = 64) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = binascii.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _text_pdf(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 18 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, item in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode("ascii"))
        data.extend(item)
        data.extend(b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    data.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(data)


if __name__ == "__main__":
    main()
