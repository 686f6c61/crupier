"""Project adoption audits and programmer-facing code comments.

The audit layer is intentionally product-facing. Unit tests can prove that the
planner returns a shape; this module helps a human decide whether the route is
usable in a real project, with explicit checks, canaries, and code comments.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import struct
import tempfile
import zlib
from difflib import unified_diff
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .adapters.google import google_env_label, google_env_present
from .errors import CrupierError
from .models import ModelRef
from .orchestrator import ModelOrchestrator
from .planner import RoutePlanner


REAL_PROVIDER_CHOICES = ("openai", "anthropic", "google", "ollama", "openrouter")
DEFAULT_PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
ADOPTION_SIGNOFF_VERDICTS = {"approve", "reject", "needs_work"}
CODE_COMMENT_DECISION_REVIEWED = {"accepted", "false_positive", "not_applicable", "reviewed", "resolved"}
CODE_COMMENT_DECISION_PENDING = {"needs_change", "rejected", "unresolved"}
CODE_COMMENT_DECISION_VERDICTS = CODE_COMMENT_DECISION_REVIEWED | CODE_COMMENT_DECISION_PENDING
HUMAN_REVIEW_GATE_IDS = {"human_feedback", "adoption_signoff", "programmer_code_comments"}


@dataclass(slots=True)
class AuditCheck:
    id: str
    status: str
    summary: str
    severity: str = "info"
    evidence: dict[str, Any] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RouteReview:
    id: str
    task: str
    status: str
    strategy: str | None = None
    models: list[str] = field(default_factory=list)
    reason: str = ""
    estimated_cost_usd: float | None = None
    human_questions: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CodeComment:
    file: str
    line: int
    title: str
    body: str
    priority: int = 2
    category: str = "integration"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CodeCommentReviewSummary:
    count: int
    reviewed_count: int
    pending_count: int
    latest_review_at: str | None = None
    pending: list[CodeComment] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.pending_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "count": self.count,
            "reviewed_count": self.reviewed_count,
            "pending_count": self.pending_count,
            "latest_review_at": self.latest_review_at,
            "pending": [comment.to_dict() for comment in self.pending],
        }


@dataclass(slots=True)
class AdoptionOption:
    path: str
    status: str
    score: int
    summary: str
    actions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProjectAdoptionPlan:
    project: str
    generated_at: str
    recommended_path: str
    confidence: str
    options: list[AdoptionOption]
    checklist: list[str]
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    framework_hints: dict[str, Any] = field(default_factory=dict)
    code_comments: list[CodeComment] = field(default_factory=list)
    code_comment_review: CodeCommentReviewSummary | None = None
    written_files: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "project": self.project,
            "generated_at": self.generated_at,
            "recommended_path": self.recommended_path,
            "confidence": self.confidence,
            "options": [option.to_dict() for option in self.options],
            "checklist": self.checklist,
            "blockers": self.blockers,
            "warnings": self.warnings,
            "framework_hints": self.framework_hints,
            "code_comments": [comment.to_dict() for comment in self.code_comments],
            "code_comment_review": self.code_comment_review.to_dict() if self.code_comment_review else None,
            "written_files": self.written_files,
        }


@dataclass(slots=True)
class AdoptionPatchSuggestion:
    adoption_path: str
    title: str
    status: str
    summary: str
    diff: str = ""
    commands: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AdoptionPatchReport:
    project: str
    generated_at: str
    adoption_path: str
    patches: list[AdoptionPatchSuggestion]
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    written_files: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "project": self.project,
            "generated_at": self.generated_at,
            "adoption_path": self.adoption_path,
            "patches": [patch.to_dict() for patch in self.patches],
            "blockers": self.blockers,
            "warnings": self.warnings,
            "written_files": self.written_files,
        }


@dataclass(slots=True)
class DoctorGate:
    id: str
    status: str
    summary: str
    severity: str = "info"
    evidence: dict[str, Any] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProjectDoctorReport:
    project: str
    generated_at: str
    readiness_mode: str
    adoption_plan: ProjectAdoptionPlan
    patch_report: AdoptionPatchReport
    audit_report: "ProjectAuditReport"
    eval_history: Any
    feedback_summary: dict[str, Any]
    gates: list[DoctorGate]
    applied_feedback_summary: dict[str, Any] = field(default_factory=dict)
    adoption_signoff_summary: dict[str, Any] = field(default_factory=dict)
    written_files: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return all(gate.status != "fail" for gate in self.gates)

    @property
    def status(self) -> str:
        return "ready" if self.ready else "blocked"

    @property
    def recommended_path(self) -> str:
        return self.adoption_plan.recommended_path

    @property
    def confidence(self) -> str:
        return self.adoption_plan.confidence

    @property
    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for gate in self.gates:
            counts[gate.status] = counts.get(gate.status, 0) + 1
        return counts

    @property
    def review_contract(self) -> dict[str, Any]:
        return build_adoption_review_contract(self.gates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "status": self.status,
            "project": self.project,
            "generated_at": self.generated_at,
            "readiness_mode": self.readiness_mode,
            "recommended_path": self.recommended_path,
            "confidence": self.confidence,
            "summary": self.summary,
            "review_contract": self.review_contract,
            "gates": [gate.to_dict() for gate in self.gates],
            "adoption_plan": self.adoption_plan.to_dict(),
            "patch_report": self.patch_report.to_dict(),
            "audit_report": self.audit_report.to_dict(),
            "eval_history": self.eval_history.to_dict()
            if hasattr(self.eval_history, "to_dict")
            else self.eval_history,
            "feedback_summary": self.feedback_summary,
            "applied_feedback_summary": self.applied_feedback_summary,
            "adoption_signoff_summary": self.adoption_signoff_summary,
            "written_files": self.written_files,
        }


@dataclass(slots=True)
class AdoptionHandoffReport:
    project: str
    generated_at: str
    status: str
    doctor: ProjectDoctorReport
    artifacts: dict[str, list[str]]
    required_human_actions: list[str]
    suggested_commands: list[str]
    written_files: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "project": self.project,
            "generated_at": self.generated_at,
            "status": self.status,
            "doctor": self.doctor.to_dict(),
            "review_contract": self.doctor.review_contract,
            "human_signoff_checklist": _handoff_signoff_checklist(self),
            "artifacts": self.artifacts,
            "required_human_actions": self.required_human_actions,
            "suggested_commands": self.suggested_commands,
            "written_files": self.written_files,
        }


def build_adoption_review_contract(gates: list[DoctorGate]) -> dict[str, Any]:
    """Separate machine evidence from human approval so green code never implies rollout approval."""

    technical_gates = [gate for gate in gates if gate.id not in HUMAN_REVIEW_GATE_IDS]
    human_gates = [gate for gate in gates if gate.id in HUMAN_REVIEW_GATE_IDS]
    technical_blockers = [gate.id for gate in technical_gates if gate.status == "fail"]
    technical_warnings = [gate.id for gate in technical_gates if gate.status == "warn"]
    human_open = [gate.id for gate in human_gates if gate.status != "pass"]
    human_blockers = [gate.id for gate in human_gates if gate.status == "fail"]
    human_warnings = [gate.id for gate in human_gates if gate.status == "warn"]
    technical_status = _contract_status(technical_gates, pass_label="ready", warn_label="ready_with_warnings")
    human_status = _contract_status(human_gates, pass_label="approved", warn_label="needs_review")
    if technical_blockers or human_blockers:
        overall_status = "blocked"
    elif human_open:
        overall_status = "needs-human-review"
    elif technical_warnings:
        overall_status = "ready_with_warnings"
    else:
        overall_status = "ready"
    if technical_blockers:
        summary = "Technical gates are failing; do not ask for rollout approval yet."
    elif human_open:
        summary = "Technical gates have no failing checks, but human review or signoff is still open."
    elif technical_warnings:
        summary = "Human gates are closed, but technical warnings remain for the owner to accept."
    else:
        summary = "Technical and human gates are closed."
    return {
        "overall_status": overall_status,
        "technical_status": technical_status,
        "human_status": human_status,
        "summary": summary,
        "requires_human_signoff": "adoption_signoff" in human_open,
        "must_not_auto_approve": overall_status != "ready",
        "technical_blockers": technical_blockers,
        "technical_warnings": technical_warnings,
        "human_blockers": human_blockers,
        "human_warnings": human_warnings,
        "human_open_gates": human_open,
        "technical_gate_count": len(technical_gates),
        "human_gate_count": len(human_gates),
    }


def _contract_status(gates: list[DoctorGate], *, pass_label: str, warn_label: str) -> str:
    if any(gate.status == "fail" for gate in gates):
        return "blocked"
    if any(gate.status == "warn" for gate in gates):
        return warn_label
    return pass_label


@dataclass(slots=True)
class ProjectAuditReport:
    project: str
    generated_at: str
    checks: list[AuditCheck]
    route_reviews: list[RouteReview]
    real_canaries: list[dict[str, Any]] = field(default_factory=list)
    code_comments: list[CodeComment] = field(default_factory=list)
    written_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    @property
    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for check in self.checks:
            counts[check.status] = counts.get(check.status, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "project": self.project,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "checks": [check.to_dict() for check in self.checks],
            "route_reviews": [review.to_dict() for review in self.route_reviews],
            "real_canaries": self.real_canaries,
            "code_comments": [comment.to_dict() for comment in self.code_comments],
            "written_files": self.written_files,
        }


class ProjectAuditRunner:
    def __init__(self, client: Any):
        self.client = client

    def run(
        self,
        *,
        dataset: str | Path | None = None,
        providers: list[str] | None = None,
        include_openai_baseline: bool = True,
        orchestrator_mode: str | None = None,
        real: bool = False,
        all_models: bool = False,
        include_code_comments: bool = True,
        code_paths: list[str | Path] | None = None,
        max_code_files: int = 200,
        write_report: bool = False,
    ) -> ProjectAuditReport:
        original_orchestrator_mode = self.client.config.orchestrator.mode
        original_planner = self.client.planner
        if orchestrator_mode:
            self.client.config.orchestrator.mode = orchestrator_mode
            orchestrator = (
                ModelOrchestrator(self.client.config, adapters=self.client.adapters)
                if orchestrator_mode in {"model", "hybrid"}
                else None
            )
            self.client.planner = RoutePlanner(self.client.config, orchestrator=orchestrator)
        try:
            checks: list[AuditCheck] = []
            checks.extend(self._configuration_checks())
            provider_names = self._provider_names(providers, include_openai_baseline=include_openai_baseline)
            checks.extend(self._provider_checks(provider_names, real=real))

            eval_report = self.client.evals.run(dataset=dataset)
            checks.append(_eval_check(eval_report))

            route_reviews = self._route_reviews()
            checks.append(_route_review_check(route_reviews))

            real_canaries: list[dict[str, Any]] = []
            if real:
                real_canaries = self._real_canaries(provider_names, all_models=all_models)
                checks.append(_real_canary_check(real_canaries))
            else:
                checks.append(
                    AuditCheck(
                        id="real_canaries",
                        status="warn",
                        severity="medium",
                        summary="Real provider canaries were skipped.",
                        actions=["Run `crupier audit --real` before production use."],
                    )
                )

            code_comments: list[CodeComment] = []
            if include_code_comments:
                code_comments = scan_code_comments(
                    self.client.config.root,
                    paths=code_paths,
                    max_files=max_code_files,
                )
                checks.append(_code_comment_check(code_comments))

            report = ProjectAuditReport(
                project=self.client.config.project.name,
                generated_at=datetime.now(UTC).isoformat(),
                checks=checks,
                route_reviews=route_reviews,
                real_canaries=real_canaries,
                code_comments=code_comments,
            )
            if write_report:
                report.written_files = [str(path) for path in write_project_audit_report(self.client.config.root, report)]
            return report
        finally:
            self.client.config.orchestrator.mode = original_orchestrator_mode
            self.client.planner = original_planner

    def _configuration_checks(self) -> list[AuditCheck]:
        config = self.client.config
        checks = [
            AuditCheck(
                id="allowlist_present",
                status="pass" if config.models.allow else "fail",
                severity="high",
                summary=(
                    f"{len(config.models.allow)} model(s) are allowlisted."
                    if config.models.allow
                    else "No model allowlist is configured."
                ),
                evidence={"allowlist_count": len(config.models.allow)},
                actions=["Run `crupier models allow provider:model --replace`."] if not config.models.allow else [],
            ),
            AuditCheck(
                id="privacy_defaults",
                status="pass" if not config.logging.store_prompts and not config.logging.store_responses else "fail",
                severity="high",
                summary="Prompt/response storage is disabled by default."
                if not config.logging.store_prompts and not config.logging.store_responses
                else "Prompt or response storage is enabled by default.",
                evidence={
                    "store_prompts": config.logging.store_prompts,
                    "store_responses": config.logging.store_responses,
                    "redact_secrets": config.logging.redact_secrets,
                },
                actions=["Keep prompt/response persistence opt-in for project adoption."]
                if config.logging.store_prompts or config.logging.store_responses
                else [],
            ),
        ]
        risky_models = [
            model
            for model in config.models.allow
            if (ModelRef.parse(model).stability == "latest" and not config.routing.allow_latest_aliases)
            or (ModelRef.parse(model).stability in {"preview", "experimental"} and not config.routing.allow_preview_models)
        ]
        checks.append(
            AuditCheck(
                id="stable_model_refs",
                status="pass" if not risky_models else "fail",
                severity="high",
                summary="Allowlist uses stable explicit model refs."
                if not risky_models
                else "Allowlist contains latest/preview/experimental models without opt-in.",
                evidence={"models": risky_models},
                actions=["Pin stable model IDs or explicitly enable latest/preview policy."] if risky_models else [],
            )
        )
        enabled = [name for name, settings in config.providers.items() if settings.enabled]
        checks.append(
            AuditCheck(
                id="enabled_providers",
                status="pass" if enabled else "fail",
                severity="high",
                summary=f"Enabled providers: {', '.join(enabled)}." if enabled else "No providers are enabled.",
                evidence={"providers": enabled},
                actions=["Enable at least one provider in crupier.toml."] if not enabled else [],
            )
        )
        return checks

    def _provider_names(self, requested: list[str] | None, *, include_openai_baseline: bool) -> list[str]:
        selected = list(requested or [])
        if not selected:
            selected = [
                provider
                for provider in REAL_PROVIDER_CHOICES
                if provider in self.client.config.providers and self.client.config.providers[provider].enabled
            ]
        if include_openai_baseline and "openai" not in selected:
            selected.insert(0, "openai")
        ordered: list[str] = []
        for provider in REAL_PROVIDER_CHOICES:
            if provider in selected and provider not in ordered:
                ordered.append(provider)
        return ordered

    def _provider_checks(self, providers: list[str], *, real: bool) -> list[AuditCheck]:
        checks: list[AuditCheck] = []
        for provider in providers:
            settings = self.client.config.providers.get(provider)
            env = _provider_env_status(settings, provider)
            refs = _model_refs_for_provider(self.client, provider, all_models=False)
            issues: list[str] = []
            if settings is None:
                issues.append(f"{provider} is not configured.")
            elif not settings.enabled:
                issues.append(f"{provider} is disabled.")
            if provider not in self.client.adapters:
                issues.append(f"{provider} adapter is unavailable.")
            if env["required"] and not env["present"]:
                issues.append(f"{env['key']} is missing.")
            if not refs:
                issues.append(f"No allowed {provider} models.")

            checks.append(
                AuditCheck(
                    id=f"provider_{provider}_configured",
                    status="fail" if issues and real else ("warn" if issues else "pass"),
                    severity="high" if real else "medium",
                    summary=f"{provider} configuration has issues." if issues else f"{provider} is configured.",
                    evidence={"issues": issues, "env": env, "allowed_models": refs},
                    actions=[
                        "Enable provider, set env vars, and add explicit allowlisted models before real adoption."
                    ]
                    if issues
                    else [],
                )
            )
            if refs:
                try:
                    readiness = self.client.capabilities.readiness(refs)
                    summary = readiness.summary()
                    status = "pass" if summary.get("ready") == len(refs) else "warn"
                    checks.append(
                        AuditCheck(
                            id=f"provider_{provider}_readiness",
                            status=status,
                            severity="medium",
                            summary=f"{provider} readiness: {summary}.",
                            evidence=readiness.to_dict(),
                            actions=["Run capability probes with `--apply` for models that still rely on inference."]
                            if status != "pass"
                            else [],
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - audit should keep reporting
                    checks.append(
                        AuditCheck(
                            id=f"provider_{provider}_readiness",
                            status="fail" if real else "warn",
                            severity="high" if real else "medium",
                            summary=f"{provider} readiness check failed.",
                            evidence={"error": _redact_secrets(str(exc))},
                        )
                    )
        return checks

    def _route_reviews(self) -> list[RouteReview]:
        reviews: list[RouteReview] = []
        for case in HUMAN_ROUTE_REVIEW_CASES:
            try:
                result = self.client.deal(
                    task=case["task"],
                    input=case.get("input"),
                    mode=case.get("mode"),
                    strategy=case.get("strategy"),
                    constraints=dict(case.get("constraints", {})),
                    response_schema=case.get("response_schema"),
                    dry_run=True,
                    trace="summary",
                )
                plan = result.route
                if plan is None:
                    reviews.append(RouteReview(id=case["id"], task=case["task"], status="fail", error="No route plan."))
                    continue
                failed = []
                if not plan.reason:
                    failed.append("missing_route_reason")
                if not plan.selection_scores:
                    failed.append("missing_selection_scores")
                reviews.append(
                    RouteReview(
                        id=case["id"],
                        task=case["task"],
                        status="pass" if not failed else "warn",
                        strategy=plan.strategy,
                        models=plan.models,
                        reason=plan.reason,
                        estimated_cost_usd=plan.estimated_cost.estimated_usd,
                        human_questions=list(case["human_questions"]),
                        failed_checks=failed,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - route review should capture planner failures
                reviews.append(
                    RouteReview(
                        id=case["id"],
                        task=case["task"],
                        status="fail",
                        error=_redact_secrets(str(exc)),
                        human_questions=list(case["human_questions"]),
                    )
                )
        return reviews

    def _real_canaries(self, providers: list[str], *, all_models: bool) -> list[dict[str, Any]]:
        canaries: list[dict[str, Any]] = []
        model_refs: list[str] = []
        for provider in providers:
            model_refs.extend(_model_refs_for_provider(self.client, provider, all_models=all_models))
        chat_refs = [model for model in model_refs if self.client.registry.get(model).model_kind != "embedding"]
        selected_for_smoke = chat_refs if all_models else _first_per_provider(chat_refs)
        for model_ref in selected_for_smoke:
            canaries.append(self._text_canary(model_ref))
        if chat_refs:
            structured_model = _prefer_provider(chat_refs, "openai") or chat_refs[0]
            canaries.append(self._structured_canary(structured_model))
            tool_model = _prefer_provider(chat_refs, "openai") or chat_refs[0]
            canaries.append(self._tool_canary(tool_model))
            file_model = _prefer_provider(chat_refs, "openai") or chat_refs[0]
            canaries.append(self._text_file_canary(file_model))
        vision_refs = [model for model in model_refs if "image" in self.client.registry.get(model).modalities_input]
        if vision_refs:
            image_model = _prefer_provider(vision_refs, "openai") or vision_refs[0]
            canaries.append(self._image_canary(image_model))
        return canaries

    def _text_canary(self, model_ref: str) -> dict[str, Any]:
        try:
            result = self.client.deal(
                task='Project audit canary. Reply with exactly: "crupier-audit-ok"',
                mode="fast",
                strategy="single",
                constraints={
                    "force_model": model_ref,
                    "max_output_tokens": 16,
                    "max_cost_usd": 0.02,
                    "store_prompt": False,
                    "store_response": False,
                },
                dry_run=False,
                trace="summary",
            )
            output = result.output_text.strip().lower()
            return {
                "id": f"text_smoke:{model_ref}",
                "kind": "text_smoke",
                "ok": "crupier-audit-ok" in output,
                "model": model_ref,
                "latency_ms": result.latency_ms,
                "cost": result.cost.to_dict(),
            }
        except Exception as exc:  # noqa: BLE001 - audit reports all canary failures
            return _canary_error(f"text_smoke:{model_ref}", "text_smoke", model_ref, exc)

    def _structured_canary(self, model_ref: str) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "total": {"type": "number"}},
            "required": ["name", "total"],
            "additionalProperties": False,
        }
        try:
            result = self.client.deal(
                task="Extract invoice data for the project audit.",
                input="Invoice for Ada, total 12.50",
                response_schema=schema,
                constraints={"force_model": model_ref, "max_output_tokens": 120, "max_cost_usd": 0.02},
                dry_run=False,
                trace="summary",
            )
            return {
                "id": f"structured:{model_ref}",
                "kind": "structured",
                "ok": result.output_json == {"name": "Ada", "total": 12.5},
                "model": model_ref,
                "latency_ms": result.latency_ms,
                "cost": result.cost.to_dict(),
            }
        except Exception as exc:  # noqa: BLE001
            return _canary_error(f"structured:{model_ref}", "structured", model_ref, exc)

    def _tool_canary(self, model_ref: str) -> dict[str, Any]:
        def lookup_user(name: str) -> dict[str, str]:
            """Look up a user by name for audit canaries."""

            return {"name": name, "id": "usr_audit", "plan": "pro"}

        try:
            result = self.client.deal(
                task="Use the lookup_user tool to find Ada. Answer with only the user id and plan.",
                tools=[lookup_user],
                constraints={"force_model": model_ref, "max_output_tokens": 120, "max_cost_usd": 0.02},
                dry_run=False,
                trace="summary",
            )
            calls = result.provider_metadata.get("tool_calls", [])
            return {
                "id": f"tool:{model_ref}",
                "kind": "tool",
                "ok": any(call.get("status") == "completed" and call.get("name") == "lookup_user" for call in calls),
                "model": model_ref,
                "latency_ms": result.latency_ms,
                "cost": result.cost.to_dict(),
                "tool_calls": [
                    {
                        "name": call.get("name"),
                        "status": call.get("status"),
                        "requires_approval": call.get("requires_approval", False),
                    }
                    for call in calls
                ],
            }
        except Exception as exc:  # noqa: BLE001
            return _canary_error(f"tool:{model_ref}", "tool", model_ref, exc)

    def _text_file_canary(self, model_ref: str) -> dict[str, Any]:
        file_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="crupier-file-", suffix=".txt", mode="w", delete=False) as handle:
                file_path = handle.name
                handle.write("The audit passphrase is citrine.\n")
            result = self.client.deal(
                task="Read the attached text file. What is the audit passphrase? Answer exactly one word.",
                files=[file_path],
                constraints={"force_model": model_ref, "max_output_tokens": 20, "max_cost_usd": 0.02},
                dry_run=False,
                trace="summary",
            )
            file_context = result.trace.final_quality_signals.get("file_context") if result.trace else {}
            return {
                "id": f"text_file:{model_ref}",
                "kind": "text_file",
                "ok": "citrine" in result.output_text.strip().lower(),
                "model": model_ref,
                "latency_ms": result.latency_ms,
                "cost": result.cost.to_dict(),
                "file_context": file_context,
            }
        except Exception as exc:  # noqa: BLE001
            return _canary_error(f"text_file:{model_ref}", "text_file", model_ref, exc)
        finally:
            if file_path:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except OSError:
                    pass

    def _image_canary(self, model_ref: str) -> dict[str, Any]:
        image_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="crupier-red-", suffix=".png", delete=False) as handle:
                image_path = handle.name
                handle.write(_solid_png_rgb(256, 256, (255, 0, 0)))
            result = self.client.deal(
                task="This image is a single solid color. What color is it? Answer exactly one word.",
                files=[image_path],
                constraints={"force_model": model_ref, "max_output_tokens": 20, "max_cost_usd": 0.02},
                dry_run=False,
                trace="summary",
            )
            calls = result.provider_metadata.get("calls", [])
            return {
                "id": f"image:{model_ref}",
                "kind": "image",
                "ok": "red" in result.output_text.strip().lower(),
                "model": model_ref,
                "latency_ms": result.latency_ms,
                "cost": result.cost.to_dict(),
                "multimodal_images": sum(int(call.get("multimodal_images", 0) or 0) for call in calls),
            }
        except Exception as exc:  # noqa: BLE001
            return _canary_error(f"image:{model_ref}", "image", model_ref, exc)
        finally:
            if image_path:
                try:
                    Path(image_path).unlink(missing_ok=True)
                except OSError:
                    pass


HUMAN_ROUTE_REVIEW_CASES: list[dict[str, Any]] = [
    {
        "id": "fast_short_answer",
        "task": "Answer a short user question in one sentence.",
        "mode": "fast",
        "human_questions": [
            "Would this feel fast enough for an interactive UI?",
            "Is the chosen model cheaper than the project's default frontier model?",
        ],
    },
    {
        "id": "structured_invoice",
        "task": "Extract invoice data as JSON.",
        "input": "Invoice for Ada, total 12.50",
        "mode": "structured",
        "response_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "total": {"type": "number"}},
            "required": ["name", "total"],
        },
        "human_questions": [
            "Does the route prefer a model with verified structured/JSON behavior?",
            "Is there a repair or fallback path if JSON is invalid?",
        ],
    },
    {
        "id": "agentic_code_change",
        "task": "Plan a code-changing agent step, run tests, and explain rollback risks.",
        "mode": "agentic",
        "constraints": {"risk_level": "high"},
        "human_questions": [
            "Does the route include critique/repair or another safety strategy for risky code changes?",
            "Would a maintainer understand why this model was selected?",
        ],
    },
    {
        "id": "research_tradeoffs",
        "task": "Compare two architecture options and identify tradeoffs, blind spots, and recommendation.",
        "mode": "research",
        "human_questions": [
            "Does the route use enough model diversity for a high-uncertainty decision?",
            "Is the added cost of fusion/panel justified for this project?",
        ],
    },
]


SOURCE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
SKIP_DIRS = {
    ".crupier",
    ".eggs",
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "htmlcov",
    "node_modules",
    "site-packages",
    "vendor",
}


COMMENT_PATTERNS: list[tuple[re.Pattern[str], str, str, int, str]] = [
    (
        re.compile(r"\b(from\s+openai\s+import|import\s+openai\b|OpenAI\s*\()"),
        "OpenAI integration point",
        "This call site can usually move behind Crupier with the OpenAI-compatible client, proxy, or autopatch. Decide strict/balanced mode before changing model behavior.",
        2,
        "drop_in",
    ),
    (
        re.compile(r"\b(from\s+anthropic\s+import|import\s+anthropic\b|Anthropic\s*\()"),
        "Anthropic integration point",
        "This call site should declare whether Claude is a required provider or an allowed fallback route through Crupier.",
        2,
        "drop_in",
    ),
    (
        re.compile(r"\b(import\s+ollama\b|ollama\.(chat|generate|embeddings)\s*\()"),
        "Ollama integration point",
        "Confirm whether this project expects Ollama Cloud or an explicit local Ollama host; Crupier treats those as different runtime configurations.",
        2,
        "drop_in",
    ),
    (
        re.compile(r"\b(google\.genai|GenerativeModel|from\s+google\s+import\s+genai)"),
        "Google/Gemini integration point",
        "Route this through a provider adapter once Gemini execution is enabled for the project, or keep it as pass-through until supported.",
        2,
        "drop_in",
    ),
    (
        re.compile(r"\b(model|model_name)\s*=\s*[\"'][^\"']+[\"']"),
        "Hard-coded model choice",
        "Check whether this model must remain strict or should become a Crupier allowlist/profile decision.",
        3,
        "routing",
    ),
    (
        re.compile(
            r"(?i)("
            + ("s" + "k-")
            + r"[a-z0-9_\-]{16,}"
            + "|"
            + ("s" + "k-ant-" + "api" + "03-")
            + r"[a-z0-9_\-]{16,}"
            + "|"
            + ("api" + "03-")
            + r"[a-z0-9_\-]{16,}"
            + "|"
            + r"AIza[0-9A-Za-z_\-]{10,}|api[_-]?key\s*=\s*[\"'][^\"']+[\"'])"
        ),
        "Possible inline credential",
        "Do not keep provider credentials in code. Move keys to environment variables or a local .env file that is ignored by source control.",
        1,
        "security",
    ),
]


def scan_code_comments(
    root: str | Path,
    *,
    paths: list[str | Path] | None = None,
    max_files: int = 200,
    max_file_size: int = 250_000,
) -> list[CodeComment]:
    root_path = Path(root).resolve()
    files = list(_iter_source_files(root_path, paths=paths, max_files=max_files, max_file_size=max_file_size))
    comments: list[CodeComment] = []
    seen: set[tuple[str, int, str]] = set()
    for path in files:
        rel = _relative_path(root_path, path)
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, start=1):
            if "re.compile" in line:
                continue
            for pattern, title, body, priority, category in COMMENT_PATTERNS:
                if not pattern.search(line):
                    continue
                comment_title = title
                comment_body = body
                comment_priority = priority
                comment_category = category
                if title == "Possible inline credential" and _is_test_fixture_path(rel):
                    comment_title = "Credential-like test fixture"
                    comment_body = (
                        "Credential-like value found in test or fixture code. Confirm it is synthetic and never a "
                        "real provider key; keep real keys in ignored environment files."
                    )
                    comment_priority = 3
                    comment_category = "test_fixture"
                key = (rel, number, comment_title)
                if key in seen:
                    continue
                seen.add(key)
                comments.append(
                    CodeComment(
                        file=rel,
                        line=number,
                        title=comment_title,
                        body=comment_body,
                        priority=comment_priority,
                        category=comment_category,
                    )
                )
    return comments


def build_adoption_plan(
    root: str | Path,
    *,
    project: str = "crupier-project",
    paths: list[str | Path] | None = None,
    max_files: int = 200,
) -> ProjectAdoptionPlan:
    root_path = Path(root).resolve()
    comments = scan_code_comments(root_path, paths=paths, max_files=max_files)
    review_summary = summarize_code_comment_reviews(root_path, comments)
    hints = _framework_hints(root_path)
    counts = _comment_counts(comments)
    blockers = _adoption_blockers(comments)
    options = _adoption_options(counts, hints, blocked=bool(blockers))
    recommended = _recommended_option(options)
    checklist = _adoption_checklist(recommended.path, counts, blocked=bool(blockers))
    warnings = _adoption_warnings(counts, hints)
    return ProjectAdoptionPlan(
        project=project,
        generated_at=datetime.now(UTC).isoformat(),
        recommended_path=recommended.path if not blockers else "fix_blockers_first",
        confidence="low" if blockers else _confidence_from_score(recommended.score),
        options=options,
        checklist=checklist,
        blockers=blockers,
        warnings=warnings,
        framework_hints=hints,
        code_comments=comments,
        code_comment_review=review_summary,
    )


def build_adoption_patches(
    root: str | Path,
    *,
    project: str = "crupier-project",
    adoption_path: str = "recommended",
    paths: list[str | Path] | None = None,
    max_files: int = 200,
) -> AdoptionPatchReport:
    root_path = Path(root).resolve()
    plan = build_adoption_plan(root_path, project=project, paths=paths, max_files=max_files)
    target = plan.recommended_path if adoption_path == "recommended" else adoption_path
    blockers = list(plan.blockers)
    warnings = list(plan.warnings)
    patches: list[AdoptionPatchSuggestion]
    if blockers or target == "fix_blockers_first":
        patches = [
            AdoptionPatchSuggestion(
                adoption_path=target,
                title="Fix adoption blockers first",
                status="blocked",
                summary="Crupier will not suggest code patches until blockers are resolved.",
                notes=blockers,
            )
        ]
    elif target == "compat_client":
        patches = _compat_client_patch_suggestions(root_path, paths=paths, max_files=max_files)
    elif target == "proxy":
        patches = _proxy_patch_suggestions()
    elif target == "autopatch":
        patches = _autopatch_patch_suggestions()
    elif target == "native_sdk":
        patches = _native_sdk_patch_suggestions()
    else:
        patches = [
            AdoptionPatchSuggestion(
                adoption_path=target,
                title="Unknown adoption path",
                status="blocked",
                summary=f"No patch generator is available for adoption path {target!r}.",
            )
        ]
        blockers.append(f"Unknown adoption path {target!r}.")
    if not patches:
        patches = [
            AdoptionPatchSuggestion(
                adoption_path=target,
                title="No safe automatic patch suggestions",
                status="manual",
                summary="No narrow import/base-url patch was found. Use the adoption checklist and code comments.",
                notes=["Run `crupier code comments` for call-site level guidance."],
            )
        ]
    return AdoptionPatchReport(
        project=project,
        generated_at=datetime.now(UTC).isoformat(),
        adoption_path=target,
        patches=patches,
        blockers=blockers,
        warnings=warnings,
    )


def build_project_doctor(
    client: Any,
    *,
    paths: list[str | Path] | None = None,
    max_files: int = 200,
    dataset: str | Path | None = None,
    providers: list[str] | None = None,
    include_openai_baseline: bool = True,
    orchestrator_mode: str | None = None,
    real: bool = False,
    all_models: bool = False,
    production: bool = False,
) -> ProjectDoctorReport:
    """Build a non-destructive adoption readiness report for an existing project."""

    root = client.config.root
    project = client.config.project.name
    adoption_plan = build_adoption_plan(root, project=project, paths=paths, max_files=max_files)
    patch_report = build_adoption_patches(
        root,
        project=project,
        adoption_path="recommended",
        paths=paths,
        max_files=max_files,
    )
    audit_report = client.audit.run(
        dataset=dataset,
        providers=providers,
        include_openai_baseline=include_openai_baseline,
        orchestrator_mode=orchestrator_mode,
        real=real,
        all_models=all_models,
        include_code_comments=True,
        code_paths=paths,
        max_code_files=max_files,
    )
    eval_history = client.evals.history()
    feedback_summary = client.feedback.summary()
    applied_feedback_summary = summarize_applied_human_feedback(client.registry, feedback_summary)
    adoption_signoff_summary = summarize_adoption_signoffs(root, project=project)
    gates = _doctor_gates(
        adoption_plan=adoption_plan,
        patch_report=patch_report,
        audit_report=audit_report,
        eval_history=eval_history,
        feedback_summary=feedback_summary,
        applied_feedback_summary=applied_feedback_summary,
        adoption_signoff_summary=adoption_signoff_summary,
        real=real,
        production=production,
    )
    return ProjectDoctorReport(
        project=project,
        generated_at=datetime.now(UTC).isoformat(),
        readiness_mode="production" if production else "adoption",
        adoption_plan=adoption_plan,
        patch_report=patch_report,
        audit_report=audit_report,
        eval_history=eval_history,
        feedback_summary=feedback_summary,
        applied_feedback_summary=applied_feedback_summary,
        adoption_signoff_summary=adoption_signoff_summary,
        gates=gates,
    )


def build_adoption_handoff(
    client: Any,
    *,
    paths: list[str | Path] | None = None,
    max_files: int = 200,
    dataset: str | Path | None = None,
    providers: list[str] | None = None,
    include_openai_baseline: bool = True,
    orchestrator_mode: str | None = None,
    real: bool = False,
    all_models: bool = False,
    production: bool = False,
) -> AdoptionHandoffReport:
    doctor = build_project_doctor(
        client,
        paths=paths,
        max_files=max_files,
        dataset=dataset,
        providers=providers,
        include_openai_baseline=include_openai_baseline,
        orchestrator_mode=orchestrator_mode,
        real=real,
        all_models=all_models,
        production=production,
    )
    root = Path(client.config.root)
    return build_adoption_handoff_from_doctor(
        root,
        project=client.config.project.name,
        doctor=doctor,
        paths=paths,
    )


def build_adoption_handoff_from_doctor(
    root: str | Path,
    *,
    project: str,
    doctor: ProjectDoctorReport,
    paths: list[str | Path] | None = None,
) -> AdoptionHandoffReport:
    root_path = Path(root)
    artifacts = _handoff_artifacts(root_path)
    actions, commands = _handoff_actions(doctor, artifacts, paths=paths)
    if doctor.readiness_mode == "config_free_adoption":
        actions.append("Initialize Crupier configuration before production rollout.")
        commands.append("crupier init")
        commands.append("crupier adopt doctor --production --real --provider anthropic --provider ollama")
    status = "ready" if doctor.ready and not actions else ("blocked" if not doctor.ready else "needs-human-review")
    return AdoptionHandoffReport(
        project=project,
        generated_at=datetime.now(UTC).isoformat(),
        status=status,
        doctor=doctor,
        artifacts=artifacts,
        required_human_actions=_dedupe(actions),
        suggested_commands=_dedupe(commands),
    )


def build_config_free_adoption_handoff(
    root: str | Path,
    *,
    project: str,
    paths: list[str | Path] | None = None,
    max_files: int = 200,
) -> AdoptionHandoffReport:
    """Build a reviewer handoff before a repo has crupier.toml."""

    root_path = Path(root)
    doctor = build_config_free_project_doctor(root_path, project=project, paths=paths, max_files=max_files)
    return build_adoption_handoff_from_doctor(
        root_path,
        project=project,
        doctor=doctor,
        paths=paths,
    )


def build_config_free_project_doctor(
    root: str | Path,
    *,
    project: str,
    paths: list[str | Path] | None = None,
    max_files: int = 200,
) -> ProjectDoctorReport:
    """Build an offline adoption doctor before a repo has crupier.toml."""

    root_path = Path(root)
    adoption_plan = build_adoption_plan(root_path, project=project, paths=paths, max_files=max_files)
    patch_report = build_adoption_patches(
        root_path,
        project=project,
        adoption_path="recommended",
        paths=paths,
        max_files=max_files,
    )
    audit_report = ProjectAuditReport(
        project=project,
        generated_at=datetime.now(UTC).isoformat(),
        checks=[
            AuditCheck(
                id="config_free_scan",
                status="pass",
                severity="medium",
                summary="Config-free adoption scan completed without provider calls.",
                actions=["Run `crupier init` before production canaries, eval history, and human-feedback gates."],
            ),
            _code_comment_check(adoption_plan.code_comments),
        ],
        route_reviews=[],
        code_comments=adoption_plan.code_comments,
    )
    feedback_summary = {"count": 0, "dry_run_source_count": 0, "production_feedback_count": 0, "groups": []}
    adoption_signoff_summary = summarize_adoption_signoffs(root_path, project=project)
    doctor = ProjectDoctorReport(
        project=project,
        generated_at=datetime.now(UTC).isoformat(),
        readiness_mode="config_free_adoption",
        adoption_plan=adoption_plan,
        patch_report=patch_report,
        audit_report=audit_report,
        eval_history={"total_runs": 0, "status": "not_checked", "reason": "config_free_adoption"},
        feedback_summary=feedback_summary,
        applied_feedback_summary={"count": 0, "applied_count": 0, "pending_count": 0, "applied": [], "pending": []},
        adoption_signoff_summary=adoption_signoff_summary,
        gates=[
            _doctor_adoption_blocker_gate(adoption_plan),
            _doctor_adoption_path_gate(adoption_plan),
            _doctor_patch_gate(patch_report),
            _doctor_audit_gate(audit_report),
            DoctorGate(
                id="configuration",
                status="warn",
                severity="medium",
                summary="No crupier.toml was required for this offline adoption report; provider and feedback gates are not evaluated yet.",
                evidence={"mode": "config_free_adoption"},
                actions=["Run `crupier init`, configure providers, then rerun `crupier adopt doctor --production --real`."],
            ),
            _doctor_adoption_signoff_gate(adoption_signoff_summary, production=False),
            _doctor_code_comment_gate(adoption_plan),
        ],
    )
    return doctor


def summarize_applied_human_feedback(registry: Any, feedback_summary: dict[str, Any]) -> dict[str, Any]:
    groups = list(feedback_summary.get("groups", []) or [])
    applied: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for group in groups:
        model = str(group.get("model") or "")
        mode = str(group.get("mode") or "overall")
        score_key = f"human:{mode}"
        expected_score = group.get("score_delta")
        try:
            card = registry.get(model)
            actual_score = card.local_eval_scores.get(score_key)
        except Exception as exc:  # noqa: BLE001 - this is a readiness report, not a hard registry operation
            pending.append(
                {
                    "model": model,
                    "mode": mode,
                    "score_key": score_key,
                    "expected_score": expected_score,
                    "reason": str(exc),
                }
            )
            continue
        item = {
            "model": model,
            "mode": mode,
            "score_key": score_key,
            "expected_score": expected_score,
            "actual_score": actual_score,
            "count": group.get("count", 0),
        }
        if actual_score == expected_score:
            applied.append(item)
        else:
            pending.append({**item, "reason": "score not applied to capability card"})
    return {
        "count": len(groups),
        "applied_count": len(applied),
        "pending_count": len(pending),
        "applied": applied,
        "pending": pending,
    }


def record_adoption_signoff(
    root: str | Path,
    *,
    project: str,
    verdict: str,
    reviewer_hash: str | None = None,
    note: str = "",
    handoff: str | Path | None = None,
    adoption_path: str | None = None,
) -> dict[str, Any]:
    """Record a project-level human adoption decision without storing prompts/responses."""

    normalized = verdict.strip().lower().replace("-", "_")
    if normalized not in ADOPTION_SIGNOFF_VERDICTS:
        raise CrupierError(
            f"Unknown adoption signoff verdict {verdict!r}. "
            f"Use one of: {', '.join(sorted(ADOPTION_SIGNOFF_VERDICTS))}."
        )
    root_path = Path(root)
    signoff_dir = root_path / ".crupier" / "handoffs"
    signoff_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    record = {
        "schema_version": 1,
        "signoff_id": f"as_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{normalized}",
        "created_at": created_at,
        "project": project,
        "verdict": normalized,
        "reviewer_hash": reviewer_hash,
        "note": _redact_secrets(" ".join(str(note or "").split())[:1000]),
        "handoff": str(handoff) if handoff else None,
        "adoption_path": adoption_path,
    }
    path = signoff_dir / "signoffs.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({key: value for key, value in record.items() if value not in (None, "")}, sort_keys=True) + "\n")
    return {**record, "path": str(path)}


def read_adoption_signoffs(root: str | Path, *, project: str | None = None) -> list[dict[str, Any]]:
    path = Path(root) / ".crupier" / "handoffs" / "signoffs.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if project and data.get("project") not in {project, None, ""}:
            continue
        records.append(data)
    return records


def summarize_adoption_signoffs(root: str | Path, *, project: str | None = None) -> dict[str, Any]:
    records = read_adoption_signoffs(root, project=project)
    counts: dict[str, int] = {}
    for record in records:
        verdict = str(record.get("verdict") or "unknown")
        counts[verdict] = counts.get(verdict, 0) + 1
    latest = records[-1] if records else None
    latest_verdict = str(latest.get("verdict")) if latest else None
    if latest_verdict == "approve":
        status = "approved"
    elif latest_verdict in {"reject", "needs_work"}:
        status = latest_verdict
    else:
        status = "missing"
    return {
        "count": len(records),
        "status": status,
        "latest": latest,
        "counts": counts,
    }


def _doctor_gates(
    *,
    adoption_plan: ProjectAdoptionPlan,
    patch_report: AdoptionPatchReport,
    audit_report: ProjectAuditReport,
    eval_history: Any,
    feedback_summary: dict[str, Any],
    applied_feedback_summary: dict[str, Any],
    adoption_signoff_summary: dict[str, Any],
    real: bool,
    production: bool,
) -> list[DoctorGate]:
    gates = [
        _doctor_adoption_blocker_gate(adoption_plan),
        _doctor_adoption_path_gate(adoption_plan),
        _doctor_patch_gate(patch_report),
        _doctor_audit_gate(audit_report),
        _doctor_real_canary_gate(audit_report, real=real, production=production),
        _doctor_eval_history_gate(eval_history, production=production),
        _doctor_feedback_gate(feedback_summary, applied_feedback_summary, production=production),
        _doctor_adoption_signoff_gate(adoption_signoff_summary, production=production),
        _doctor_code_comment_gate(adoption_plan),
    ]
    return gates


def _doctor_adoption_blocker_gate(plan: ProjectAdoptionPlan) -> DoctorGate:
    if plan.blockers:
        return DoctorGate(
            id="adoption_blockers",
            status="fail",
            severity="high",
            summary=f"{len(plan.blockers)} adoption blocker(s) must be resolved first.",
            evidence={"blockers": plan.blockers},
            actions=list(plan.blockers),
        )
    return DoctorGate(
        id="adoption_blockers",
        status="pass",
        severity="high",
        summary="No adoption blockers detected.",
        evidence={"blockers": []},
    )


def _doctor_adoption_path_gate(plan: ProjectAdoptionPlan) -> DoctorGate:
    if not plan.ready:
        return DoctorGate(
            id="adoption_path",
            status="fail",
            severity="high",
            summary="No adoption path is recommended until blockers are fixed.",
            evidence={"recommended_path": plan.recommended_path, "confidence": plan.confidence},
            actions=["Fix adoption blockers, then rerun `crupier adopt doctor`."],
        )
    return DoctorGate(
        id="adoption_path",
        status="pass",
        severity="high",
        summary=f"Recommended adoption path is {plan.recommended_path} with {plan.confidence} confidence.",
        evidence={
            "recommended_path": plan.recommended_path,
            "confidence": plan.confidence,
            "options": [option.to_dict() for option in plan.options],
        },
        actions=plan.checklist[:5],
    )


def _doctor_patch_gate(report: AdoptionPatchReport) -> DoctorGate:
    if report.blockers:
        return DoctorGate(
            id="patch_suggestions",
            status="fail",
            severity="high",
            summary="Patch suggestions are blocked.",
            evidence={"blockers": report.blockers},
            actions=list(report.blockers),
        )
    suggested = [patch for patch in report.patches if patch.status == "suggested" and patch.diff]
    manual = [patch for patch in report.patches if patch.status == "manual"]
    if suggested:
        status = "pass"
        summary = f"{len(suggested)} reviewable patch suggestion(s) are available."
    else:
        status = "warn"
        summary = "Only manual adoption guidance is available for the recommended path."
    commands: list[str] = []
    for patch in report.patches:
        commands.extend(patch.commands)
    return DoctorGate(
        id="patch_suggestions",
        status=status,
        severity="medium",
        summary=summary,
        evidence={
            "adoption_path": report.adoption_path,
            "patch_count": len(report.patches),
            "suggested_count": len(suggested),
            "manual_count": len(manual),
        },
        actions=commands[:5],
    )


def _doctor_audit_gate(report: ProjectAuditReport) -> DoctorGate:
    failures = [check.id for check in report.checks if check.status == "fail"]
    warnings = [check.id for check in report.checks if check.status == "warn"]
    if failures:
        status = "fail"
        summary = f"Project audit has {len(failures)} failing check(s)."
    elif warnings:
        status = "warn"
        summary = f"Project audit passed with {len(warnings)} warning(s)."
    else:
        status = "pass"
        summary = "Project audit checks passed."
    return DoctorGate(
        id="project_audit",
        status=status,
        severity="high" if failures else "medium",
        summary=summary,
        evidence={"summary": report.summary, "failures": failures, "warnings": warnings},
        actions=_audit_actions(report),
    )


def _doctor_real_canary_gate(report: ProjectAuditReport, *, real: bool, production: bool) -> DoctorGate:
    canary_check = next((check for check in report.checks if check.id == "real_canaries"), None)
    if not real:
        return DoctorGate(
            id="real_canaries",
            status="fail" if production else "warn",
            severity="high",
            summary="Real provider canaries were not run in this doctor report.",
            evidence=canary_check.evidence if canary_check else {"count": 0},
            actions=["Run `crupier adopt doctor --production --real` before production adoption."],
        )
    failures = [item.get("id") for item in report.real_canaries if not item.get("ok")]
    return DoctorGate(
        id="real_canaries",
        status="pass" if not failures and report.real_canaries else "fail",
        severity="high",
        summary="Real provider canaries passed." if not failures and report.real_canaries else "Real provider canaries failed.",
        evidence={"count": len(report.real_canaries), "failures": failures},
        actions=["Inspect failed canaries and rerun `crupier verify`."]
        if failures or not report.real_canaries
        else [],
    )


def _doctor_eval_history_gate(history: Any, *, production: bool) -> DoctorGate:
    total_runs = int(getattr(history, "total_runs", 0) or 0)
    scores = list(getattr(history, "model_scores", []) or [])
    confident_scores = [
        score
        for score in scores
        if getattr(score, "confidence", "") in {"medium", "high"} and getattr(score, "appearances", 0)
    ]
    if total_runs == 0:
        return DoctorGate(
            id="eval_history",
            status="fail" if production else "warn",
            severity="medium",
            summary="No recorded A/B compare history exists yet.",
            evidence={"total_runs": 0},
            actions=[
                "Run `crupier eval compare-dataset --record-history --write-report` with project-relevant cases.",
                "Create a review packet with `crupier feedback review --compare-report <report.json>`.",
            ],
        )
    return DoctorGate(
        id="eval_history",
        status="pass" if confident_scores else ("fail" if production else "warn"),
        severity="medium",
        summary=f"Found {total_runs} recorded compare run(s)."
        if confident_scores
        else "Compare history exists but confidence is still low.",
        evidence={
            "total_runs": total_runs,
            "last_run_at": getattr(history, "last_run_at", None),
            "model_scores": [
                {
                    "model": getattr(score, "model", ""),
                    "mode": getattr(score, "mode", ""),
                    "appearances": getattr(score, "appearances", 0),
                    "confidence": getattr(score, "confidence", ""),
                    "trend": getattr(score, "trend", ""),
                }
                for score in scores[:20]
            ],
        },
        actions=[] if confident_scores else ["Record more compare-dataset runs before treating eval scores as stable."],
    )


def _doctor_feedback_gate(
    summary: dict[str, Any],
    applied_summary: dict[str, Any],
    *,
    production: bool,
) -> DoctorGate:
    count = int(summary.get("count", 0) or 0)
    dry_run_source_count = int(summary.get("dry_run_source_count", 0) or 0)
    production_feedback_count = int(summary.get("production_feedback_count", count - dry_run_source_count) or 0)
    if count == 0:
        return DoctorGate(
            id="human_feedback",
            status="fail" if production else "warn",
            severity="medium",
            summary="No human feedback has been recorded yet.",
            evidence={"count": 0},
            actions=[
                "Generate review packets with `crupier feedback review --compare-report <report.json>`.",
                "Use one of the packet's `crupier feedback record` commands after a maintainer reviews the result.",
            ],
        )
    if production and production_feedback_count == 0:
        return DoctorGate(
            id="human_feedback",
            status="fail",
            severity="medium",
            summary=(
                f"{count} human feedback record(s) found, but all came from dry-run compare reports."
            ),
            evidence={"feedback": summary},
            actions=[
                "Run a real compare with `--no-dry-run`, generate a fresh review packet, and import/record that human verdict.",
                "Use dry-run feedback only for non-production calibration.",
            ],
        )
    pending_applied = int(applied_summary.get("pending_count", 0) or 0)
    applied_count = int(applied_summary.get("applied_count", 0) or 0)
    if pending_applied:
        return DoctorGate(
            id="human_feedback",
            status="fail" if production else "warn",
            severity="medium",
            summary=f"{count} human feedback record(s) found, but {pending_applied} model/mode score group(s) are not applied.",
            evidence={"feedback": summary, "applied_feedback": applied_summary},
            actions=["Run `crupier feedback apply` so human judgement affects future route selection."],
        )
    rejected = [
        group
        for group in summary.get("groups", [])
        if int(group.get("verdicts", {}).get("reject", 0) or 0) > 0 or group.get("status") == "negative"
    ]
    return DoctorGate(
        id="human_feedback",
        status=("fail" if production else "warn") if rejected else "pass",
        severity="medium",
        summary=f"{count} human feedback record(s) found and {applied_count} score group(s) applied."
        if not rejected
        else f"{count} human feedback record(s) found, including rejected routes.",
        evidence={"count": count, "groups": summary.get("groups", [])[:20], "applied_feedback": applied_summary},
        actions=["Review rejected feedback before applying human scores to model cards."] if rejected else [],
    )


def _doctor_adoption_signoff_gate(summary: dict[str, Any], *, production: bool) -> DoctorGate:
    latest = summary.get("latest")
    status = str(summary.get("status") or "missing")
    if latest and status == "approved":
        return DoctorGate(
            id="adoption_signoff",
            status="pass",
            severity="high",
            summary="Latest human adoption signoff approves rollout.",
            evidence=summary,
        )
    if latest and status in {"reject", "needs_work"}:
        return DoctorGate(
            id="adoption_signoff",
            status="fail",
            severity="high",
            summary=f"Latest human adoption signoff is {status}; rollout is blocked.",
            evidence=summary,
            actions=["Address the signoff note, regenerate handoff, and record a new approval before rollout."],
        )
    return DoctorGate(
        id="adoption_signoff",
        status="fail" if production else "warn",
        severity="high" if production else "medium",
        summary="No project-level human adoption signoff has been recorded yet.",
        evidence=summary,
        actions=[
            "After reviewing the latest handoff, run `crupier adopt signoff --verdict approve` or record rejection/needs_work."
        ],
    )


def _doctor_code_comment_gate(plan: ProjectAdoptionPlan) -> DoctorGate:
    comments = plan.code_comments
    review = plan.code_comment_review or CodeCommentReviewSummary(
        count=len(comments),
        reviewed_count=0,
        pending_count=len(comments),
        pending=comments,
    )
    pending = list(review.pending)
    high_priority = [comment.to_dict() for comment in pending if comment.priority <= 1]
    if not comments:
        return DoctorGate(
            id="programmer_code_comments",
            status="pass",
            severity="medium",
            summary="No AI integration hotspots found in scanned files.",
            evidence={"count": 0, "reviewed_count": 0, "pending_count": 0, "high_priority": []},
        )
    if review.ready:
        return DoctorGate(
            id="programmer_code_comments",
            status="pass",
            severity="medium",
            summary=f"{len(comments)} programmer code comment(s) have been reviewed.",
            evidence=review.to_dict(),
        )
    return DoctorGate(
        id="programmer_code_comments",
        status="warn",
        severity="medium",
        summary=f"{review.pending_count} of {len(comments)} programmer code comment(s) need review.",
        evidence={**review.to_dict(), "high_priority": high_priority[:20]},
        actions=[
            "Run `crupier code comments --write-report` and inspect the report before editing.",
            "After a programmer reviews the current comments, run `crupier code comments --ack-reviewed`.",
        ],
    )


def _audit_actions(report: ProjectAuditReport) -> list[str]:
    actions: list[str] = []
    for check in report.checks:
        if check.status in {"fail", "warn"}:
            actions.extend(check.actions)
    return actions[:8]


def write_project_audit_report(root: str | Path, report: ProjectAuditReport) -> list[Path]:
    audits_dir = Path(root) / ".crupier" / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = audits_dir / f"project_audit_{timestamp}.json"
    md_path = audits_dir / f"project_audit_{timestamp}.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(format_project_audit_markdown(report), encoding="utf-8")
    return [json_path, md_path]


def write_adoption_patch_report(root: str | Path, report: AdoptionPatchReport) -> list[Path]:
    audits_dir = Path(root) / ".crupier" / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = audits_dir / f"adoption_patches_{timestamp}.json"
    md_path = audits_dir / f"adoption_patches_{timestamp}.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(format_adoption_patch_markdown(report), encoding="utf-8")
    return [json_path, md_path]


def write_adoption_plan_report(root: str | Path, plan: ProjectAdoptionPlan) -> list[Path]:
    audits_dir = Path(root) / ".crupier" / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = audits_dir / f"adoption_plan_{timestamp}.json"
    md_path = audits_dir / f"adoption_plan_{timestamp}.md"
    json_path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(format_adoption_plan_markdown(plan), encoding="utf-8")
    return [json_path, md_path]


def write_project_doctor_report(root: str | Path, report: ProjectDoctorReport) -> list[Path]:
    audits_dir = Path(root) / ".crupier" / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = audits_dir / f"project_doctor_{timestamp}.json"
    md_path = audits_dir / f"project_doctor_{timestamp}.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(format_project_doctor_markdown(report), encoding="utf-8")
    return [json_path, md_path]


def write_adoption_handoff_report(root: str | Path, report: AdoptionHandoffReport) -> list[Path]:
    handoff_dir = Path(root) / ".crupier" / "handoffs"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = handoff_dir / f"adoption_handoff_{timestamp}.json"
    md_path = handoff_dir / f"adoption_handoff_{timestamp}.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(format_adoption_handoff_markdown(report), encoding="utf-8")
    report.written_files = [str(json_path), str(md_path)]
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [json_path, md_path]


def write_adoption_package_index(root: str | Path, payload: dict[str, Any]) -> tuple[list[Path], dict[str, Any]]:
    package_dir = Path(root) / ".crupier" / "packages"
    package_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = package_dir / f"adoption_package_{timestamp}.json"
    md_path = package_dir / f"adoption_package_{timestamp}.md"
    enriched = json.loads(json.dumps(payload))
    artifact_groups = dict(enriched.get("artifact_groups", {}) or {})
    artifact_groups["adoption_package"] = [str(json_path), str(md_path)]
    enriched["artifact_groups"] = artifact_groups
    enriched["written_files"] = [path for paths in artifact_groups.values() for path in paths]
    json_path.write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(format_adoption_package_markdown(enriched), encoding="utf-8")
    return [json_path, md_path], enriched


def write_code_comments_report(root: str | Path, comments: list[CodeComment]) -> list[Path]:
    audits_dir = Path(root) / ".crupier" / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = audits_dir / f"code_comments_{timestamp}.json"
    md_path = audits_dir / f"code_comments_{timestamp}.md"
    payload = {"count": len(comments), "comments": [comment.to_dict() for comment in comments]}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(format_code_comments_markdown(comments), encoding="utf-8")
    return [json_path, md_path]


def write_code_review_comments(root: str | Path, comments: list[CodeComment]) -> list[Path]:
    review_dir = Path(root) / ".crupier" / "code-comments"
    review_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = review_dir / f"review_comments_{timestamp}.jsonl"
    md_path = review_dir / f"review_comments_{timestamp}.md"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for comment in comments:
            record = {
                "schema_version": 1,
                "fingerprint": _code_comment_fingerprint(comment),
                "file": comment.file,
                "line": comment.line,
                "priority": comment.priority,
                "category": comment.category,
                "title": comment.title,
                "body": comment.body,
                "review_comment": _code_review_comment_body(comment),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    md_path.write_text(format_code_review_comments_markdown(comments), encoding="utf-8")
    return [jsonl_path, md_path]


def write_code_comment_decision_template(root: str | Path, comments: list[CodeComment]) -> Path:
    decisions_dir = Path(root) / ".crupier" / "code-comments" / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = decisions_dir / f"code_comment_decisions_{timestamp}.json"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "instructions": (
            "Set each verdict to accepted, false_positive, not_applicable, reviewed, resolved, "
            "needs_change, rejected, or unresolved. Only reviewed/resolved verdicts close the "
            "programmer comment gate."
        ),
        "allowed_verdicts": sorted(CODE_COMMENT_DECISION_VERDICTS),
        "comments": [
            {
                "fingerprint": _code_comment_fingerprint(comment),
                "file": comment.file,
                "line": comment.line,
                "title": comment.title,
                "priority": comment.priority,
                "category": comment.category,
                "body": comment.body,
                "verdict": "unresolved",
                "note": "",
            }
            for comment in comments
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def import_code_comment_decisions(
    root: str | Path,
    comments: list[CodeComment],
    decisions_path: str | Path,
    *,
    reviewer_hash: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Import a human-edited code-comment decision template without storing source snippets."""

    root_path = Path(root)
    path = Path(decisions_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CrupierError(f"Could not read code comment decisions from {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CrupierError(f"Invalid code comment decisions JSON in {path}: {exc}") from exc
    items = payload.get("comments") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise CrupierError("Code comment decisions file must contain a comments list.")

    current = {_code_comment_fingerprint(comment): comment for comment in comments}
    seen: set[str] = set()
    reviewed: dict[str, CodeComment] = {}
    decisions: list[dict[str, Any]] = []
    unknown_fingerprints: list[str] = []
    decision_counts: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        fingerprint = str(item.get("fingerprint") or "")
        verdict = str(item.get("verdict") or "unresolved").strip().lower().replace("-", "_")
        if verdict not in CODE_COMMENT_DECISION_VERDICTS:
            raise CrupierError(
                f"Unknown code comment verdict {verdict!r}. "
                f"Use one of: {', '.join(sorted(CODE_COMMENT_DECISION_VERDICTS))}."
            )
        decision_counts[verdict] = decision_counts.get(verdict, 0) + 1
        comment = current.get(fingerprint)
        if comment is None:
            if fingerprint:
                unknown_fingerprints.append(fingerprint)
            continue
        seen.add(fingerprint)
        if verdict in CODE_COMMENT_DECISION_REVIEWED:
            reviewed[fingerprint] = comment
        decisions.append(
            {
                "fingerprint": fingerprint,
                "file": comment.file,
                "line": comment.line,
                "title": comment.title,
                "priority": comment.priority,
                "category": comment.category,
                "verdict": verdict,
                "note": _redact_secrets(" ".join(str(item.get("note") or "").split())[:500]),
            }
        )

    missing_current = sorted(set(current) - seen)
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    review_id = f"ccr_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_decisions"
    record = {
        "schema_version": 1,
        "review_id": review_id,
        "source": "code_comment_decisions",
        "created_at": created_at,
        "reviewer_hash": reviewer_hash,
        "note": _redact_secrets(" ".join(str(note or "").split())[:1000]),
        "decision_file": str(path),
        "comment_count": len(reviewed),
        "comment_fingerprints": list(reviewed),
        "decision_counts": decision_counts,
        "unknown_fingerprints": unknown_fingerprints[:50],
        "unknown_count": len(unknown_fingerprints),
        "missing_current_count": len(missing_current),
        "pending_decision_count": len(current) - len(reviewed),
        "decisions": decisions,
    }
    reviews_dir = root_path / ".crupier" / "code-comments"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    log_path = reviews_dir / "reviews.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({key: value for key, value in record.items() if value not in (None, "")}, sort_keys=True) + "\n"
        )
    return {**record, "path": str(log_path)}


def write_code_comments_sarif(root: str | Path, comments: list[CodeComment]) -> Path:
    sarif_dir = Path(root) / ".crupier" / "code-comments"
    sarif_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = sarif_dir / f"code_comments_{timestamp}.sarif"
    path.write_text(json.dumps(format_code_comments_sarif(comments), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def acknowledge_code_comments(
    root: str | Path,
    comments: list[CodeComment],
    *,
    reviewer_hash: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    root_path = Path(root)
    reviews_dir = root_path / ".crupier" / "code-comments"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    review_id = f"ccr_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{len(comments)}"
    record = {
        "schema_version": 1,
        "review_id": review_id,
        "created_at": created_at,
        "reviewer_hash": reviewer_hash,
        "note": _redact_secrets(" ".join(str(note or "").split())[:1000]),
        "comment_count": len(comments),
        "comment_fingerprints": [_code_comment_fingerprint(comment) for comment in comments],
        "comments": [
            {
                "fingerprint": _code_comment_fingerprint(comment),
                "file": comment.file,
                "line": comment.line,
                "title": comment.title,
                "priority": comment.priority,
                "category": comment.category,
            }
            for comment in comments
        ],
    }
    path = reviews_dir / "reviews.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({key: value for key, value in record.items() if value not in (None, "")}, sort_keys=True) + "\n")
    return {**record, "path": str(path)}


def summarize_code_comment_reviews(root: str | Path, comments: list[CodeComment]) -> CodeCommentReviewSummary:
    current = {_code_comment_fingerprint(comment): comment for comment in comments}
    reviewed: set[str] = set()
    latest_review_at: str | None = None
    for record in _read_code_comment_review_records(Path(root)):
        created_at = record.get("created_at")
        if created_at and (latest_review_at is None or str(created_at) > latest_review_at):
            latest_review_at = str(created_at)
        for fingerprint in record.get("comment_fingerprints", []) or []:
            reviewed.add(str(fingerprint))
    pending = [comment for fingerprint, comment in current.items() if fingerprint not in reviewed]
    return CodeCommentReviewSummary(
        count=len(comments),
        reviewed_count=len(current) - len(pending),
        pending_count=len(pending),
        latest_review_at=latest_review_at,
        pending=pending,
    )


def format_project_doctor_markdown(report: ProjectDoctorReport) -> str:
    lines = [
        f"# Crupier Project Doctor: {report.project}",
        "",
        f"Generated: {report.generated_at}",
        f"Status: {report.status}",
        f"Readiness mode: {report.readiness_mode}",
        f"Recommended path: {report.recommended_path}",
        f"Confidence: {report.confidence}",
        f"Summary: {', '.join(f'{name}={count}' for name, count in sorted(report.summary.items()))}",
    ]
    _append_review_contract_markdown(lines, report.review_contract)
    lines.extend(["", "## Gates"])
    for gate in report.gates:
        lines.append(f"- {gate.status.upper()} [{gate.severity}] {gate.id}: {gate.summary}")
        for action in gate.actions:
            lines.append(f"  - action: {action}")
    if report.adoption_plan.blockers:
        lines.extend(["", "## Blockers"])
        for blocker in report.adoption_plan.blockers:
            lines.append(f"- {blocker}")
    lines.extend(["", "## Adoption Checklist"])
    for item in report.adoption_plan.checklist:
        lines.append(f"- {item}")
    lines.extend(["", "## Patch Suggestions"])
    for patch in report.patch_report.patches:
        lines.append(f"- {patch.status.upper()} {patch.title}")
        if patch.commands:
            lines.append(f"  - commands: {', '.join(patch.commands)}")
    if report.audit_report.real_canaries:
        lines.extend(["", "## Real Canaries"])
        for canary in report.audit_report.real_canaries:
            status = "PASS" if canary.get("ok") else "FAIL"
            lines.append(f"- {status} {canary.get('id')}")
    if report.adoption_plan.code_comments:
        lines.extend(["", "## Programmer Code Comments"])
        for comment in report.adoption_plan.code_comments[:50]:
            lines.append(f"- P{comment.priority} {comment.file}:{comment.line} {comment.title}")
    lines.append("")
    return "\n".join(lines)


def format_adoption_handoff_markdown(report: AdoptionHandoffReport) -> str:
    lines = [
        f"# Crupier Adoption Handoff: {report.project}",
        "",
        f"Generated: {report.generated_at}",
        f"Status: {report.status}",
        f"Doctor status: {report.doctor.status}",
        f"Readiness mode: {report.doctor.readiness_mode}",
        f"Recommended path: {report.doctor.recommended_path}",
        f"Confidence: {report.doctor.confidence}",
        "",
        "## Review Meaning",
        "",
        "A ready doctor means no automatic gate is failing. It is not a human approval.",
        "`needs-human-review` means a programmer or product owner still has to inspect the route, outputs, cost/latency, and integration notes before rollout.",
    ]
    _append_review_contract_markdown(lines, report.doctor.review_contract)
    lines.extend(["", "## Current Gates"])
    for gate in report.doctor.gates:
        lines.append(f"- {gate.status.upper()} [{gate.severity}] {gate.id}: {gate.summary}")
    lines.extend(["", "## Human Signoff Checklist"])
    for item in _handoff_signoff_checklist(report):
        lines.append(f"- {item}")
    lines.extend(["", "## Human Actions"])
    if report.required_human_actions:
        for action in report.required_human_actions:
            lines.append(f"- {action}")
    else:
        lines.append("- No required human actions remain in this handoff.")
    lines.extend(["", "## Suggested Commands"])
    for command in report.suggested_commands:
        lines.append(f"- `{command}`")
    pending_comments = _pending_handoff_code_comments(report)
    if pending_comments:
        lines.extend(["", "## Pending Programmer Code Comments"])
        for comment in pending_comments[:20]:
            lines.append(f"- P{comment.priority} {comment.file}:{comment.line} {comment.title}")
            lines.append(f"  - {comment.body}")
        if len(pending_comments) > 20:
            lines.append(f"- ... {len(pending_comments) - 20} more comment(s) in the code-comment report.")
    lines.extend(["", "## Recent Artifacts"])
    for name, paths in sorted(report.artifacts.items()):
        lines.append(f"### {name}")
        if not paths:
            lines.append("- none found")
        for path in paths:
            lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def format_adoption_package_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Crupier Adoption Package: {payload.get('project', 'unknown')}",
        "",
        f"Status: {payload.get('status', 'unknown')}",
        f"Doctor status: {payload.get('doctor_status', 'unknown')}",
        f"Readiness mode: {payload.get('readiness_mode', 'unknown')}",
        f"Recommended path: {payload.get('recommended_path', 'unknown')}",
    ]
    _append_review_contract_markdown(lines, payload.get("review_contract", {}) or {})
    lines.extend(["", "## Open First"])
    artifact_groups = payload.get("artifact_groups", {}) or {}
    handoffs = list(artifact_groups.get("adoption_handoff", []) or [])
    doctors = list(artifact_groups.get("project_doctor", []) or [])
    review_packets = list(artifact_groups.get("code_review_comments", []) or [])
    sarif_files = list(artifact_groups.get("code_sarif", []) or [])
    decision_templates = list(artifact_groups.get("code_comment_decisions", []) or [])
    if handoffs:
        lines.append(f"- Handoff: `{_first_markdown(handoffs) or handoffs[-1]}`")
    if doctors:
        lines.append(f"- Doctor: `{_first_markdown(doctors) or doctors[-1]}`")
    if review_packets:
        lines.append(f"- Programmer review comments: `{_first_markdown(review_packets) or review_packets[-1]}`")
    if sarif_files:
        lines.append(f"- SARIF annotations: `{sarif_files[0]}`")
    if decision_templates:
        lines.append(f"- Programmer decision template: `{decision_templates[0]}`")
    lines.extend(["", "## Required Human Actions"])
    actions = payload.get("required_human_actions", []) or []
    if actions:
        for action in actions:
            lines.append(f"- {action}")
    else:
        lines.append("- No required human actions remain in this package.")
    lines.extend(["", "## Suggested Commands"])
    commands = payload.get("suggested_commands", []) or []
    if commands:
        for command in commands:
            lines.append(f"- `{command}`")
    else:
        lines.append("- No suggested commands.")
    lines.extend(["", "## Artifacts"])
    for name, paths in sorted(artifact_groups.items()):
        lines.append(f"### {name}")
        for path in paths:
            lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def _append_review_contract_markdown(lines: list[str], contract: dict[str, Any]) -> None:
    if not contract:
        return
    lines.extend(
        [
            "",
            "## Review Contract",
            f"- Overall: {contract.get('overall_status', 'unknown')}",
            f"- Technical: {contract.get('technical_status', 'unknown')}",
            f"- Human: {contract.get('human_status', 'unknown')}",
            f"- Auto-approval blocked: {str(bool(contract.get('must_not_auto_approve', True))).lower()}",
        ]
    )
    summary = contract.get("summary")
    if summary:
        lines.append(f"- Summary: {summary}")
    human_open = contract.get("human_open_gates") or []
    if human_open:
        lines.append(f"- Human open gates: {', '.join(str(item) for item in human_open)}")
    technical_blockers = contract.get("technical_blockers") or []
    if technical_blockers:
        lines.append(f"- Technical blockers: {', '.join(str(item) for item in technical_blockers)}")


def _first_markdown(paths: list[str]) -> str | None:
    return next((path for path in paths if path.endswith(".md")), None)


def _handoff_signoff_checklist(report: AdoptionHandoffReport) -> list[str]:
    gates = {gate.id: gate for gate in report.doctor.gates}
    checklist: list[str] = [
        "Confirm the recommended adoption path is acceptable for this project's ownership, tests, deployment model, and rollback plan.",
        "Reject or delay rollout if real outputs are technically valid but not useful, safe, cheap enough, or maintainable for the project.",
    ]
    if report.doctor.readiness_mode == "config_free_adoption":
        checklist.append("Initialize Crupier configuration before treating this handoff as production evidence.")
    if (gate := gates.get("human_feedback")) and gate.status != "pass":
        checklist.append("Review real compare output and import/apply human verdicts so the selector learns from the judgement.")
    if (gate := gates.get("adoption_signoff")) and gate.status != "pass":
        checklist.append("Record the final adoption signoff as approve, reject, or needs_work after reviewing this handoff.")
    if (gate := gates.get("programmer_code_comments")) and gate.status != "pass":
        checklist.append("Inspect pending programmer code comments, then acknowledge the reviewed fingerprints.")
    if (gate := gates.get("patch_suggestions")) and gate.status != "pass":
        checklist.append("Inspect patch guidance and choose the integration path intentionally before editing source.")
    if (gate := gates.get("real_canaries")) and gate.status != "pass":
        checklist.append("Run real provider canaries with the enabled production providers.")
    if (gate := gates.get("eval_history")) and gate.status != "pass":
        checklist.append("Record project-relevant compare history before relying on automatic routing decisions.")
    if report.status == "ready":
        checklist.append("Record the final owner approval outside Crupier according to the project's normal release process.")
    return _dedupe(checklist)


def _pending_handoff_code_comments(report: AdoptionHandoffReport) -> list[CodeComment]:
    review = report.doctor.adoption_plan.code_comment_review
    if review:
        return list(review.pending)
    return list(report.doctor.adoption_plan.code_comments)


def format_adoption_plan_markdown(plan: ProjectAdoptionPlan) -> str:
    lines = [
        f"# Crupier Adoption Plan: {plan.project}",
        "",
        f"Generated: {plan.generated_at}",
        f"Recommended path: {plan.recommended_path}",
        f"Confidence: {plan.confidence}",
        f"Ready: {str(plan.ready).lower()}",
        "",
    ]
    if plan.blockers:
        lines.append("## Blockers")
        for blocker in plan.blockers:
            lines.append(f"- {blocker}")
        lines.append("")
    lines.append("## Options")
    for option in plan.options:
        lines.append(f"- {option.status.upper()} {option.path} score={option.score}: {option.summary}")
        for action in option.actions:
            lines.append(f"  - action: {action}")
        for risk in option.risks:
            lines.append(f"  - risk: {risk}")
    lines.extend(["", "## Checklist"])
    for item in plan.checklist:
        lines.append(f"- {item}")
    if plan.code_comments:
        lines.extend(["", "## Programmer Code Comments"])
        for comment in plan.code_comments:
            lines.append(f"- P{comment.priority} {comment.file}:{comment.line} {comment.title}")
    lines.append("")
    return "\n".join(lines)


def format_adoption_patch_markdown(report: AdoptionPatchReport) -> str:
    lines = [
        f"# Crupier Adoption Patch Suggestions: {report.project}",
        "",
        f"Generated: {report.generated_at}",
        f"Adoption path: {report.adoption_path}",
        f"Ready: {str(report.ready).lower()}",
        "",
    ]
    if report.blockers:
        lines.append("## Blockers")
        for blocker in report.blockers:
            lines.append(f"- {blocker}")
        lines.append("")
    lines.append("## Suggestions")
    for patch in report.patches:
        lines.append(f"### {patch.title}")
        lines.append("")
        lines.append(f"- Status: {patch.status}")
        lines.append(f"- Path: {patch.adoption_path}")
        lines.append(f"- Summary: {patch.summary}")
        for command in patch.commands:
            lines.append(f"- Command: `{command}`")
        for note in patch.notes:
            lines.append(f"- Note: {note}")
        if patch.files:
            lines.append(f"- Files: {', '.join(patch.files)}")
        if patch.diff:
            lines.append("")
            lines.append("```diff")
            lines.append(patch.diff.rstrip())
            lines.append("```")
        lines.append("")
    for warning in report.warnings:
        lines.append(f"- Warning: {warning}")
    lines.append("")
    return "\n".join(lines)


def format_project_audit_markdown(report: ProjectAuditReport) -> str:
    lines = [
        f"# Crupier Project Audit: {report.project}",
        "",
        f"Generated: {report.generated_at}",
        f"Status: {'ready' if report.ok else 'not ready'}",
        "",
        "## Checks",
    ]
    for check in report.checks:
        lines.append(f"- {check.status.upper()} [{check.severity}] {check.id}: {check.summary}")
        for action in check.actions:
            lines.append(f"  - action: {action}")
    lines.extend(["", "## Human Route Reviews"])
    for review in report.route_reviews:
        lines.append(f"- {review.status.upper()} {review.id}: {review.strategy or 'no-strategy'}")
        if review.models:
            lines.append(f"  - models: {', '.join(review.models)}")
        if review.reason:
            lines.append(f"  - reason: {review.reason}")
        for question in review.human_questions:
            lines.append(f"  - human check: {question}")
    if report.real_canaries:
        lines.extend(["", "## Real Canaries"])
        for canary in report.real_canaries:
            status = "PASS" if canary.get("ok") else "FAIL"
            lines.append(f"- {status} {canary.get('id')}")
    if report.code_comments:
        lines.extend(["", "## Programmer Code Comments"])
        for comment in report.code_comments:
            lines.append(f"- P{comment.priority} {comment.file}:{comment.line} {comment.title}")
            lines.append(f"  - {comment.body}")
    lines.append("")
    return "\n".join(lines)


def format_code_comments_markdown(comments: list[CodeComment]) -> str:
    lines = ["# Crupier Programmer Code Comments", ""]
    if not comments:
        lines.append("No AI integration hotspots found.")
        lines.append("")
        return "\n".join(lines)
    for comment in comments:
        lines.append(f"- P{comment.priority} {comment.file}:{comment.line} {comment.title}")
        lines.append(f"  - {comment.body}")
    lines.append("")
    return "\n".join(lines)


def format_code_review_comments_markdown(comments: list[CodeComment]) -> str:
    lines = [
        "# Crupier Code Review Comments",
        "",
        "Use these as PR/review comments for other programmers. They are metadata-only and do not include source snippets.",
        "",
    ]
    if not comments:
        lines.append("No AI integration hotspots found.")
        lines.append("")
        return "\n".join(lines)
    lines.extend(["## Review Checklist", ""])
    for comment in comments:
        lines.append(f"- [ ] P{comment.priority} `{comment.file}:{comment.line}` {comment.title}")
    lines.extend(["", "## Comments", ""])
    for comment in comments:
        lines.append(f"### `{comment.file}:{comment.line}` P{comment.priority} {comment.title}")
        lines.append("")
        lines.append(_code_review_comment_body(comment))
        lines.append("")
    return "\n".join(lines)


def format_code_comments_sarif(comments: list[CodeComment]) -> dict[str, Any]:
    rules: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for comment in comments:
        rule_id = _code_comment_rule_id(comment)
        rules.setdefault(
            rule_id,
            {
                "id": rule_id,
                "name": comment.title,
                "shortDescription": {"text": comment.title},
                "fullDescription": {"text": comment.body},
                "properties": {"category": comment.category, "priority": comment.priority},
            },
        )
        results.append(
            {
                "ruleId": rule_id,
                "level": _sarif_level(comment.priority),
                "message": {"text": _code_review_comment_body(comment)},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": comment.file},
                            "region": {"startLine": max(1, int(comment.line))},
                        }
                    }
                ],
                "partialFingerprints": {"crupierCodeComment": _code_comment_fingerprint(comment)},
                "properties": {"category": comment.category, "priority": comment.priority},
            }
        )
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Crupier",
                        "informationUri": "https://pypi.org/project/crupier/",
                        "rules": list(rules.values()),
                    }
                },
                "results": results,
            }
        ],
    }


def _code_comment_rule_id(comment: CodeComment) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", comment.title.lower()).strip("-") or "code-comment"
    return f"crupier.{comment.category}.{slug}"


def _sarif_level(priority: int) -> str:
    if priority <= 1:
        return "error"
    if priority == 2:
        return "warning"
    return "note"


def _code_review_comment_body(comment: CodeComment) -> str:
    return (
        f"**Crupier P{comment.priority}: {comment.title}**\n\n"
        f"{comment.body}\n\n"
        f"Category: `{comment.category}`. "
        "After review, run `crupier code comments --ack-reviewed` to acknowledge the current fingerprints."
    )


def _read_code_comment_review_records(root: Path) -> list[dict[str, Any]]:
    path = root / ".crupier" / "code-comments" / "reviews.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            records.append(data)
    return records


def _handoff_artifacts(root: Path) -> dict[str, list[str]]:
    return {
        "doctor_reports": _latest_paths(root / ".crupier" / "audits", "project_doctor_*.md"),
        "code_comment_reports": _latest_paths(root / ".crupier" / "audits", "code_comments_*.md"),
        "feedback_review_reports": _latest_paths(root / ".crupier" / "feedback" / "reviews", "human_review_*.md"),
        "feedback_decision_templates": _latest_paths(
            root / ".crupier" / "feedback" / "decisions",
            "human_decisions_*.json",
        ),
        "compare_reports": _latest_paths(root / ".crupier" / "evals" / "runs", "compare*.json"),
        "compare_history": _latest_paths(root / ".crupier" / "evals" / "history", "compare_runs.jsonl"),
        "code_comment_review_log": _latest_paths(root / ".crupier" / "code-comments", "reviews.jsonl"),
        "code_review_comment_packets": _latest_paths(root / ".crupier" / "code-comments", "review_comments_*.md"),
        "code_comment_sarif": _latest_paths(root / ".crupier" / "code-comments", "code_comments_*.sarif"),
        "code_comment_decision_templates": _latest_paths(
            root / ".crupier" / "code-comments" / "decisions",
            "code_comment_decisions_*.json",
        ),
        "adoption_signoffs": _latest_paths(root / ".crupier" / "handoffs", "signoffs.jsonl"),
    }


def _latest_paths(root: Path, pattern: str, *, limit: int = 5) -> list[str]:
    if not root.exists():
        return []
    paths = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return [str(path) for path in paths[:limit]]


def _handoff_actions(
    doctor: ProjectDoctorReport,
    artifacts: dict[str, list[str]],
    *,
    paths: list[str | Path] | None,
) -> tuple[list[str], list[str]]:
    actions: list[str] = []
    commands: list[str] = []
    gates = {gate.id: gate for gate in doctor.gates}
    feedback_gate = gates.get("human_feedback")
    if feedback_gate and feedback_gate.status != "pass":
        production_template = _latest_production_decision_template(artifacts.get("feedback_decision_templates", []))
        if production_template:
            actions.append("Fill the latest human decision template with verdicts and import the accepted/rejected variants.")
            commands.append(f"crupier feedback import-decisions --decisions {production_template} --apply-to-registry")
        elif artifacts.get("feedback_decision_templates"):
            actions.append(
                "Existing human decision templates came from dry-run reports; create a real compare before production feedback."
            )
            commands.append(
                "crupier eval compare-dataset --dataset examples/model-compare-eval.json --no-dry-run "
                "--record-history --write-report"
            )
        elif artifacts.get("feedback_review_reports"):
            actions.append("Review the latest feedback review packet and record a human verdict for accepted/rejected variants.")
            if artifacts.get("compare_reports"):
                commands.append(
                    f"crupier feedback review --compare-report {artifacts['compare_reports'][0]} "
                    "--write-report --write-decisions-template"
                )
        else:
            actions.append("Generate a feedback review packet from a compare report, then record a human verdict.")
            if artifacts.get("compare_reports"):
                commands.append(
                    f"crupier feedback review --compare-report {artifacts['compare_reports'][0]} "
                    "--write-report --write-decisions-template"
                )
            else:
                commands.append(
                    "crupier eval compare-dataset --dataset examples/model-compare-eval.json --record-history --write-report"
                )
        if not production_template and not artifacts.get("feedback_decision_templates"):
            commands.append("crupier feedback apply")

    signoff_gate = gates.get("adoption_signoff")
    if signoff_gate and signoff_gate.status != "pass":
        actions.append("Record project-level adoption signoff after reviewing the handoff.")
        commands.append('crupier adopt signoff --verdict approve --note "handoff reviewed"')
        commands.append('crupier adopt signoff --verdict reject --note "reason for rejection"')

    code_gate = gates.get("programmer_code_comments")
    if code_gate and code_gate.status != "pass":
        actions.append("Review programmer code comments and acknowledge the current set after inspection.")
        path_args = " ".join(str(path) for path in paths or [])
        command = "crupier code comments"
        if path_args:
            command += f" {path_args}"
        decision_templates = artifacts.get("code_comment_decision_templates", [])
        if decision_templates:
            commands.append(f"{command} --import-decisions {decision_templates[0]}")
        commands.append(f"{command} --write-report --write-review-comments --write-decisions-template")
        commands.append(f"{command} --ack-reviewed")

    history_gate = gates.get("eval_history")
    if history_gate and history_gate.status != "pass":
        actions.append("Record more A/B compare history until project-relevant model scores are stable enough.")
        commands.append("crupier eval compare-dataset --dataset examples/model-compare-eval.json --record-history --write-report")

    canary_gate = gates.get("real_canaries")
    if canary_gate and canary_gate.status != "pass":
        actions.append("Run real provider canaries before production adoption.")
        commands.append("crupier adopt doctor --production --real --provider anthropic --provider ollama")

    blocker_gate = gates.get("adoption_blockers")
    if blocker_gate and blocker_gate.status != "pass":
        actions.append("Resolve adoption blockers before changing project routing behavior.")
        commands.append("crupier adopt plan --write-report")

    patch_gate = gates.get("patch_suggestions")
    if patch_gate and patch_gate.status != "pass":
        actions.append("Inspect manual patch guidance and decide which integration path to apply.")
        commands.append("crupier adopt patches --path recommended --write-report")

    return _dedupe(actions), _dedupe(commands)


def _latest_production_decision_template(paths: list[str]) -> str | None:
    for path in paths:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and not bool(data.get("source_dry_run", True)):
            return path
    return None


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


def _code_comment_fingerprint(comment: CodeComment) -> str:
    payload = json.dumps(
        {
            "file": comment.file,
            "line": comment.line,
            "title": comment.title,
            "body": comment.body,
            "priority": comment.priority,
            "category": comment.category,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _comment_counts(comments: list[CodeComment]) -> dict[str, Any]:
    by_title: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for comment in comments:
        by_title[comment.title] = by_title.get(comment.title, 0) + 1
        by_category[comment.category] = by_category.get(comment.category, 0) + 1
    provider_titles = {
        "openai": "OpenAI integration point",
        "anthropic": "Anthropic integration point",
        "ollama": "Ollama integration point",
        "google": "Google/Gemini integration point",
    }
    providers = {name: by_title.get(title, 0) for name, title in provider_titles.items()}
    return {
        "total": len(comments),
        "by_title": by_title,
        "by_category": by_category,
        "providers": providers,
        "provider_count": sum(1 for count in providers.values() if count),
        "drop_in": by_category.get("drop_in", 0),
        "hardcoded_models": by_title.get("Hard-coded model choice", 0),
        "inline_credentials": by_title.get("Possible inline credential", 0),
        "credential_fixtures": by_title.get("Credential-like test fixture", 0),
    }


def _framework_hints(root: Path) -> dict[str, Any]:
    files = {
        "pyproject": root / "pyproject.toml",
        "requirements": root / "requirements.txt",
        "setup_py": root / "setup.py",
        "package_json": root / "package.json",
    }
    has_python = any(files[name].exists() for name in ["pyproject", "requirements", "setup_py"])
    has_node = files["package_json"].exists()
    package_names: list[str] = []
    if files["package_json"].exists():
        try:
            data = json.loads(files["package_json"].read_text(encoding="utf-8"))
            deps = {**dict(data.get("dependencies", {})), **dict(data.get("devDependencies", {}))}
            package_names = sorted(deps)
        except (OSError, json.JSONDecodeError, TypeError):
            package_names = []
    python_text = ""
    for name in ["pyproject", "requirements", "setup_py"]:
        path = files[name]
        if path.exists():
            try:
                python_text += "\n" + path.read_text(encoding="utf-8", errors="ignore")[:20_000].lower()
            except OSError:
                pass
    node_names = {name.lower() for name in package_names}
    frameworks: list[str] = []
    if "fastapi" in python_text:
        frameworks.append("fastapi")
    if "flask" in python_text:
        frameworks.append("flask")
    if "django" in python_text:
        frameworks.append("django")
    if "next" in node_names or "next.js" in node_names:
        frameworks.append("nextjs")
    if "express" in node_names:
        frameworks.append("express")
    return {
        "python": has_python,
        "node": has_node,
        "frameworks": sorted(set(frameworks)),
        "package_managers": {
            "pyproject": files["pyproject"].exists(),
            "requirements": files["requirements"].exists(),
            "package_json": has_node,
        },
    }


def _adoption_blockers(comments: list[CodeComment]) -> list[str]:
    blockers: list[str] = []
    inline_credentials = [
        comment
        for comment in comments
        if comment.title == "Possible inline credential" and not _is_test_fixture_path(comment.file)
    ]
    if inline_credentials:
        blockers.append(
            f"Move {len(inline_credentials)} possible inline credential(s) out of source before adopting Crupier."
        )
    return blockers


def _is_test_fixture_path(path: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name
    return any(part in {"test", "tests", "__tests__", "spec", "fixtures"} for part in parts) or name.startswith(
        ("test_", "spec_")
    )


def _adoption_options(counts: dict[str, Any], hints: dict[str, Any], *, blocked: bool) -> list[AdoptionOption]:
    providers = counts["providers"]
    openai_count = int(providers.get("openai", 0))
    drop_in = int(counts.get("drop_in", 0))
    hardcoded = int(counts.get("hardcoded_models", 0))
    provider_count = int(counts.get("provider_count", 0))
    python = bool(hints.get("python"))
    node = bool(hints.get("node"))
    blocked_status = "blocked" if blocked else None

    proxy_score = 35 + (30 if openai_count else 0) + (15 if drop_in >= 4 else 0) + (10 if node else 0)
    compat_score = 25 + (40 if python and openai_count else 0) + (10 if 0 < drop_in <= 3 else 0)
    autopatch_score = 15 + (35 if python and openai_count else 0)
    native_score = 55 + (15 if provider_count >= 2 else 0) + (10 if hardcoded else 0) + (10 if drop_in == 0 else 0)

    options = [
        AdoptionOption(
            path="proxy",
            status=blocked_status or ("viable" if openai_count else "not_applicable"),
            score=_clamp_score(proxy_score if openai_count else 20),
            summary="Use `crupier serve` as an OpenAI-compatible base URL for projects that already call OpenAI-like APIs.",
            actions=[
                "Run `crupier serve --port 8787` in a controlled environment.",
                "Point OpenAI-compatible clients at `OPENAI_BASE_URL=http://127.0.0.1:8787/v1`.",
                "Start with dry-run or `compat-mode=strict`, then move to balanced routing after evals pass.",
            ],
            risks=[
                "Provider-specific SDK features may need pass-through or native SDK migration.",
                "Streaming/tools/file behavior must be checked against the project's exact usage.",
            ],
            evidence={"openai_call_sites": openai_count, "node_project": node, "drop_in_comments": drop_in},
        ),
        AdoptionOption(
            path="compat_client",
            status=blocked_status or ("viable" if python and openai_count else "not_applicable"),
            score=_clamp_score(compat_score),
            summary="Replace Python OpenAI client imports with `crupier.compat.openai.OpenAI` for small import-level changes.",
            actions=[
                "Replace `from openai import OpenAI` with `from crupier.compat.openai import OpenAI` where appropriate.",
                "Run existing tests plus `crupier eval compare-dataset` before enabling real provider execution.",
            ],
            risks=["Only implemented OpenAI-like response shapes are safe; inspect unsupported SDK surface first."],
            evidence={"python_project": python, "openai_call_sites": openai_count},
        ),
        AdoptionOption(
            path="autopatch",
            status=blocked_status or ("experimental" if python and openai_count else "not_applicable"),
            score=_clamp_score(autopatch_score),
            summary="Use `crupier.install('openai')` for test harnesses or experiments where minimal code edits matter.",
            actions=[
                "Enable only in a test entrypoint or controlled bootstrap.",
                "Record route traces and human feedback before considering production use.",
            ],
            risks=["Global monkeypatching can surprise maintainers; prefer explicit compat client or proxy for production."],
            evidence={"python_project": python, "openai_call_sites": openai_count},
        ),
        AdoptionOption(
            path="native_sdk",
            status=blocked_status or "viable",
            score=_clamp_score(native_score),
            summary="Use `Crupier.from_project(...).deal(...)` for deeper agent, tool, multimodal, or multi-provider integration.",
            actions=[
                "Create a small adapter around the project's AI boundary that calls `Crupier.deal`.",
                "Move hard-coded model choices into `[models].allow`, profiles, constraints, and eval datasets.",
                "Use `crupier audit --real` and `crupier eval compare-dataset --record-history` before rollout.",
            ],
            risks=["Requires more intentional refactoring than proxy or compat-client adoption."],
            evidence={"provider_count": provider_count, "hardcoded_models": hardcoded, "drop_in_comments": drop_in},
        ),
    ]
    recommended = _recommended_option(options)
    return [
        AdoptionOption(
            path=option.path,
            status="recommended" if not blocked and option.path == recommended.path else option.status,
            score=option.score,
            summary=option.summary,
            actions=option.actions,
            risks=option.risks,
            evidence=option.evidence,
        )
        for option in sorted(options, key=lambda item: item.score, reverse=True)
    ]


def _recommended_option(options: list[AdoptionOption]) -> AdoptionOption:
    viable = [option for option in options if option.status in {"viable", "recommended", "experimental"}]
    if not viable:
        return max(options, key=lambda option: option.score)
    return max(viable, key=lambda option: option.score)


def _adoption_checklist(path: str, counts: dict[str, Any], *, blocked: bool) -> list[str]:
    checklist: list[str] = []
    if blocked:
        checklist.append("Remove inline credentials and re-run `crupier adopt plan`.")
    checklist.extend(
        [
            "Run `crupier update --online` and keep explicit allowlisted models.",
            "Run `crupier verify` for every enabled provider.",
            "Run `crupier audit --real` before production rollout.",
            "Create or reuse a project eval dataset and run `crupier eval compare-dataset --record-history`.",
            "Use `crupier feedback record` when a technically passing route is not acceptable to a human reviewer.",
        ]
    )
    if path == "proxy":
        checklist.insert(1, "Start `crupier serve` and point an OpenAI-compatible base URL at it.")
    elif path == "compat_client":
        checklist.insert(1, "Replace narrow Python OpenAI client imports with `crupier.compat.openai.OpenAI`.")
    elif path == "autopatch":
        checklist.insert(1, "Enable `crupier.install('openai')` only in a controlled test bootstrap.")
    elif path == "native_sdk":
        checklist.insert(1, "Wrap the project's AI boundary with `Crupier.from_project(...).deal(...)`.")
    if counts.get("hardcoded_models"):
        checklist.append("Review hard-coded model comments and move routing choices into Crupier config/profiles.")
    return checklist


def _adoption_warnings(counts: dict[str, Any], hints: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if counts.get("provider_count", 0) > 1:
        warnings.append("Mixed provider usage detected; native SDK adoption may be cleaner than one compatibility shim.")
    if not counts.get("drop_in"):
        warnings.append("No existing AI SDK call sites were found; start with native SDK integration.")
    if hints.get("node") and hints.get("python"):
        warnings.append("Both Python and Node project files detected; adoption may need separate app-boundary choices.")
    if counts.get("credential_fixtures"):
        warnings.append("Credential-like test fixtures detected; confirm they are synthetic before rollout.")
    return warnings


def _confidence_from_score(score: int) -> str:
    if score >= 75:
        return "high"
    if score >= 55:
        return "medium"
    return "low"


def _clamp_score(score: int) -> int:
    return max(0, min(100, int(score)))


def _compat_client_patch_suggestions(
    root: Path,
    *,
    paths: list[str | Path] | None,
    max_files: int,
) -> list[AdoptionPatchSuggestion]:
    patches: list[AdoptionPatchSuggestion] = []
    for path in _iter_source_files(root, paths=paths, max_files=max_files, max_file_size=250_000):
        if path.suffix != ".py":
            continue
        try:
            original = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        updated = _compat_client_python_rewrite(original)
        if updated == original:
            continue
        rel = _relative_path(root, path)
        diff = "".join(
            unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=rel,
                tofile=rel,
            )
        )
        patches.append(
            AdoptionPatchSuggestion(
                adoption_path="compat_client",
                title=f"Use Crupier OpenAI-compatible client in {rel}",
                status="suggested",
                summary="Replace a narrow Python OpenAI client import with the Crupier compatibility client.",
                diff=diff,
                commands=[
                    "crupier eval run",
                    "crupier audit --real",
                ],
                notes=[
                    "Review call sites that use OpenAI SDK surfaces beyond responses/chat/embeddings before applying.",
                    "This is a suggested diff only; Crupier did not modify the file.",
                ],
                files=[rel],
            )
        )
    return patches


def _compat_client_python_rewrite(text: str) -> str:
    lines = text.splitlines(keepends=True)
    rewritten: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "from openai import OpenAI":
            rewritten.append(line.replace("from openai import OpenAI", "from crupier.compat.openai import OpenAI"))
            continue
        if stripped == "from openai import OpenAI, AsyncOpenAI":
            rewritten.append(
                line.replace(
                    "from openai import OpenAI, AsyncOpenAI",
                    "from crupier.compat.openai import OpenAI\n# TODO: review AsyncOpenAI usage before routing through Crupier.\n",
                )
            )
            continue
        rewritten.append(line)
    return "".join(rewritten)


def _proxy_patch_suggestions() -> list[AdoptionPatchSuggestion]:
    return [
        AdoptionPatchSuggestion(
            adoption_path="proxy",
            title="Run Crupier as an OpenAI-compatible proxy",
            status="manual",
            summary="No source patch is required when the project can configure an OpenAI-compatible base URL.",
            commands=[
                "crupier serve --port 8787 --compat openai",
                "export OPENAI_BASE_URL=http://127.0.0.1:8787/v1",
                "crupier eval compare-dataset --dataset examples/model-compare-eval.json --record-history",
            ],
            notes=[
                "Use strict compatibility first if the project depends on exact model selection.",
                "Add `--no-dry-run` only after `crupier verify` and `crupier audit --real` pass.",
            ],
        )
    ]


def _autopatch_patch_suggestions() -> list[AdoptionPatchSuggestion]:
    diff = """--- /dev/null
+++ crupier_bootstrap.py
@@
+import crupier
+
+# Experimental: enable only in controlled tests or local adoption probes.
+crupier.install("openai")
"""
    return [
        AdoptionPatchSuggestion(
            adoption_path="autopatch",
            title="Add an explicit Crupier autopatch bootstrap",
            status="manual",
            summary="Create a small bootstrap module for controlled monkeypatch experiments.",
            diff=diff,
            commands=[
                "python -c 'import crupier_bootstrap; import openai; print(openai.OpenAI)'",
                "crupier audit --real",
            ],
            notes=[
                "Autopatch is not the preferred production route.",
                "Import the bootstrap before importing OpenAI clients in the test process.",
            ],
            files=["crupier_bootstrap.py"],
        )
    ]


def _native_sdk_patch_suggestions() -> list[AdoptionPatchSuggestion]:
    diff = '''--- /dev/null
+++ crupier_adapter.py
@@
+from crupier import Crupier
+
+
+_crupier = Crupier.from_project(".")
+
+
+def route_ai(task, *, input=None, mode="agentic", constraints=None, **kwargs):
+    """Project AI boundary routed through Crupier."""
+    return _crupier.deal(
+        task=task,
+        input=input,
+        mode=mode,
+        constraints=constraints or {},
+        trace="summary",
+        **kwargs,
+    )
'''
    return [
        AdoptionPatchSuggestion(
            adoption_path="native_sdk",
            title="Add a native Crupier AI boundary module",
            status="manual",
            summary="Create a small project adapter and migrate AI call sites into it deliberately.",
            diff=diff,
            commands=[
                "crupier adopt plan",
                "crupier eval compare-dataset --dataset examples/model-compare-eval.json --record-history",
                "crupier feedback summary",
            ],
            notes=[
                "Use this path for agents, tools, structured output, multimodal files, or mixed providers.",
                "Keep the adapter boundary small so maintainers can review behavior changes.",
            ],
            files=["crupier_adapter.py"],
        )
    ]


def _iter_source_files(
    root: Path,
    *,
    paths: list[str | Path] | None,
    max_files: int,
    max_file_size: int,
):
    candidates = [root / Path(path) for path in paths] if paths else [root]
    yielded = 0
    for candidate in candidates:
        if not candidate.exists():
            continue
        iterable = [candidate] if candidate.is_file() else candidate.rglob("*")
        for path in iterable:
            if yielded >= max_files:
                return
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            try:
                relative_parts = path.resolve().relative_to(root).parts
            except ValueError:
                relative_parts = path.resolve().parts
            if any(_skip_source_path_part(part) for part in relative_parts):
                continue
            try:
                if path.stat().st_size > max_file_size:
                    continue
            except OSError:
                continue
            yielded += 1
            yield path


def _skip_source_path_part(part: str) -> bool:
    return part in SKIP_DIRS or part.endswith(".egg-info")


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _eval_check(report: Any) -> AuditCheck:
    return AuditCheck(
        id="routing_evals",
        status="pass" if report.ok else "fail",
        severity="high",
        summary=f"Routing evals passed {report.passed}/{report.total}."
        if report.ok
        else f"Routing evals failed {report.failed}/{report.total}.",
        evidence=report.to_dict(),
        actions=["Fix failing project routing evals before production use."] if not report.ok else [],
    )


def _route_review_check(reviews: list[RouteReview]) -> AuditCheck:
    failures = [review.id for review in reviews if review.status == "fail"]
    warnings = [review.id for review in reviews if review.status == "warn"]
    status = "fail" if failures else ("warn" if warnings else "pass")
    return AuditCheck(
        id="human_route_reviews",
        status=status,
        severity="high" if failures else "medium",
        summary="Human route review previews are explainable."
        if status == "pass"
        else "Some human route review previews need attention.",
        evidence={"failures": failures, "warnings": warnings},
        actions=["Inspect route_reviews and adjust profiles/evals/allowlist."] if status != "pass" else [],
    )


def _real_canary_check(canaries: list[dict[str, Any]]) -> AuditCheck:
    failures = [item["id"] for item in canaries if not item.get("ok")]
    return AuditCheck(
        id="real_canaries",
        status="pass" if not failures and canaries else "fail",
        severity="high",
        summary="Real provider canaries passed." if not failures and canaries else "Real provider canaries failed.",
        evidence={"failures": failures, "count": len(canaries)},
        actions=["Inspect failed canaries before using Crupier in a real project."] if failures or not canaries else [],
    )


def _code_comment_check(comments: list[CodeComment]) -> AuditCheck:
    high_priority = [comment.to_dict() for comment in comments if comment.priority <= 1]
    return AuditCheck(
        id="programmer_code_comments",
        status="warn" if comments else "pass",
        severity="medium",
        summary=f"Generated {len(comments)} programmer code comment(s)."
        if comments
        else "No AI integration hotspots found in scanned source files.",
        evidence={"count": len(comments), "high_priority": high_priority[:20]},
        actions=["Review code_comments and decide proxy/client/autopatch/native SDK path."] if comments else [],
    )


def _model_refs_for_provider(client: Any, provider: str, *, all_models: bool) -> list[str]:
    refs: list[str] = []
    for model in client.config.models.allow:
        ref = ModelRef.parse(model)
        if ref.provider == provider:
            refs.append(ref.key)
            if not all_models:
                break
    return refs


def _first_per_provider(models: list[str]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for model in models:
        provider = ModelRef.parse(model).provider
        if provider in seen:
            continue
        selected.append(model)
        seen.add(provider)
    return selected


def _prefer_provider(models: list[str], provider: str) -> str | None:
    return next((model for model in models if ModelRef.parse(model).provider == provider), None)


def _solid_png_rgb(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    row = b"\x00" + bytes(rgb) * width
    raw = b"".join(row for _ in range(height))
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(raw)),
            chunk(b"IEND", b""),
        ]
    )


def _provider_env_status(settings: Any, provider: str) -> dict[str, Any]:
    env_key = getattr(settings, "env_key", None) or DEFAULT_PROVIDER_ENV_KEYS.get(provider)
    host = getattr(settings, "host", None)
    required = provider in {"openai", "anthropic", "google", "openrouter"} or (
        provider == "ollama" and _ollama_cloud_host(host)
    )
    if provider == "google":
        return {
            "key": google_env_label(settings),
            "required": required,
            "present": google_env_present(settings),
            "host": host,
        }
    return {
        "key": env_key,
        "required": required,
        "present": bool(env_key and os.environ.get(env_key)),
        "host": host,
    }


def _ollama_cloud_host(host: str | None) -> bool:
    if not host:
        return False
    lowered = host.lower()
    return "ollama.com" in lowered and not lowered.startswith("http://localhost") and not lowered.startswith(
        "http://127.0.0.1"
    )


def _canary_error(canary_id: str, kind: str, model_ref: str, exc: Exception) -> dict[str, Any]:
    return {
        "id": canary_id,
        "kind": kind,
        "ok": False,
        "model": model_ref,
        "error_type": exc.__class__.__name__,
        "error": _redact_secrets(str(exc)),
    }


_SECRET_REPLACERS = (
    (re.compile(("s" + "k-") + r"[A-Za-z0-9_\-]{10,}"), "[redacted]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{12,}", re.IGNORECASE), r"\1[redacted]"),
    (re.compile(r"([A-Z][A-Z0-9_]*_API_KEY=)[^\s]+"), r"\1[redacted]"),
)


def _redact_secrets(message: str) -> str:
    redacted = message
    for pattern, replacement in _SECRET_REPLACERS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def ensure_audit_ok(report: ProjectAuditReport) -> None:
    if not report.ok:
        raise CrupierError("Project audit is not ready. Inspect report.checks for failures.")
