"""Crupier command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from .client import Crupier
from .config import CrupierConfig, write_default_project, write_models_allow
from .evals import CompareVariant
from .errors import CrupierConfigError, CrupierError
from .feedback import (
    build_human_review_packet,
    import_human_decisions,
    write_human_decision_template,
    write_human_review_packet,
)
from .models import ModelRef
from .probes import AVAILABLE_PROBES
from .adapters.google import google_env_label, google_env_present
from .release import ReleaseCheck, check_pypi_project_name, run_release_checks
from .project_audit import (
    acknowledge_code_comments,
    build_adoption_handoff,
    build_adoption_handoff_from_doctor,
    build_adoption_patches,
    build_adoption_plan,
    build_config_free_adoption_handoff,
    build_config_free_project_doctor,
    build_project_doctor,
    import_code_comment_decisions,
    record_adoption_signoff,
    scan_code_comments,
    summarize_code_comment_reviews,
    write_adoption_patch_report,
    write_adoption_handoff_report,
    write_adoption_package_index,
    write_adoption_plan_report,
    write_code_comment_decision_template,
    write_code_review_comments,
    write_code_comments_sarif,
    write_code_comments_report,
    write_project_doctor_report,
)
from .server import build_openai_compatible_server

REAL_PROVIDER_CHOICES = ("openai", "anthropic", "google", "ollama")
DEFAULT_PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CrupierError as exc:
        print(f"crupier: error: {exc}", file=sys.stderr)
        hint = getattr(exc, "hint", None)
        if hint:
            print(f"hint: {hint}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crupier")
    parser.add_argument("--project", default=".", help="Project directory containing crupier.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create crupier.toml and .crupier directories")
    init.add_argument("--force", action="store_true", help="Overwrite existing crupier.toml")
    init.set_defaults(func=cmd_init)

    update = subparsers.add_parser("update", help="Refresh capability cards")
    update.add_argument("--dry-run", action="store_true", help="Preview changes without writing files")
    update.add_argument("--online", action="store_true", help="Discover current models from enabled providers")
    update.add_argument("--provider", choices=REAL_PROVIDER_CHOICES, help="Only update one provider")
    update.add_argument("--json", action="store_true", help="Print JSON")
    update.set_defaults(func=cmd_update)

    models = subparsers.add_parser("models", help="Model registry commands")
    model_subparsers = models.add_subparsers(dest="models_command", required=True)
    models_list = model_subparsers.add_parser("list", help="List known models")
    models_list.add_argument("--all", action="store_true", help="Show built-in models beyond the project allowlist")
    models_list.add_argument("--json", action="store_true", help="Print JSON")
    models_list.set_defaults(func=cmd_models_list)
    models_discover = model_subparsers.add_parser("discover", help="List models from enabled real providers")
    models_discover.add_argument("--provider", choices=REAL_PROVIDER_CHOICES, help="Provider to query")
    models_discover.add_argument("--json", action="store_true", help="Print JSON")
    models_discover.set_defaults(func=cmd_models_discover)
    models_allow = model_subparsers.add_parser("allow", help="Add or replace allowed models in crupier.toml")
    models_allow.add_argument("models", nargs="+", help="Model refs, e.g. openai:gpt-5.5 anthropic:claude-opus-4-8")
    models_allow.add_argument("--replace", action="store_true", help="Replace the allow list instead of appending")
    models_allow.set_defaults(func=cmd_models_allow)

    registry = subparsers.add_parser("registry", help="Registry commands")
    registry_subparsers = registry.add_subparsers(dest="registry_command", required=True)
    registry_snapshot = registry_subparsers.add_parser("snapshot", help="Registry snapshot commands")
    snapshot_subparsers = registry_snapshot.add_subparsers(dest="snapshot_command", required=True)
    snapshot_create = snapshot_subparsers.add_parser("create", help="Create a registry snapshot")
    snapshot_create.add_argument("name", nargs="?", help="Snapshot name; defaults to a UTC reg_* timestamp")
    snapshot_create.add_argument("--allowed-only", action="store_true", help="Snapshot only models in [models].allow")
    snapshot_create.add_argument("--json", action="store_true", help="Print JSON")
    snapshot_create.set_defaults(func=cmd_registry_snapshot_create)
    snapshot_list = snapshot_subparsers.add_parser("list", help="List registry snapshots")
    snapshot_list.add_argument("--json", action="store_true", help="Print JSON")
    snapshot_list.set_defaults(func=cmd_registry_snapshot_list)
    snapshot_diff = snapshot_subparsers.add_parser("diff", help="Compare snapshots or a snapshot against current")
    snapshot_diff.add_argument("left", help="Left snapshot name")
    snapshot_diff.add_argument("right", nargs="?", default="current", help="Right snapshot name, or 'current'")
    snapshot_diff.add_argument("--json", action="store_true", help="Print JSON")
    snapshot_diff.set_defaults(func=cmd_registry_snapshot_diff)
    snapshot_use = snapshot_subparsers.add_parser("use", help="Restore local registry cards from a snapshot")
    snapshot_use.add_argument("name", help="Snapshot name")
    snapshot_use.add_argument("--restore-allowlist", action="store_true", help="Replace [models].allow with snapshot allowlist")
    snapshot_use.add_argument("--json", action="store_true", help="Print JSON")
    snapshot_use.set_defaults(func=cmd_registry_snapshot_use)

    capabilities = subparsers.add_parser("capabilities", help="Capability verification commands")
    capability_subparsers = capabilities.add_subparsers(dest="capabilities_command", required=True)
    capability_probe = capability_subparsers.add_parser("probe", help="Probe model capabilities with real adapters")
    capability_probe.add_argument("--provider", choices=REAL_PROVIDER_CHOICES, help="Only probe one provider")
    capability_probe.add_argument("--model", action="append", help="Exact model ref to probe; can be passed multiple times")
    capability_probe.add_argument("--all", action="store_true", help="Probe all known models instead of the allowlist")
    capability_probe.add_argument("--probe", action="append", choices=AVAILABLE_PROBES, help="Probe case to run")
    capability_probe.add_argument("--apply", action="store_true", help="Persist probe results into capability cards")
    capability_probe.add_argument("--dry-run", action="store_true", help="Plan probes without provider calls")
    capability_probe.add_argument("--json", action="store_true", help="Print JSON")
    capability_probe.set_defaults(func=cmd_capabilities_probe)
    capability_readiness = capability_subparsers.add_parser("readiness", help="Check production readiness for model capabilities")
    capability_readiness.add_argument("--provider", choices=REAL_PROVIDER_CHOICES, help="Only check one provider")
    capability_readiness.add_argument("--model", action="append", help="Exact model ref to check; can be passed multiple times")
    capability_readiness.add_argument("--all", action="store_true", help="Check all known models instead of the allowlist")
    capability_readiness.add_argument("--strict", action="store_true", help="Require every known capability probe to be verified")
    capability_readiness.add_argument("--json", action="store_true", help="Print JSON")
    capability_readiness.set_defaults(func=cmd_capabilities_readiness)

    profiles = subparsers.add_parser("profiles", help="Profile commands")
    profile_subparsers = profiles.add_subparsers(dest="profiles_command", required=True)
    profiles_list = profile_subparsers.add_parser("list", help="List configured route profiles")
    profiles_list.add_argument("--json", action="store_true", help="Print JSON")
    profiles_list.set_defaults(func=cmd_profiles_list)

    release = subparsers.add_parser("release", help="Release readiness commands")
    release_subparsers = release.add_subparsers(dest="release_command", required=True)
    release_check = release_subparsers.add_parser("check", help="Check local package release readiness")
    release_check.add_argument("--skip-build", action="store_true", help="Skip distribution build and install smoke validations")
    release_check.add_argument(
        "--strict-public",
        action="store_true",
        help="Fail if any release warning remains or build/install smoke checks are skipped",
    )
    release_check.add_argument(
        "--verify-providers",
        action="store_true",
        help="Require real provider/model readiness checks using configured environment keys",
    )
    release_check.add_argument(
        "--provider",
        action="append",
        choices=REAL_PROVIDER_CHOICES,
        help="Provider to verify for release readiness; can be passed multiple times",
    )
    release_check.add_argument(
        "--no-openai-baseline",
        action="store_true",
        help="Do not add OpenAI as a baseline provider when verifying providers",
    )
    release_check.add_argument(
        "--skip-provider-smoke",
        action="store_true",
        help="Do not execute real model smoke calls during provider verification",
    )
    release_check.add_argument(
        "--verify-all-models",
        action="store_true",
        help="Verify all allowed models per provider instead of one representative model",
    )
    release_check.add_argument(
        "--check-pypi-name",
        action="store_true",
        help="Check whether the configured PyPI project name is available or already claimed",
    )
    release_check.add_argument(
        "--allow-existing-pypi-project",
        action="store_true",
        help="Treat an existing PyPI project name as allowed when you already own the project",
    )
    release_check.add_argument("--json", action="store_true", help="Print JSON")
    release_check.set_defaults(func=cmd_release_check)

    eval_parser = subparsers.add_parser("eval", help="Run routing evals")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_subparsers.add_parser("run", help="Run routing eval cases")
    eval_run.add_argument("--dataset", help="JSON/JSONL eval dataset; defaults to built-in routing evals")
    eval_run.add_argument(
        "--orchestrator-mode",
        choices=["deterministic", "model", "hybrid"],
        help="Override [orchestrator].mode for this run. model/hybrid may call the orchestrator model.",
    )
    eval_run.add_argument("--write-report", action="store_true", help="Write JSON report under .crupier/evals/runs")
    eval_run.add_argument("--json", action="store_true", help="Print JSON")
    eval_run.set_defaults(func=cmd_eval_run)
    eval_compare = eval_subparsers.add_parser("compare", help="Compare route/model variants for one task")
    eval_compare.add_argument("task", help="Task to compare")
    eval_compare.add_argument("--input", dest="input_value", help="Optional input payload; JSON is parsed when possible")
    eval_compare.add_argument("--mode", help="Base mode/profile")
    eval_compare.add_argument("--strategy", help="Base strategy")
    eval_compare.add_argument("--model", action="append", help="Force one model variant; can be passed multiple times")
    eval_compare.add_argument(
        "--variant",
        action="append",
        help='Variant JSON, e.g. {"name":"cheap","mode":"cheap","constraints":{"max_cost_usd":0.01}}',
    )
    eval_compare.add_argument("--max-cost-usd", type=float, help="Hard budget for every variant")
    eval_compare.add_argument("--max-output-tokens", type=int, help="Provider output token cap")
    eval_compare.add_argument("--response-schema", help="JSON Schema object for structured output")
    eval_compare.add_argument("--expect-contains", action="append", help="Deterministic output substring check")
    eval_compare.add_argument("--no-dry-run", action="store_true", help="Attempt real provider execution")
    eval_compare.add_argument("--write-report", action="store_true", help="Write JSON report under .crupier/evals/runs")
    eval_compare.add_argument("--json", action="store_true", help="Print JSON")
    eval_compare.set_defaults(func=cmd_eval_compare)
    eval_compare_dataset = eval_subparsers.add_parser(
        "compare-dataset",
        help="Compare route/model variants across an eval dataset",
    )
    eval_compare_dataset.add_argument("--dataset", help="JSON/JSONL eval dataset; defaults to built-in routing evals")
    eval_compare_dataset.add_argument("--model", action="append", help="Force one model variant; can be passed multiple times")
    eval_compare_dataset.add_argument(
        "--variant",
        action="append",
        help='Variant JSON, e.g. {"name":"cheap","mode":"cheap","constraints":{"max_cost_usd":0.01}}',
    )
    eval_compare_dataset.add_argument("--max-cost-usd", type=float, help="Hard budget for every variant")
    eval_compare_dataset.add_argument("--max-output-tokens", type=int, help="Provider output token cap")
    eval_compare_dataset.add_argument("--expect-contains", action="append", help="Deterministic output substring check")
    eval_compare_dataset.add_argument("--no-dry-run", action="store_true", help="Attempt real provider execution")
    eval_compare_dataset.add_argument("--apply", action="store_true", help="Write aggregate eval scores into capability cards")
    eval_compare_dataset.add_argument("--min-count", type=int, default=3, help="Minimum appearances before applying a score")
    eval_compare_dataset.add_argument(
        "--min-confidence",
        choices=["low", "medium", "high"],
        default="medium",
        help="Minimum confidence before applying a score",
    )
    eval_compare_dataset.add_argument(
        "--record-history",
        action="store_true",
        help="Append metadata-only aggregate results under .crupier/evals/history",
    )
    eval_compare_dataset.add_argument("--write-report", action="store_true", help="Write JSON report under .crupier/evals/runs")
    eval_compare_dataset.add_argument("--json", action="store_true", help="Print JSON")
    eval_compare_dataset.set_defaults(func=cmd_eval_compare_dataset)
    eval_history = eval_subparsers.add_parser("history", help="Summarize compare-dataset history")
    eval_history.add_argument("--model", help="Filter to one model ref")
    eval_history.add_argument("--mode", help="Filter to one mode/profile")
    eval_history.add_argument("--apply", action="store_true", help="Apply historical aggregate scores to capability cards")
    eval_history.add_argument("--dry-run", action="store_true", help="Preview apply without writing capability cards")
    eval_history.add_argument("--min-count", type=int, default=3, help="Minimum historical appearances before applying")
    eval_history.add_argument(
        "--min-confidence",
        choices=["low", "medium", "high"],
        default="medium",
        help="Minimum historical confidence before applying",
    )
    eval_history.add_argument("--json", action="store_true", help="Print JSON")
    eval_history.set_defaults(func=cmd_eval_history)

    feedback = subparsers.add_parser("feedback", help="Human feedback commands")
    feedback_subparsers = feedback.add_subparsers(dest="feedback_command", required=True)
    feedback_record = feedback_subparsers.add_parser("record", help="Record human judgement for a route/result")
    feedback_record.add_argument("--trace-id", help="Stored trace ID to infer models, mode, and strategy from")
    feedback_record.add_argument("--compare-report", help="Compare or compare-dataset JSON report to infer route metadata from")
    feedback_record.add_argument("--variant", help="Variant name/model in --compare-report; defaults to report winner")
    feedback_record.add_argument("--case-id", help="Case ID when --compare-report points at a compare-dataset report")
    feedback_record.add_argument(
        "--allow-dry-run-source",
        action="store_true",
        help="Allow feedback from a dry-run compare report for non-production calibration",
    )
    feedback_record.add_argument("--model", action="append", help="Model ref to score; can be passed multiple times")
    feedback_record.add_argument("--mode", help="Route mode/profile for this feedback")
    feedback_record.add_argument("--strategy", help="Route strategy for this feedback")
    feedback_record.add_argument("--rating", type=int, required=True, help="Human rating from 1 to 5")
    feedback_record.add_argument(
        "--verdict",
        choices=["accept", "reject", "needs_work", "unknown"],
        default="unknown",
        help="Human outcome for the result",
    )
    feedback_record.add_argument("--tag", action="append", help="Issue/signal tag; can be passed multiple times")
    feedback_record.add_argument("--note", default="", help="Short redacted note; prompts/responses are not required")
    feedback_record.add_argument("--reviewer-hash", help="Optional reviewer hash, not a raw identity")
    feedback_record.add_argument("--json", action="store_true", help="Print JSON")
    feedback_record.set_defaults(func=cmd_feedback_record)
    feedback_review = feedback_subparsers.add_parser("review", help="Create a human review packet from an eval compare report")
    feedback_review.add_argument("--compare-report", required=True, help="Compare or compare-dataset JSON report to review")
    feedback_review.add_argument("--case-id", help="Case ID when reviewing a compare-dataset report")
    feedback_review.add_argument("--variant", help="Variant name/model to review; defaults to all variants")
    feedback_review.add_argument("--no-preview", action="store_true", help="Omit output previews from the review packet")
    feedback_review.add_argument("--write-report", action="store_true", help="Write JSON and Markdown under .crupier/feedback/reviews")
    feedback_review.add_argument(
        "--write-decisions-template",
        action="store_true",
        help="Write an editable human decision JSON template under .crupier/feedback/decisions",
    )
    feedback_review.add_argument("--reviewer-hash", help="Optional reviewer hash to prefill in a decision template")
    feedback_review.add_argument("--json", action="store_true", help="Print JSON")
    feedback_review.set_defaults(func=cmd_feedback_review)
    feedback_import = feedback_subparsers.add_parser(
        "import-decisions",
        help="Import an edited human decision template as feedback records",
    )
    feedback_import.add_argument("--decisions", required=True, help="Human decision JSON template to import")
    feedback_import.add_argument("--dry-run", action="store_true", help="Validate and preview records without writing")
    feedback_import.add_argument("--reviewer-hash", help="Default reviewer hash when entries omit one")
    feedback_import.add_argument(
        "--allow-dry-run-source",
        action="store_true",
        help="Allow importing dry-run decision templates for non-production calibration",
    )
    feedback_import.add_argument(
        "--apply-to-registry",
        action="store_true",
        help="Apply imported feedback scores to capability cards after import",
    )
    feedback_import.add_argument("--min-count", type=int, default=1, help="Minimum records per model/mode group when applying")
    feedback_import.add_argument("--json", action="store_true", help="Print JSON")
    feedback_import.set_defaults(func=cmd_feedback_import_decisions)
    feedback_summary = feedback_subparsers.add_parser("summary", help="Summarize human feedback by model and mode")
    feedback_summary.add_argument("--model", help="Filter to one model ref")
    feedback_summary.add_argument("--mode", help="Filter to one mode/profile")
    feedback_summary.add_argument("--json", action="store_true", help="Print JSON")
    feedback_summary.set_defaults(func=cmd_feedback_summary)
    feedback_apply = feedback_subparsers.add_parser("apply", help="Apply human feedback scores to capability cards")
    feedback_apply.add_argument("--min-count", type=int, default=1, help="Minimum records per model/mode group")
    feedback_apply.add_argument("--dry-run", action="store_true", help="Preview without writing capability cards")
    feedback_apply.add_argument("--json", action="store_true", help="Print JSON")
    feedback_apply.set_defaults(func=cmd_feedback_apply)

    audit = subparsers.add_parser("audit", help="Run project adoption audit with human review checks")
    audit.add_argument("--dataset", help="JSON/JSONL routing eval dataset; defaults to built-in routing evals")
    audit.add_argument(
        "--orchestrator-mode",
        choices=["deterministic", "model", "hybrid"],
        help="Override [orchestrator].mode for this audit.",
    )
    audit.add_argument(
        "--provider",
        action="append",
        choices=REAL_PROVIDER_CHOICES,
        help="Provider to audit; can be passed multiple times. OpenAI is included by default as baseline.",
    )
    audit.add_argument("--no-openai-baseline", action="store_true", help="Do not add OpenAI as a baseline provider")
    audit.add_argument("--real", action="store_true", help="Run real provider canaries with small budgets")
    audit.add_argument("--all", action="store_true", help="Run real text canaries for all allowlisted chat models")
    audit.add_argument("--no-code-comments", action="store_true", help="Skip source-code integration comments")
    audit.add_argument("--code-path", action="append", help="Limit code comment scan to a path; can be passed multiple times")
    audit.add_argument("--max-code-files", type=int, default=200, help="Maximum source files to scan for comments")
    audit.add_argument("--write-report", action="store_true", help="Write JSON and Markdown reports under .crupier/audits")
    audit.add_argument("--json", action="store_true", help="Print JSON")
    audit.set_defaults(func=cmd_audit)

    adopt = subparsers.add_parser("adopt", help="Project adoption planning commands")
    adopt_subparsers = adopt.add_subparsers(dest="adopt_command", required=True)
    adopt_doctor = adopt_subparsers.add_parser("doctor", help="Run one non-destructive project adoption readiness report")
    adopt_doctor.add_argument("paths", nargs="*", help="Files or directories to scan; defaults to the project root")
    adopt_doctor.add_argument("--dataset", help="JSON/JSONL routing eval dataset; defaults to built-in routing evals")
    adopt_doctor.add_argument(
        "--orchestrator-mode",
        choices=["deterministic", "model", "hybrid"],
        help="Override [orchestrator].mode for this doctor run.",
    )
    adopt_doctor.add_argument(
        "--provider",
        action="append",
        choices=REAL_PROVIDER_CHOICES,
        help="Provider to audit; can be passed multiple times. OpenAI is included by default as baseline.",
    )
    adopt_doctor.add_argument("--no-openai-baseline", action="store_true", help="Do not add OpenAI as a baseline provider")
    adopt_doctor.add_argument("--real", action="store_true", help="Run real provider canaries with small budgets")
    adopt_doctor.add_argument("--all", action="store_true", help="Run real text canaries for all allowlisted chat models")
    adopt_doctor.add_argument(
        "--production",
        action="store_true",
        help="Require production evidence: real canaries, eval history, and human feedback",
    )
    adopt_doctor.add_argument("--max-files", type=int, default=200, help="Maximum source files to scan")
    adopt_doctor.add_argument("--write-report", action="store_true", help="Write JSON and Markdown reports under .crupier/audits")
    adopt_doctor.add_argument("--json", action="store_true", help="Print JSON")
    adopt_doctor.set_defaults(func=cmd_adopt_doctor)
    adopt_handoff = adopt_subparsers.add_parser("handoff", help="Create a human adoption handoff with next actions")
    adopt_handoff.add_argument("paths", nargs="*", help="Files or directories to scan; defaults to the project root")
    adopt_handoff.add_argument("--dataset", help="JSON/JSONL routing eval dataset; defaults to built-in routing evals")
    adopt_handoff.add_argument(
        "--orchestrator-mode",
        choices=["deterministic", "model", "hybrid"],
        help="Override [orchestrator].mode for this handoff run.",
    )
    adopt_handoff.add_argument(
        "--provider",
        action="append",
        choices=REAL_PROVIDER_CHOICES,
        help="Provider to include in real canaries; can be passed multiple times.",
    )
    adopt_handoff.add_argument("--no-openai-baseline", action="store_true", help="Do not add OpenAI as a baseline provider")
    adopt_handoff.add_argument("--real", action="store_true", help="Run real provider canaries with small budgets")
    adopt_handoff.add_argument("--all", action="store_true", help="Run real text canaries for all allowlisted chat models")
    adopt_handoff.add_argument(
        "--production",
        action="store_true",
        help="Require production evidence: real canaries, eval history, and human feedback",
    )
    adopt_handoff.add_argument("--max-files", type=int, default=200, help="Maximum source files to scan")
    adopt_handoff.add_argument("--write-report", action="store_true", help="Write JSON and Markdown reports under .crupier/handoffs")
    adopt_handoff.add_argument("--json", action="store_true", help="Print JSON")
    adopt_handoff.set_defaults(func=cmd_adopt_handoff)
    adopt_package = adopt_subparsers.add_parser(
        "package",
        help="Write the full human adoption review package for a project",
    )
    adopt_package.add_argument("paths", nargs="*", help="Files or directories to scan; defaults to the project root")
    adopt_package.add_argument("--dataset", help="JSON/JSONL routing eval dataset; defaults to built-in routing evals")
    adopt_package.add_argument(
        "--orchestrator-mode",
        choices=["deterministic", "model", "hybrid"],
        help="Override [orchestrator].mode for this package run.",
    )
    adopt_package.add_argument(
        "--provider",
        action="append",
        choices=REAL_PROVIDER_CHOICES,
        help="Provider to include in real canaries; can be passed multiple times.",
    )
    adopt_package.add_argument("--no-openai-baseline", action="store_true", help="Do not add OpenAI as a baseline provider")
    adopt_package.add_argument("--real", action="store_true", help="Run real provider canaries with small budgets")
    adopt_package.add_argument("--all", action="store_true", help="Run real text canaries for all allowlisted chat models")
    adopt_package.add_argument(
        "--production",
        action="store_true",
        help="Require production evidence: real canaries, eval history, human feedback, and signoff",
    )
    adopt_package.add_argument("--max-files", type=int, default=200, help="Maximum source files to scan")
    adopt_package.add_argument("--json", action="store_true", help="Print JSON")
    adopt_package.set_defaults(func=cmd_adopt_package)
    adopt_signoff = adopt_subparsers.add_parser("signoff", help="Record human approval/rejection for project adoption")
    adopt_signoff.add_argument(
        "--verdict",
        required=True,
        choices=["approve", "reject", "needs_work"],
        help="Human adoption decision after reviewing the handoff",
    )
    adopt_signoff.add_argument("--handoff", help="Optional handoff report path that was reviewed")
    adopt_signoff.add_argument("--adoption-path", help="Optional adoption path being approved or rejected")
    adopt_signoff.add_argument("--reviewer-hash", help="Optional reviewer hash, not a raw identity")
    adopt_signoff.add_argument("--note", default="", help="Short redacted note; do not paste prompts/responses/secrets")
    adopt_signoff.add_argument("--json", action="store_true", help="Print JSON")
    adopt_signoff.set_defaults(func=cmd_adopt_signoff)
    adopt_plan = adopt_subparsers.add_parser("plan", help="Recommend proxy/client/autopatch/native adoption path")
    adopt_plan.add_argument("paths", nargs="*", help="Files or directories to scan; defaults to the project root")
    adopt_plan.add_argument("--max-files", type=int, default=200, help="Maximum source files to scan")
    adopt_plan.add_argument("--write-report", action="store_true", help="Write JSON and Markdown reports under .crupier/audits")
    adopt_plan.add_argument("--json", action="store_true", help="Print JSON")
    adopt_plan.set_defaults(func=cmd_adopt_plan)
    adopt_patches = adopt_subparsers.add_parser("patches", help="Generate non-applied adoption patch suggestions")
    adopt_patches.add_argument("paths", nargs="*", help="Files or directories to scan; defaults to the project root")
    adopt_patches.add_argument(
        "--path",
        choices=["recommended", "proxy", "compat_client", "autopatch", "native_sdk"],
        default="recommended",
        help="Adoption path to generate suggestions for",
    )
    adopt_patches.add_argument("--max-files", type=int, default=200, help="Maximum source files to scan")
    adopt_patches.add_argument("--write-report", action="store_true", help="Write JSON and Markdown reports under .crupier/audits")
    adopt_patches.add_argument("--json", action="store_true", help="Print JSON")
    adopt_patches.set_defaults(func=cmd_adopt_patches)

    code = subparsers.add_parser("code", help="Code-facing helper commands")
    code_subparsers = code.add_subparsers(dest="code_command", required=True)
    code_comments = code_subparsers.add_parser("comments", help="Generate programmer comments for AI integration hotspots")
    code_comments.add_argument("paths", nargs="*", help="Files or directories to scan; defaults to the project root")
    code_comments.add_argument("--max-files", type=int, default=200, help="Maximum source files to scan")
    code_comments.add_argument("--write-report", action="store_true", help="Write JSON and Markdown reports under .crupier/audits")
    code_comments.add_argument(
        "--write-review-comments",
        action="store_true",
        help="Write PR/review-comment Markdown and JSONL under .crupier/code-comments",
    )
    code_comments.add_argument(
        "--write-sarif",
        action="store_true",
        help="Write SARIF annotations under .crupier/code-comments for code scanning tools",
    )
    code_comments.add_argument(
        "--write-decisions-template",
        action="store_true",
        help="Write an editable JSON template for granular programmer decisions",
    )
    code_comments.add_argument(
        "--import-decisions",
        help="Import a human-edited code-comment decision template",
    )
    code_comments.add_argument("--ack-reviewed", action="store_true", help="Record current comments as reviewed by a programmer")
    code_comments.add_argument("--reviewer-hash", help="Optional reviewer hash, not a raw identity")
    code_comments.add_argument("--note", default="", help="Short redacted review note")
    code_comments.add_argument("--json", action="store_true", help="Print JSON")
    code_comments.set_defaults(func=cmd_code_comments)

    deal = subparsers.add_parser("deal", help="Plan a route for a task")
    deal.add_argument("task", help="Task to route")
    deal.add_argument("--input", dest="input_value", help="Optional input payload; JSON is parsed when possible")
    deal.add_argument("--file", dest="files", action="append", help="File path or URL to include; can be passed multiple times")
    deal.add_argument("--mode", default=None, help="Route mode/profile")
    deal.add_argument("--strategy", default=None, help="Force a strategy")
    deal.add_argument("--force-model", help="Force an exact allowed model ref, e.g. openai:gpt-4.1-mini")
    deal.add_argument("--max-cost-usd", type=float, help="Hard budget for this request")
    deal.add_argument("--max-output-tokens", type=int, help="Provider output token cap")
    deal.add_argument("--response-schema", help="JSON Schema object for structured output")
    deal.add_argument("--trace", choices=["none", "summary", "debug"], default="none")
    deal.add_argument("--store-trace", action="store_true", help="Persist a trace under .crupier/traces")
    deal.add_argument("--store-prompt", action="store_true", help="Allow storing prompt/input data for replay")
    deal.add_argument("--store-response", action="store_true", help="Allow storing model output text/JSON")
    deal.add_argument("--json", action="store_true", help="Print JSON")
    deal.add_argument("--no-dry-run", action="store_true", help="Attempt real provider execution")
    deal.set_defaults(func=cmd_deal)

    route = subparsers.add_parser("route", help="Show route decision without provider calls")
    route.add_argument("task", help="Task to route")
    route.add_argument("--input", dest="input_value", help="Optional input payload; JSON is parsed when possible")
    route.add_argument("--file", dest="files", action="append", help="File path or URL to include; can be passed multiple times")
    route.add_argument("--mode", default=None, help="Route mode/profile")
    route.add_argument("--strategy", default=None, help="Force a strategy")
    route.add_argument("--force-model", help="Force an exact allowed model ref, e.g. openai:gpt-4.1-mini")
    route.add_argument("--max-cost-usd", type=float, help="Hard budget for this request")
    route.add_argument("--max-output-tokens", type=int, help="Provider output token cap")
    route.add_argument("--response-schema", help="JSON Schema object for structured output")
    route.add_argument("--json", action="store_true", help="Print JSON")
    route.set_defaults(func=cmd_route)

    trace_cmd = subparsers.add_parser("trace", help="Persistent trace commands")
    trace_subparsers = trace_cmd.add_subparsers(dest="trace_command", required=True)
    trace_list = trace_subparsers.add_parser("list", help="List stored traces")
    trace_list.add_argument("--json", action="store_true", help="Print JSON")
    trace_list.set_defaults(func=cmd_trace_list)
    trace_show = trace_subparsers.add_parser("show", help="Show a stored trace")
    trace_show.add_argument("trace_id", help="Trace ID to show")
    trace_show.add_argument("--json", action="store_true", help="Print JSON")
    trace_show.set_defaults(func=cmd_trace_show)
    trace_delete = trace_subparsers.add_parser("delete", help="Delete a stored trace")
    trace_delete.add_argument("trace_id", help="Trace ID to delete")
    trace_delete.set_defaults(func=cmd_trace_delete)
    trace_replay = trace_subparsers.add_parser("replay", help="Replay a stored trace")
    trace_replay.add_argument("trace_id", help="Trace ID to replay")
    trace_replay.add_argument("--no-dry-run", action="store_true", help="Attempt real provider execution")
    trace_replay.add_argument("--trace", choices=["none", "summary", "debug"], default="summary")
    trace_replay.add_argument("--json", action="store_true", help="Print JSON")
    trace_replay.set_defaults(func=cmd_trace_replay)

    smoke = subparsers.add_parser("smoke", help="Run real minimal provider calls for allowed models")
    smoke.add_argument("--provider", choices=REAL_PROVIDER_CHOICES, help="Only test one provider")
    smoke.add_argument("--model", action="append", help="Exact model ref to test; can be passed multiple times")
    smoke.add_argument("--all", action="store_true", help="Test all allowed models instead of one per provider")
    smoke.add_argument("--show-output", action="store_true", help="Print model output preview")
    smoke.add_argument("--json", action="store_true", help="Print JSON")
    smoke.set_defaults(func=cmd_smoke)

    verify = subparsers.add_parser("verify", help="Verify real provider/model readiness")
    verify.add_argument(
        "--provider",
        action="append",
        choices=REAL_PROVIDER_CHOICES,
        help="Provider to verify; can be passed multiple times. OpenAI is included by default as baseline.",
    )
    verify.add_argument("--no-openai-baseline", action="store_true", help="Do not add OpenAI as a baseline provider")
    verify.add_argument("--skip-smoke", action="store_true", help="Do not execute real model smoke calls")
    verify.add_argument("--all", action="store_true", help="Smoke/readiness-check all allowed models per provider")
    verify.add_argument("--json", action="store_true", help="Print JSON")
    verify.set_defaults(func=cmd_verify)

    serve = subparsers.add_parser("serve", help="Run an OpenAI-compatible local HTTP server")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve.add_argument("--port", type=int, default=8787, help="Bind port")
    serve.add_argument("--compat", choices=["openai"], default="openai", help="Compatibility API to expose")
    serve.add_argument(
        "--compat-mode",
        choices=["strict", "balanced", "aggressive", "locked"],
        default="balanced",
        help="How much Crupier may adapt requested models",
    )
    serve.add_argument("--no-dry-run", action="store_true", help="Attempt real provider execution")
    serve.set_defaults(func=cmd_serve)

    return parser


def cmd_init(args: argparse.Namespace) -> int:
    path = write_default_project(args.project, force=args.force)
    print(f"Created {path}")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    if args.provider and args.provider not in client.adapters:
        print(
            f"Provider {args.provider!r} is not enabled in crupier.toml or has no adapter. "
            f"Set [providers.{args.provider}].enabled = true and configure its env_key/host.",
            file=sys.stderr,
        )
        return 1
    report = client.update(dry_run=args.dry_run, online=args.online, provider=args.provider)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_update_report(report)
    return 0


def _print_update_report(report: Any) -> None:
    mode = "dry-run" if report.dry_run else "applied"
    print(f"update: {mode}")
    if report.requires_confirmation:
        print("requires_confirmation: true")

    diff = report.diff or {}
    added = list(diff.get("added", report.added_models))
    removed = list(diff.get("removed", report.removed_models))
    changed = list(diff.get("changed", []))
    unchanged = diff.get("unchanged", len(report.unchanged_models))

    print(f"added: {len(added)}")
    for model in added:
        print(f"  + {model}")
    print(f"removed: {len(removed)}")
    for model in removed:
        print(f"  - {model}")
    print(f"modified: {len(changed)}")
    for item in changed:
        print(f"  * {item['model']} ({', '.join(item.get('fields', []))})")
    print(f"unchanged: {unchanged}")

    state_counts: dict[str, int] = {}
    for item in report.model_states:
        for state in item.get("states", []):
            state_counts[state] = state_counts.get(state, 0) + 1
    if state_counts:
        summary = " ".join(f"{name}={count}" for name, count in sorted(state_counts.items()))
        print(f"states: {summary}")

    if report.written_files:
        print(f"written_files: {len(report.written_files)}")
    for warning in report.warnings:
        print(f"warning: {warning}")


def cmd_models_list(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    cards = client.models.list(allowed_only=not args.all)
    states_by_model = {item["model"]: item for item in client.registry.model_states(models=[card.model_ref.key for card in cards])}
    if args.json:
        data = []
        for card in cards:
            item = card.to_dict()
            item["registry_state"] = states_by_model.get(card.model_ref.key, {})
            data.append(item)
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    for card in cards:
        state = states_by_model.get(card.model_ref.key, {})
        states = ",".join(state.get("states", [])) or "unknown"
        print(
            f"{card.model_ref.key}\tprovider={card.model_ref.provider}\t"
            f"stability={card.model_ref.stability}\tcost={card.cost_tier}\tquality={card.quality_tier}\t"
            f"states={states}"
        )
    return 0


def cmd_models_discover(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    if args.provider and args.provider not in client.adapters:
        print(
            f"Provider {args.provider!r} is not enabled in crupier.toml or has no adapter. "
            f"Set [providers.{args.provider}].enabled = true and configure its env_key/host.",
            file=sys.stderr,
        )
        return 1
    models = client.models.discover(provider=args.provider)
    if args.json:
        print(json.dumps([model.to_dict() for model in models], indent=2, sort_keys=True))
        return 0
    if not models:
        provider = args.provider or "enabled providers"
        print(f"No models discovered for {provider}. Check provider enabled flags and API keys.")
        return 0
    for model in models:
        label = f"\tname={model.name}" if model.name else ""
        print(f"{model.model_ref}{label}")
    return 0


def cmd_models_allow(args: argparse.Namespace) -> int:
    path = write_models_allow(args.project, args.models, replace=args.replace)
    action = "Replaced" if args.replace else "Updated"
    print(f"{action} [models].allow in {path}")
    return 0


def cmd_registry_snapshot_create(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    result = client.registry.snapshot_create(args.name, allowed_only=args.allowed_only)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print(f"Created registry snapshot {result['name']} ({result['card_count']} cards)")
    print(result["path"])
    return 0


def cmd_registry_snapshot_list(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    snapshots = client.registry.snapshot_list()
    if args.json:
        print(json.dumps(snapshots, indent=2, sort_keys=True))
        return 0
    if not snapshots:
        print("No registry snapshots found.")
        return 0
    for item in snapshots:
        print(
            f"{item['name']}\tcards={item['card_count']}\tallowlist={item['allowlist_count']}\t"
            f"created_at={item.get('created_at')}"
        )
    return 0


def cmd_registry_snapshot_diff(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    diff = client.registry.snapshot_diff(args.left, args.right)
    if args.json:
        print(json.dumps(diff, indent=2, sort_keys=True))
        return 0
    print(f"left: {diff['left']['name']} ({diff['left']['card_count']} cards)")
    print(f"right: {diff['right']['name']} ({diff['right']['card_count']} cards)")
    print(f"added: {len(diff['added'])}")
    for model in diff["added"]:
        print(f"  + {model}")
    print(f"removed: {len(diff['removed'])}")
    for model in diff["removed"]:
        print(f"  - {model}")
    print(f"changed: {len(diff['changed'])}")
    for item in diff["changed"]:
        print(f"  * {item['model']} ({', '.join(item['fields'])})")
    print(f"unchanged: {diff['unchanged']}")
    return 0


def cmd_registry_snapshot_use(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    result = client.registry.snapshot_use(args.name, restore_allowlist=args.restore_allowlist)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print(f"Restored registry snapshot {result['snapshot']} ({len(result['restored_models'])} cards)")
    print(f"written: {len(result['written_files'])}")
    print(f"removed: {len(result['removed_files'])}")
    if result["allowlist_restored"]:
        print("allowlist: restored")
    return 0


def cmd_capabilities_probe(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    model_refs = _capability_probe_model_refs(
        client,
        provider=args.provider,
        explicit=args.model,
        all_models=args.all,
    )
    if not model_refs:
        target = args.provider or "project allowlist"
        print(f"No models found for {target}. Use `crupier models allow ...` or pass --all.", file=sys.stderr)
        return 1

    report = client.capabilities.probe(
        model_refs,
        probes=args.probe,
        apply=args.apply,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0
    _print_probe_report(report)
    return 0


def cmd_capabilities_readiness(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    model_refs = _capability_probe_model_refs(
        client,
        provider=args.provider,
        explicit=args.model,
        all_models=args.all,
    )
    if not model_refs:
        target = args.provider or "project allowlist"
        print(f"No models found for {target}. Use `crupier models allow ...` or pass --all.", file=sys.stderr)
        return 1

    report = client.capabilities.readiness(model_refs, strict=args.strict)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0
    _print_readiness_report(report)
    return 0


def _print_probe_report(report: Any) -> None:
    mode = "dry-run" if report.dry_run else "executed"
    persistence = "applied" if report.applied else "not-applied"
    print(f"capability_probe: {mode} {persistence}")
    summary = report.summary()
    if summary:
        print("summary: " + " ".join(f"{name}={count}" for name, count in sorted(summary.items())))
    for result in report.results:
        line = f"{result.status}\t{result.model}\t{result.probe}"
        if result.latency_ms is not None:
            line += f"\tlatency_ms={result.latency_ms}"
        if result.error:
            line += f"\terror={result.error}"
        print(line)
    if report.written_files:
        print(f"written_files: {len(report.written_files)}")
    for warning in report.warnings:
        print(f"warning: {warning}")


def _print_readiness_report(report: Any) -> None:
    mode = "strict" if report.strict else "standard"
    print(f"capability_readiness: {mode}")
    summary = report.summary()
    if summary:
        print("summary: " + " ".join(f"{name}={count}" for name, count in sorted(summary.items())))
    for item in report.items:
        pieces = [item.status, item.model]
        if item.missing_probes:
            pieces.append("missing=" + ",".join(item.missing_probes))
        if item.inferred_probes:
            pieces.append("inferred=" + ",".join(item.inferred_probes))
        if item.failed_probes:
            pieces.append("failed=" + ",".join(item.failed_probes))
        print("\t".join(pieces))


def cmd_profiles_list(args: argparse.Namespace) -> int:
    config = CrupierConfig.from_toml(Path(args.project))
    if args.json:
        data = {
            name: {"prefer": profile.prefer, "strategy": profile.strategy, **profile.options}
            for name, profile in config.profiles.items()
        }
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    for name, profile in sorted(config.profiles.items()):
        print(f"{name}\tstrategy={profile.strategy}\tprefer={','.join(profile.prefer)}")
    return 0


def cmd_release_check(args: argparse.Namespace) -> int:
    report = run_release_checks(args.project, build=not args.skip_build)
    if args.check_pypi_name:
        report.checks.append(
            check_pypi_project_name(
                report.project,
                allow_existing=args.allow_existing_pypi_project,
            )
        )
    if args.verify_providers:
        verify_report = _build_verify_report(
            Crupier.from_project(args.project),
            requested=args.provider,
            include_openai_baseline=not args.no_openai_baseline,
            run_smoke=not args.skip_provider_smoke,
            all_models=args.verify_all_models,
        )
        report.checks.append(_provider_readiness_release_check(verify_report))
    if args.strict_public:
        report.checks.append(_strict_public_release_check(report))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.ok else 1
    _print_release_check_report(report)
    return 0 if report.ok else 1


def _print_release_check_report(report: Any) -> None:
    print("release_check: " + ("ready" if report.ok else "not-ready"))
    print(f"project: {report.project}")
    if report.version:
        print(f"version: {report.version}")
    print("summary: " + " ".join(f"{name}={count}" for name, count in sorted(report.summary.items())))
    for check in report.checks:
        print(f"{check.status}\t{check.id}\t{check.summary}")
        for action in check.actions:
            print(f"  action: {action}")
    if report.build and not report.build.get("skipped"):
        build_status = "ok" if report.build.get("ok") else "failed"
        print(f"build: {build_status} wheels={report.build.get('wheel_count', 0)}")


def cmd_eval_run(args: argparse.Namespace) -> int:
    config = CrupierConfig.from_toml(Path(args.project))
    if args.orchestrator_mode:
        config.orchestrator.mode = args.orchestrator_mode
    client = Crupier(config)
    report = client.evals.run(dataset=args.dataset, write_report=args.write_report)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.ok else 1
    _print_eval_report(report)
    return 0 if report.ok else 1


def cmd_eval_compare(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    report = client.evals.compare(
        task=args.task,
        input=_parse_input(args.input_value),
        mode=args.mode,
        strategy=args.strategy,
        constraints=_cli_constraints(args),
        variants=_compare_variants(args),
        response_schema=_parse_response_schema(args.response_schema),
        expect_contains=args.expect_contains,
        dry_run=not args.no_dry_run,
        write_report=args.write_report,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.winner else 1
    _print_compare_report(report)
    return 0 if report.winner else 1


def cmd_eval_compare_dataset(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    report = client.evals.compare_dataset(
        dataset=args.dataset,
        variants=_compare_variants(args),
        constraints=_cli_constraints(args),
        expect_contains=args.expect_contains,
        dry_run=not args.no_dry_run,
        apply=args.apply,
        min_count=args.min_count,
        min_confidence=args.min_confidence,
        record_history=args.record_history,
        write_report=args.write_report,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.ok else 1
    _print_compare_dataset_report(report)
    return 0 if report.ok else 1


def cmd_eval_history(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    report = client.evals.history(
        model=args.model,
        mode=args.mode,
        apply=args.apply,
        min_count=args.min_count,
        min_confidence=args.min_confidence,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0
    _print_compare_history_report(report)
    return 0


def _print_eval_report(report: Any) -> None:
    status = "pass" if report.ok else "fail"
    print(
        f"eval: {status} dataset={report.name} orchestrator={report.orchestrator_mode} "
        f"passed={report.passed}/{report.total}"
    )
    for result in report.results:
        line = f"{result.status}\t{result.id}"
        if result.strategy:
            line += f"\tstrategy={result.strategy}"
        if result.models:
            line += f"\tmodels={','.join(result.models)}"
        print(line)
        for check in result.failed_checks:
            print(f"  failed: {check}")
        if result.error:
            print(f"  error: {result.error}")
    if report.written_path:
        print(f"written_report: {report.written_path}")


def _print_compare_report(report: Any) -> None:
    mode = "dry-run" if report.dry_run else "real"
    print(f"compare: {mode} passed={report.passed}/{report.total}")
    if report.winner:
        print(f"winner: {report.winner}")
    print(f"recommendation: {report.recommendation}")
    for item in report.variants:
        line = f"{item.status}\t{item.name}"
        if item.strategy:
            line += f"\tstrategy={item.strategy}"
        if item.models:
            line += f"\tmodels={','.join(item.models)}"
        if item.estimated_cost_usd is not None:
            line += f"\test_cost={item.estimated_cost_usd:.8f}"
        if item.actual_cost_usd is not None:
            line += f"\tactual_cost={item.actual_cost_usd:.8f}"
        if item.latency_ms is not None:
            line += f"\tlatency_ms={item.latency_ms}"
        elif item.estimated_latency_ms is not None:
            line += f"\test_latency_ms={item.estimated_latency_ms}"
        print(line)
        for check in item.failed_checks:
            print(f"  failed: {check}")
        if item.error:
            print(f"  error: {item.error}")
        if item.output_preview:
            print(f"  preview: {item.output_preview}")
        for question in item.human_questions:
            print(f"  human_check: {question}")
    if report.written_path:
        print(f"written_report: {report.written_path}")


def _print_compare_dataset_report(report: Any) -> None:
    mode = "dry-run" if report.dry_run else "real"
    print(
        f"compare_dataset: {mode} dataset={report.name} "
        f"passed_cases={report.passed_cases}/{report.total_cases}"
    )
    for case in report.cases:
        status = "pass" if case.ok else "fail"
        print(f"{status}\t{case.id}\twinner={case.winner}")
    if report.model_scores:
        print("model_scores:")
        for score in report.model_scores:
            line = (
                f"  {score.model}\tmode={score.mode}\tappearances={score.appearances}\t"
                f"passed={score.passed}\twins={score.wins}\t{score.score_key}={score.score_delta}"
            )
            if score.avg_actual_cost_usd is not None:
                line += f"\tavg_actual_cost={score.avg_actual_cost_usd:.8f}"
            elif score.avg_estimated_cost_usd is not None:
                line += f"\tavg_est_cost={score.avg_estimated_cost_usd:.8f}"
            if score.avg_latency_ms is not None:
                line += f"\tavg_latency_ms={score.avg_latency_ms}"
            elif score.avg_estimated_latency_ms is not None:
                line += f"\tavg_est_latency_ms={score.avg_estimated_latency_ms}"
            print(line)
    if report.apply_report:
        print(f"applied_scores: {len(report.apply_report.get('updated', []))}")
        print(
            f"apply_gate: min_count={report.apply_report.get('min_count')} "
            f"min_confidence={report.apply_report.get('min_confidence')}"
        )
        for item in report.apply_report.get("updated", []):
            print(f"  updated\t{item['model']}\t{item['score_key']}={item['new_score']}")
        for item in report.apply_report.get("skipped", []):
            print(f"  skipped\t{item['model']}\t{item['reason']}")
    if getattr(report, "history_path", None):
        print(f"history: {report.history_path}")
    if report.written_path:
        print(f"written_report: {report.written_path}")


def _print_compare_history_report(report: Any) -> None:
    print(f"compare_history: runs={report.total_runs}")
    if report.last_run_at:
        print(f"last_run_at: {report.last_run_at}")
    if not report.model_scores:
        print("No compare history found.")
    for score in report.model_scores:
        line = (
            f"{score.model}\tmode={score.mode}\truns={score.runs}\tappearances={score.appearances}\t"
            f"passed={score.passed}\twins={score.wins}\t{score.score_key}={score.score_delta}\t"
            f"confidence={score.confidence}\ttrend={score.trend}"
        )
        if score.avg_actual_cost_usd is not None:
            line += f"\tavg_actual_cost={score.avg_actual_cost_usd:.8f}"
        elif score.avg_estimated_cost_usd is not None:
            line += f"\tavg_est_cost={score.avg_estimated_cost_usd:.8f}"
        if score.avg_latency_ms is not None:
            line += f"\tavg_latency_ms={score.avg_latency_ms}"
        elif score.avg_estimated_latency_ms is not None:
            line += f"\tavg_est_latency_ms={score.avg_estimated_latency_ms}"
        print(line)
    if report.apply_report:
        print(f"applied_scores: {len(report.apply_report.get('updated', []))}")
        print(
            f"apply_gate: min_count={report.apply_report.get('min_count')} "
            f"min_confidence={report.apply_report.get('min_confidence')}"
        )
        for item in report.apply_report.get("updated", []):
            print(f"  updated\t{item['model']}\t{item['score_key']}={item['new_score']}")
        for item in report.apply_report.get("skipped", []):
            print(f"  skipped\t{item['model']}\t{item['reason']}")
    for warning in report.warnings:
        print(f"warning: {warning}")


def cmd_feedback_record(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    if args.trace_id and args.compare_report:
        raise CrupierError("Use either --trace-id or --compare-report, not both.")
    derived = (
        _feedback_from_compare_report(
            args.project,
            report_path=args.compare_report,
            variant=args.variant,
            case_id=args.case_id,
            allow_dry_run_source=args.allow_dry_run_source,
        )
        if args.compare_report
        else {}
    )
    record = client.feedback.record(
        project=client.config.project.name,
        trace_id=args.trace_id,
        models=args.model or derived.get("models"),
        mode=args.mode or derived.get("mode"),
        strategy=args.strategy or derived.get("strategy"),
        rating=args.rating,
        verdict=args.verdict,
        tags=[*(args.tag or []), *derived.get("tags", [])],
        note=args.note or derived.get("note", ""),
        reviewer_hash=args.reviewer_hash,
        trace_store=client.traces,
    )
    if args.json:
        print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
        return 0
    print(f"feedback_recorded: {record.feedback_id}")
    print(f"models: {', '.join(record.models)}")
    if record.mode:
        print(f"mode: {record.mode}")
    if record.strategy:
        print(f"strategy: {record.strategy}")
    print(f"rating: {record.rating}")
    print(f"verdict: {record.verdict}")
    return 0


def cmd_feedback_review(args: argparse.Namespace) -> int:
    packet = build_human_review_packet(
        args.project,
        report_path=args.compare_report,
        case_id=args.case_id,
        variant=args.variant,
        include_output_preview=not args.no_preview,
    )
    if args.write_report:
        packet.written_files = [str(path) for path in write_human_review_packet(args.project, packet)]
    if args.write_decisions_template:
        decision_path = write_human_decision_template(args.project, packet, reviewer_hash=args.reviewer_hash)
        packet.written_files = [*packet.written_files, str(decision_path)]
    if args.json:
        print(json.dumps(packet.to_dict(), indent=2, sort_keys=True))
        return 0 if packet.ok else 1
    _print_human_review_packet(packet)
    return 0 if packet.ok else 1


def _print_human_review_packet(packet: Any) -> None:
    print("human_review: " + ("ready" if packet.ok else "empty"))
    print(f"source: {packet.source_path}")
    print(f"type: {packet.source_type}")
    print(f"items: {packet.total_items} recommended={packet.recommended_items}")
    if packet.dry_run:
        print("warning: dry-run compare report; run --no-dry-run before production feedback.")
    for warning in packet.warnings:
        if "dry run" not in warning:
            print(f"warning: {warning}")
    for item in packet.items:
        label = item.id + (" recommended" if item.recommended else "")
        line = f"{item.status}\t{label}\tvariant={item.variant}"
        if item.models:
            line += f"\tmodels={','.join(item.models)}"
        print(line)
        for question in item.human_questions:
            print(f"  human_check: {question}")
        for name, command in item.feedback_commands.items():
            print(f"  {name}: {command}")
    if packet.written_files:
        print("written_files:")
        for path in packet.written_files:
            print(f"  {path}")


def cmd_feedback_import_decisions(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    decision_path = Path(args.decisions).expanduser()
    if not decision_path.is_absolute() and not decision_path.exists():
        decision_path = Path(args.project) / decision_path
    result = import_human_decisions(
        client.feedback,
        project=client.config.project.name,
        decision_path=decision_path,
        dry_run=args.dry_run,
        reviewer_hash=args.reviewer_hash,
        allow_dry_run_source=args.allow_dry_run_source,
    )
    payload: dict[str, Any] = result.to_dict()
    if args.apply_to_registry and not args.dry_run:
        payload["apply_report"] = client.feedback.apply_to_registry(
            client.registry,
            min_count=args.min_count,
            dry_run=False,
        )
    elif args.apply_to_registry and args.dry_run:
        payload["apply_report"] = {"dry_run": True, "updated": [], "skipped": [], "written_files": []}

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    mode = "dry-run" if args.dry_run else "imported"
    print(f"feedback_import_decisions: {mode}")
    print(f"decisions: {result.decision_path}")
    print(f"records: {result.imported}")
    if result.skipped:
        print(f"skipped: {len(result.skipped)}")
    if payload.get("apply_report"):
        apply_report = payload["apply_report"]
        print(f"applied_scores: {len(apply_report.get('updated', []))}")
        for item in apply_report.get("updated", []):
            print(f"  updated\t{item['model']}\t{item['score_key']}={item['new_score']}")
    return 0


def cmd_feedback_summary(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    summary = client.feedback.summary(model=args.model, mode=args.mode)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if not summary["groups"]:
        print("No human feedback found.")
        return 0
    print(f"feedback: records={summary['count']} groups={len(summary['groups'])}")
    for item in summary["groups"]:
        print(
            f"{item['status']}\t{item['model']}\tmode={item['mode']}\t"
            f"count={item['count']}\tavg_rating={item['avg_rating']}\tscore_delta={item['score_delta']}"
        )
        tags = ", ".join(f"{tag['tag']}:{tag['count']}" for tag in item.get("top_tags", []))
        if tags:
            print(f"  tags: {tags}")
    return 0


def cmd_feedback_apply(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    report = client.feedback.apply_to_registry(
        client.registry,
        min_count=args.min_count,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    mode = "dry-run" if report["dry_run"] else "applied"
    print(f"feedback_apply: {mode}")
    for item in report["updated"]:
        print(
            f"updated\t{item['model']}\tmode={item['mode']}\t"
            f"{item['score_key']}={item['new_score']}\tcount={item['count']}"
        )
    for item in report["skipped"]:
        print(f"skipped\t{item['model']}\t{item['reason']}")
    if report["written_files"]:
        print(f"written_files: {len(report['written_files'])}")
    return 0


def _feedback_from_compare_report(
    project_root: str | Path,
    *,
    report_path: str,
    variant: str | None,
    case_id: str | None,
    allow_dry_run_source: bool = False,
) -> dict[str, Any]:
    data = _read_feedback_report(project_root, report_path)
    comparison, selected_case_id = _select_comparison_from_report(data, case_id=case_id)
    source_dry_run = _comparison_dry_run(data, comparison)
    if source_dry_run and not allow_dry_run_source:
        raise CrupierError(
            "Compare report is dry-run. Run `crupier eval compare --no-dry-run --write-report` "
            "before production feedback, or pass --allow-dry-run-source for non-production calibration."
        )
    selected_variant = _select_variant_from_comparison(comparison, variant=variant)
    models = [ModelRef.parse(model).key for model in selected_variant.get("models", [])]
    if not models:
        raise CrupierError("Selected compare variant has no route models to score.")
    selected_name = str(selected_variant.get("name") or variant or comparison.get("winner") or "variant")
    tags = ["compare_report", f"compare_variant:{selected_name}"]
    if source_dry_run:
        tags.append("dry_run_source")
    if selected_case_id:
        tags.append(f"compare_case:{selected_case_id}")
    return {
        "models": models,
        "mode": selected_variant.get("mode") or comparison.get("mode"),
        "strategy": selected_variant.get("strategy"),
        "tags": tags,
        "note": f"Reviewed compare report variant {selected_name}.",
    }


def _comparison_dry_run(data: dict[str, Any], comparison: dict[str, Any]) -> bool:
    if "dry_run" in comparison:
        return bool(comparison.get("dry_run"))
    return bool(data.get("dry_run", True))


def _read_feedback_report(project_root: str | Path, report_path: str) -> dict[str, Any]:
    path = Path(report_path).expanduser()
    if not path.is_absolute() and not path.exists():
        path = Path(project_root) / path
    if not path.exists():
        raise CrupierError(f"Compare report not found: {report_path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CrupierError(f"Compare report is not valid JSON: {report_path}") from exc
    if not isinstance(data, dict):
        raise CrupierError("Compare report must be a JSON object.")
    return data


def _select_comparison_from_report(data: dict[str, Any], *, case_id: str | None) -> tuple[dict[str, Any], str | None]:
    if isinstance(data.get("variants"), list):
        return data, None
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise CrupierError("Report must be from `eval compare` or `eval compare-dataset`.")
    selected: dict[str, Any] | None = None
    if case_id:
        selected = next((case for case in cases if str(case.get("id")) == case_id), None)
        if selected is None:
            raise CrupierError(f"Case {case_id!r} was not found in compare-dataset report.")
    elif len(cases) == 1:
        selected = cases[0]
    else:
        raise CrupierError("Use --case-id when a compare-dataset report contains more than one case.")
    comparison = selected.get("comparison") if isinstance(selected, dict) else None
    if not isinstance(comparison, dict) or not isinstance(comparison.get("variants"), list):
        raise CrupierError("Selected compare-dataset case does not contain variant details.")
    return comparison, str(selected.get("id")) if selected.get("id") is not None else None


def _select_variant_from_comparison(comparison: dict[str, Any], *, variant: str | None) -> dict[str, Any]:
    variants = comparison.get("variants")
    if not isinstance(variants, list):
        raise CrupierError("Compare report does not contain variants.")
    target = variant or comparison.get("winner")
    if not target:
        raise CrupierError("Pass --variant because the compare report has no winner.")
    for item in variants:
        if not isinstance(item, dict):
            continue
        if str(item.get("name")) == str(target):
            return item
        if str(target) in {str(model) for model in item.get("models", []) or []}:
            return item
    raise CrupierError(f"Variant {target!r} was not found in compare report.")


def cmd_audit(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    report = client.audit.run(
        dataset=args.dataset,
        providers=args.provider,
        include_openai_baseline=not args.no_openai_baseline,
        orchestrator_mode=args.orchestrator_mode,
        real=args.real,
        all_models=args.all,
        include_code_comments=not args.no_code_comments,
        code_paths=args.code_path,
        max_code_files=args.max_code_files,
        write_report=args.write_report,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.ok else 1
    _print_audit_report(report)
    return 0 if report.ok else 1


def _print_audit_report(report: Any) -> None:
    print("audit: " + ("ready" if report.ok else "not-ready"))
    print("summary: " + " ".join(f"{name}={count}" for name, count in sorted(report.summary.items())))
    for check in report.checks:
        line = f"{check.status}\t{check.id}\t{check.summary}"
        print(line)
        for action in check.actions:
            print(f"  action: {action}")
    if report.route_reviews:
        print("route_reviews:")
        for review in report.route_reviews:
            line = f"  {review.status}\t{review.id}"
            if review.strategy:
                line += f"\tstrategy={review.strategy}"
            if review.models:
                line += f"\tmodels={','.join(review.models)}"
            print(line)
            for question in review.human_questions:
                print(f"    human_check: {question}")
    if report.real_canaries:
        print("real_canaries:")
        for item in report.real_canaries:
            status = "ok" if item.get("ok") else "failed"
            line = f"  {status}\t{item.get('id')}"
            if item.get("latency_ms") is not None:
                line += f"\tlatency_ms={item['latency_ms']}"
            if item.get("error"):
                line += f"\terror={item['error']}"
            print(line)
    if report.code_comments:
        print(f"code_comments: {len(report.code_comments)}")
        for comment in report.code_comments[:20]:
            print(f"  P{comment.priority}\t{comment.file}:{comment.line}\t{comment.title}")
        if len(report.code_comments) > 20:
            print(f"  ... {len(report.code_comments) - 20} more")
    if report.written_files:
        for path in report.written_files:
            print(f"written_report: {path}")


def cmd_adopt_doctor(args: argparse.Namespace) -> int:
    try:
        client = Crupier.from_project(args.project)
    except CrupierConfigError:
        if args.real or args.production or args.dataset or args.provider or args.orchestrator_mode or args.all:
            raise CrupierError(
                "Config-free doctor only supports offline adoption review. "
                "Run `crupier init` before --real, --production, providers, datasets, --all, or orchestrator overrides."
            )
        report = build_config_free_project_doctor(
            args.project,
            project=_adopt_project_name(args.project),
            paths=args.paths or None,
            max_files=args.max_files,
        )
    else:
        report = build_project_doctor(
            client,
            paths=args.paths or None,
            max_files=args.max_files,
            dataset=args.dataset,
            providers=args.provider,
            include_openai_baseline=not args.no_openai_baseline,
            orchestrator_mode=args.orchestrator_mode,
            real=args.real,
            all_models=args.all,
            production=args.production,
        )
    if args.write_report:
        report.written_files = [str(path) for path in write_project_doctor_report(args.project, report)]
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.ready else 1
    _print_project_doctor(report)
    return 0 if report.ready else 1


def cmd_adopt_handoff(args: argparse.Namespace) -> int:
    try:
        client = Crupier.from_project(args.project)
    except CrupierConfigError:
        if args.real or args.production or args.dataset or args.provider or args.orchestrator_mode:
            raise CrupierError(
                "Config-free handoff only supports offline adoption review. "
                "Run `crupier init` before --real, --production, providers, datasets, or orchestrator overrides."
            )
        report = build_config_free_adoption_handoff(
            args.project,
            project=_adopt_project_name(args.project),
            paths=args.paths or None,
            max_files=args.max_files,
        )
    else:
        report = build_adoption_handoff(
            client,
            paths=args.paths or None,
            max_files=args.max_files,
            dataset=args.dataset,
            providers=args.provider,
            include_openai_baseline=not args.no_openai_baseline,
            orchestrator_mode=args.orchestrator_mode,
            real=args.real,
            all_models=args.all,
            production=args.production,
        )
    if args.write_report:
        report.written_files = [str(path) for path in write_adoption_handoff_report(args.project, report)]
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.doctor.ready else 1
    _print_adoption_handoff(report)
    return 0 if report.doctor.ready else 1


def cmd_adopt_package(args: argparse.Namespace) -> int:
    paths = args.paths or None
    try:
        client = Crupier.from_project(args.project)
    except CrupierConfigError:
        if args.real or args.production or args.dataset or args.provider or args.orchestrator_mode or args.all:
            raise CrupierError(
                "Config-free package only supports offline adoption review. "
                "Run `crupier init` before --real, --production, providers, datasets, --all, or orchestrator overrides."
            )
        project_name = _adopt_project_name(args.project)
        doctor = build_config_free_project_doctor(
            args.project,
            project=project_name,
            paths=paths,
            max_files=args.max_files,
        )
    else:
        project_name = client.config.project.name
        doctor = build_project_doctor(
            client,
            paths=paths,
            max_files=args.max_files,
            dataset=args.dataset,
            providers=args.provider,
            include_openai_baseline=not args.no_openai_baseline,
            orchestrator_mode=args.orchestrator_mode,
            real=args.real,
            all_models=args.all,
            production=args.production,
        )

    artifact_groups: dict[str, list[str]] = {}
    comments = doctor.adoption_plan.code_comments
    artifact_groups["code_comments"] = [str(path) for path in write_code_comments_report(args.project, comments)]
    artifact_groups["code_review_comments"] = [str(path) for path in write_code_review_comments(args.project, comments)]
    artifact_groups["code_sarif"] = [str(write_code_comments_sarif(args.project, comments))]
    artifact_groups["code_comment_decisions"] = [str(write_code_comment_decision_template(args.project, comments))]
    artifact_groups["adoption_patches"] = [str(path) for path in write_adoption_patch_report(args.project, doctor.patch_report)]
    artifact_groups["project_doctor"] = [str(path) for path in write_project_doctor_report(args.project, doctor)]
    doctor.written_files = artifact_groups["project_doctor"]
    handoff = build_adoption_handoff_from_doctor(
        args.project,
        project=project_name,
        doctor=doctor,
        paths=paths,
    )
    artifact_groups["adoption_handoff"] = [str(path) for path in write_adoption_handoff_report(args.project, handoff)]
    written_files = [path for paths_for_group in artifact_groups.values() for path in paths_for_group]
    payload = {
        "project": project_name,
        "status": handoff.status,
        "ready": handoff.ready,
        "doctor_status": doctor.status,
        "doctor_ready": doctor.ready,
        "readiness_mode": doctor.readiness_mode,
        "recommended_path": doctor.recommended_path,
        "summary": doctor.summary,
        "review_contract": doctor.review_contract,
        "artifact_groups": artifact_groups,
        "written_files": written_files,
        "required_human_actions": handoff.required_human_actions,
        "suggested_commands": handoff.suggested_commands,
    }
    package_paths, payload = write_adoption_package_index(args.project, payload)
    artifact_groups = payload["artifact_groups"]
    written_files = payload["written_files"]
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"adoption_package: {handoff.status}")
        print(f"doctor: {doctor.status}")
        print(f"readiness_mode: {doctor.readiness_mode}")
        print(f"recommended_path: {doctor.recommended_path}")
        print("artifacts:")
        for name, paths_for_group in artifact_groups.items():
            print(f"  {name}: {len(paths_for_group)}")
            for path in paths_for_group:
                print(f"    {path}")
        print("package_index:")
        for path in package_paths:
            print(f"  {path}")
        if handoff.required_human_actions:
            print("human_actions:")
            for action in handoff.required_human_actions:
                print(f"  - {action}")
    return 0 if doctor.ready else 1


def cmd_adopt_signoff(args: argparse.Namespace) -> int:
    try:
        client = Crupier.from_project(args.project)
        project_name = client.config.project.name
    except CrupierConfigError:
        project_name = _adopt_project_name(args.project)
    record = record_adoption_signoff(
        args.project,
        project=project_name,
        verdict=args.verdict,
        reviewer_hash=args.reviewer_hash,
        note=args.note,
        handoff=args.handoff,
        adoption_path=args.adoption_path,
    )
    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        print(f"adoption_signoff: {record['verdict']}")
        print(f"project: {record['project']}")
        print(f"path: {record['path']}")
        if record.get("handoff"):
            print(f"handoff: {record['handoff']}")
    return 0


def cmd_adopt_plan(args: argparse.Namespace) -> int:
    project_name = _adopt_project_name(args.project)
    plan = build_adoption_plan(
        args.project,
        project=project_name,
        paths=args.paths or None,
        max_files=args.max_files,
    )
    if args.write_report:
        plan.written_files = [str(path) for path in write_adoption_plan_report(args.project, plan)]
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0 if plan.ready else 1
    _print_adoption_plan(plan)
    return 0 if plan.ready else 1


def cmd_adopt_patches(args: argparse.Namespace) -> int:
    project_name = _adopt_project_name(args.project)
    report = build_adoption_patches(
        args.project,
        project=project_name,
        adoption_path=args.path,
        paths=args.paths or None,
        max_files=args.max_files,
    )
    if args.write_report:
        report.written_files = [str(path) for path in write_adoption_patch_report(args.project, report)]
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.ready else 1
    _print_adoption_patch_report(report)
    return 0 if report.ready else 1


def _adopt_project_name(project_root: str | Path) -> str:
    root = Path(project_root)
    package_json = root / "package.json"
    if package_json.exists():
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
            name = str(data.get("name") or "").strip()
            if name:
                return name
        except (OSError, json.JSONDecodeError):
            pass
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            match = re.search(r"(?m)^name\s*=\s*['\"]([^'\"]+)['\"]", pyproject.read_text(encoding="utf-8"))
            if match:
                return match.group(1)
        except OSError:
            pass
    return root.resolve().name


def _print_project_doctor(report: Any) -> None:
    print("doctor: " + report.status)
    print(f"readiness_mode: {report.readiness_mode}")
    print(f"recommended_path: {report.recommended_path}")
    print(f"confidence: {report.confidence}")
    print("summary: " + " ".join(f"{name}={count}" for name, count in sorted(report.summary.items())))
    print("gates:")
    for gate in report.gates:
        print(f"  {gate.status}\t{gate.id}\t{gate.summary}")
        for action in gate.actions[:4]:
            print(f"    action: {action}")
    if report.adoption_plan.blockers:
        print("blockers:")
        for blocker in report.adoption_plan.blockers:
            print(f"  - {blocker}")
    print("next_steps:")
    for item in report.adoption_plan.checklist[:6]:
        print(f"  - {item}")
    if report.patch_report.patches:
        print(f"patch_suggestions: {len(report.patch_report.patches)}")
        for patch in report.patch_report.patches[:5]:
            print(f"  {patch.status}\t{patch.title}")
    feedback_count = int(report.feedback_summary.get("count", 0) or 0)
    print(f"human_feedback_records: {feedback_count}")
    applied_feedback = getattr(report, "applied_feedback_summary", {}) or {}
    if applied_feedback:
        print(
            "human_feedback_applied_groups: "
            f"{applied_feedback.get('applied_count', 0)}/{applied_feedback.get('count', 0)}"
        )
    signoff = getattr(report, "adoption_signoff_summary", {}) or {}
    if signoff:
        print(f"adoption_signoff: {signoff.get('status', 'missing')}")
    if report.written_files:
        for path in report.written_files:
            print(f"written_report: {path}")


def _print_adoption_handoff(report: Any) -> None:
    print("handoff: " + report.status)
    print(f"doctor: {report.doctor.status}")
    print(f"readiness_mode: {report.doctor.readiness_mode}")
    print(f"recommended_path: {report.doctor.recommended_path}")
    signoff = getattr(report.doctor, "adoption_signoff_summary", {}) or {}
    if signoff:
        print(f"adoption_signoff: {signoff.get('status', 'missing')}")
    if report.required_human_actions:
        print("human_actions:")
        for action in report.required_human_actions:
            print(f"  - {action}")
    if report.suggested_commands:
        print("commands:")
        for command in report.suggested_commands:
            print(f"  {command}")
    print("artifacts:")
    for name, paths in sorted(report.artifacts.items()):
        print(f"  {name}: {len(paths)}")
        for path in paths[:3]:
            print(f"    {path}")
    if report.written_files:
        for path in report.written_files:
            print(f"written_report: {path}")


def _print_adoption_plan(plan: Any) -> None:
    print("adoption: " + ("ready" if plan.ready else "blocked"))
    print(f"recommended_path: {plan.recommended_path}")
    print(f"confidence: {plan.confidence}")
    if plan.blockers:
        print("blockers:")
        for blocker in plan.blockers:
            print(f"  - {blocker}")
    print("options:")
    for option in plan.options:
        print(f"  {option.status}\t{option.path}\tscore={option.score}\t{option.summary}")
        for action in option.actions[:3]:
            print(f"    action: {action}")
        for risk in option.risks[:2]:
            print(f"    risk: {risk}")
    print("checklist:")
    for item in plan.checklist:
        print(f"  - {item}")
    if plan.code_comments:
        print(f"code_comments: {len(plan.code_comments)}")
        for comment in plan.code_comments[:20]:
            print(f"  P{comment.priority}\t{comment.file}:{comment.line}\t{comment.title}")
        if len(plan.code_comments) > 20:
            print(f"  ... {len(plan.code_comments) - 20} more")
    for warning in plan.warnings:
        print(f"warning: {warning}")
    for path in plan.written_files:
        print(f"written_report: {path}")


def _print_adoption_patch_report(report: Any) -> None:
    print("adoption_patches: " + ("ready" if report.ready else "blocked"))
    print(f"adoption_path: {report.adoption_path}")
    if report.blockers:
        print("blockers:")
        for blocker in report.blockers:
            print(f"  - {blocker}")
    for patch in report.patches:
        print(f"{patch.status}\t{patch.title}")
        print(f"  {patch.summary}")
        for command in patch.commands:
            print(f"  command: {command}")
        for note in patch.notes:
            print(f"  note: {note}")
        if patch.diff:
            print("  diff:")
            for line in patch.diff.rstrip().splitlines():
                print(f"    {line}")
    for warning in report.warnings:
        print(f"warning: {warning}")
    for path in report.written_files:
        print(f"written_report: {path}")


def cmd_code_comments(args: argparse.Namespace) -> int:
    if args.ack_reviewed and args.import_decisions:
        raise CrupierError("--ack-reviewed and --import-decisions are mutually exclusive.")
    comments = scan_code_comments(
        args.project,
        paths=args.paths or None,
        max_files=args.max_files,
    )
    written_files: list[str] = []
    if args.write_report:
        written_files = [str(path) for path in write_code_comments_report(args.project, comments)]
    if args.write_review_comments:
        written_files.extend(str(path) for path in write_code_review_comments(args.project, comments))
    if args.write_sarif:
        written_files.append(str(write_code_comments_sarif(args.project, comments)))
    if args.write_decisions_template:
        written_files.append(str(write_code_comment_decision_template(args.project, comments)))
    acknowledged: dict[str, Any] | None = None
    imported_decisions: dict[str, Any] | None = None
    if args.import_decisions:
        imported_decisions = import_code_comment_decisions(
            args.project,
            comments,
            args.import_decisions,
            reviewer_hash=args.reviewer_hash,
            note=args.note,
        )
    if args.ack_reviewed:
        acknowledged = acknowledge_code_comments(
            args.project,
            comments,
            reviewer_hash=args.reviewer_hash,
            note=args.note,
        )
    review_summary = summarize_code_comment_reviews(args.project, comments)
    if args.json:
        print(
            json.dumps(
                {
                    "count": len(comments),
                    "comments": [comment.to_dict() for comment in comments],
                    "review": review_summary.to_dict(),
                    "acknowledged": acknowledged,
                    "imported_decisions": imported_decisions,
                    "written_files": written_files,
                    "review_comment_files": [
                        path for path in written_files if "review_comments_" in Path(path).name
                    ],
                    "sarif_files": [
                        path for path in written_files if Path(path).suffix == ".sarif"
                    ],
                    "decision_template_files": [
                        path for path in written_files if "code_comment_decisions_" in Path(path).name
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not comments:
        print("No AI integration hotspots found.")
    else:
        print(f"code_comments: {len(comments)}")
        print(
            f"review: reviewed={review_summary.reviewed_count} "
            f"pending={review_summary.pending_count}"
        )
        for comment in comments:
            print(f"P{comment.priority}\t{comment.file}:{comment.line}\t{comment.title}")
            print(f"  {comment.body}")
    if acknowledged:
        print(f"ack_reviewed: {acknowledged['review_id']} count={acknowledged['comment_count']}")
        print(f"review_log: {acknowledged['path']}")
    if imported_decisions:
        print(
            f"imported_decisions: {imported_decisions['review_id']} "
            f"reviewed={imported_decisions['comment_count']} pending={imported_decisions['pending_decision_count']}"
        )
        print(f"review_log: {imported_decisions['path']}")
    for path in written_files:
        print(f"written_report: {path}")
    return 0


def cmd_deal(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    trace: bool | str = False if args.trace == "none" else args.trace
    result = client.deal(
        task=args.task,
        input=_parse_input(args.input_value),
        files=args.files,
        mode=args.mode,
        strategy=args.strategy,
        constraints=_cli_constraints(args),
        response_schema=_parse_response_schema(args.response_schema),
        trace=trace,
        dry_run=not args.no_dry_run,
    )
    if args.json:
        print(json.dumps(result.to_dict(trace_summary=args.trace != "debug"), indent=2, sort_keys=True))
        return 0
    print(result.output_text)
    if result.route:
        print(f"route: {result.route.strategy} | {result.route.model_summary}")
    if result.warnings:
        for warning in result.warnings:
            print(f"warning: {warning}")
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    result = client.deal(
        task=args.task,
        input=_parse_input(args.input_value),
        files=args.files,
        mode=args.mode,
        strategy=args.strategy,
        constraints=_cli_constraints(args),
        response_schema=_parse_response_schema(args.response_schema),
        trace="summary",
        dry_run=True,
    )
    if args.json:
        print(json.dumps(result.route.to_dict() if result.route else {}, indent=2, sort_keys=True))
        return 0
    if not result.route:
        print("No route planned.")
        return 1
    print(f"strategy: {result.route.strategy}")
    print(f"models: {result.route.model_summary}")
    print(f"estimated_cost_usd: {result.route.estimated_cost.estimated_usd:.8f}")
    print(f"reason: {result.route.reason}")
    if result.route.selection_scores:
        print("scores:")
        for item in result.route.selection_scores:
            print(f"  {item['score']:>5.1f}  {item['model']}")
            for term in item.get("terms", []):
                print(f"         {term['value']:>5.1f}  {term['name']}: {term['reason']}")
    return 0


def cmd_trace_list(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    refs = client.traces.list()
    if args.json:
        print(json.dumps([ref.to_dict() for ref in refs], indent=2, sort_keys=True))
        return 0
    if not refs:
        print("No stored traces found.")
        return 0
    for ref in refs:
        models = ",".join(ref.models or [])
        replayable = "replayable" if ref.replayable else "metadata-only"
        print(f"{ref.trace_id}\t{ref.created_at}\t{ref.strategy}\t{models}\t{replayable}\t{ref.summary}")
    return 0


def cmd_trace_show(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    record = client.traces.read(args.trace_id)
    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    route = record.get("result", {}).get("route") or {}
    print(f"trace_id: {record.get('trace_id')}")
    print(f"created_at: {record.get('created_at')}")
    print(f"project: {record.get('project')}")
    print(f"replayable: {record.get('replayable')}")
    print(f"summary: {record.get('request', {}).get('summary')}")
    print(f"strategy: {route.get('strategy')}")
    print(f"models: {', '.join(_route_models_from_record(route))}")
    print(f"cost: {record.get('result', {}).get('cost')}")
    print(f"storage: {record.get('storage_decision')}")
    return 0


def cmd_trace_delete(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    path = client.traces.delete(args.trace_id)
    print(f"Deleted {path}")
    return 0


def cmd_trace_replay(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    trace: bool | str = False if args.trace == "none" else args.trace
    result = client.traces.replay(args.trace_id, client, dry_run=not args.no_dry_run, trace=trace)
    if args.json:
        print(json.dumps(result.to_dict(trace_summary=args.trace != "debug"), indent=2, sort_keys=True))
        return 0
    print(result.output_text)
    if result.route:
        print(f"route: {result.route.strategy} | {result.route.model_summary}")
    return 0


def _route_models_from_record(route: dict[str, Any]) -> list[str]:
    models: list[str] = []
    for step in route.get("steps", []) or []:
        for model in [step.get("model"), *(step.get("models") or [])]:
            if model and model not in models:
                models.append(str(model))
    return models


def cmd_smoke(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    model_refs = _smoke_model_refs(client.config, provider=args.provider, explicit=args.model, all_models=args.all)
    if not model_refs:
        provider = args.provider or "enabled providers"
        print(
            f"No allowed models found for {provider}. Use `crupier models discover` and `crupier models allow ...` first.",
            file=sys.stderr,
        )
        return 1

    results = []
    ok = True
    for model_ref in model_refs:
        try:
            result = client.deal(
                task='Smoke test. Reply with exactly: "crupier-ok"',
                mode="fast",
                strategy="single",
                constraints={
                    "force_model": model_ref,
                    "max_output_tokens": 16,
                    "store_prompt": False,
                    "store_response": False,
                },
                dry_run=False,
                trace="summary",
            )
            output_text = result.output_text.strip()
            passed = "crupier-ok" in output_text.lower()
            ok = ok and passed
            item = {
                "ok": passed,
                "provider": ModelRef.parse(model_ref).provider,
                "model": model_ref,
                "latency_ms": result.latency_ms,
                "calls": result.provider_metadata.get("calls", []),
            }
            if args.show_output:
                item["output_preview"] = output_text[:240]
            results.append(item)
        except Exception as exc:  # noqa: BLE001 - smoke should continue across providers
            ok = False
            results.append(
                {
                    "ok": False,
                    "provider": ModelRef.parse(model_ref).provider,
                    "model": model_ref,
                    "error": _redact_secrets(str(exc)),
                    "error_type": exc.__class__.__name__,
                }
            )

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for item in results:
            status = "ok" if item["ok"] else "failed"
            line = f"{status}\t{item['model']}"
            if item.get("latency_ms") is not None:
                line += f"\tlatency_ms={item['latency_ms']}"
            if item.get("error"):
                line += f"\terror={item['error']}"
            print(line)
            if args.show_output and item.get("output_preview"):
                print(f"output: {item['output_preview']}")
    return 0 if ok else 1


def cmd_verify(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    report = _build_verify_report(
        client,
        requested=args.provider,
        include_openai_baseline=not args.no_openai_baseline,
        run_smoke=not args.skip_smoke,
        all_models=args.all,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_verify_report(report)
    return 0 if report["ok"] else 1


def cmd_serve(args: argparse.Namespace) -> int:
    client = Crupier.from_project(args.project)
    server = build_openai_compatible_server(
        crupier=client,
        host=args.host,
        port=args.port,
        dry_run=not args.no_dry_run,
        compat_mode=args.compat_mode,
    )
    host, port = server.server_address
    print(f"crupier serve: http://{host}:{port}/v1 ({args.compat}, mode={args.compat_mode})", file=sys.stderr)
    print("Set OPENAI_BASE_URL to this URL for OpenAI-compatible clients.", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("crupier serve: stopped", file=sys.stderr)
    finally:
        server.server_close()
    return 0


def _parse_input(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_response_schema(value: str | None) -> Any:
    if value is None:
        return None
    parsed = _parse_input(value)
    if not isinstance(parsed, dict):
        raise CrupierError("--response-schema must be a JSON object.")
    return parsed


def _cli_constraints(args: argparse.Namespace) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    if getattr(args, "max_cost_usd", None) is not None:
        constraints["max_cost_usd"] = args.max_cost_usd
    if getattr(args, "max_output_tokens", None) is not None:
        constraints["max_output_tokens"] = args.max_output_tokens
    if getattr(args, "force_model", None):
        constraints["force_model"] = args.force_model
    if getattr(args, "store_trace", False) or getattr(args, "store_prompt", False) or getattr(args, "store_response", False):
        constraints["store_trace"] = True
    if getattr(args, "store_prompt", False):
        constraints["store_prompt"] = True
    if getattr(args, "store_response", False):
        constraints["store_response"] = True
    return constraints


def _compare_variants(args: argparse.Namespace) -> list[CompareVariant] | None:
    variants: list[CompareVariant] = []
    for model in getattr(args, "model", None) or []:
        model_ref = ModelRef.parse(model).key
        variants.append(
            CompareVariant(
                name=model_ref,
                constraints={"force_model": model_ref},
            )
        )
    for raw in getattr(args, "variant", None) or []:
        data = _parse_input(raw)
        if not isinstance(data, dict):
            raise CrupierError("--variant must be a JSON object.")
        constraints = dict(data.get("constraints", {}))
        if data.get("model"):
            constraints["force_model"] = ModelRef.parse(str(data["model"])).key
        variant = CompareVariant.from_dict({**data, "constraints": constraints})
        variants.append(variant)
    return variants or None


def _smoke_model_refs(
    config: CrupierConfig,
    *,
    provider: str | None,
    explicit: list[str] | None,
    all_models: bool,
) -> list[str]:
    if explicit:
        refs = [ModelRef.parse(model).key for model in explicit]
        return [model for model in refs if provider is None or ModelRef.parse(model).provider == provider]

    enabled = {
        name
        for name, settings in config.providers.items()
        if settings.enabled and (provider is None or name == provider)
    }
    selected: list[str] = []
    seen_provider: set[str] = set()
    for model in config.models.allow:
        ref = ModelRef.parse(model)
        if ref.provider not in enabled:
            continue
        if all_models or ref.provider not in seen_provider:
            selected.append(ref.key)
            seen_provider.add(ref.provider)
    return selected


def _verify_provider_names(
    config: CrupierConfig,
    *,
    requested: list[str] | None,
    include_openai_baseline: bool,
) -> list[str]:
    selected = list(requested or [])
    if not selected:
        selected = [
            provider
            for provider in REAL_PROVIDER_CHOICES
            if provider in config.providers and config.providers[provider].enabled
        ]
    if include_openai_baseline and "openai" not in selected:
        selected.insert(0, "openai")
    seen: set[str] = set()
    ordered: list[str] = []
    for provider in REAL_PROVIDER_CHOICES:
        if provider in selected and provider not in seen:
            ordered.append(provider)
            seen.add(provider)
    return ordered


def _build_verify_report(
    client: Crupier,
    *,
    requested: list[str] | None,
    include_openai_baseline: bool,
    run_smoke: bool,
    all_models: bool,
) -> dict[str, Any]:
    providers = _verify_provider_names(
        client.config,
        requested=requested,
        include_openai_baseline=include_openai_baseline,
    )
    items = [
        _verify_provider(
            client,
            provider,
            run_smoke=run_smoke,
            all_models=all_models,
        )
        for provider in providers
    ]
    return {
        "ok": all(item["status"] == "ready" for item in items),
        "openai_baseline": include_openai_baseline,
        "providers": providers,
        "summary": _status_summary(items),
        "items": items,
    }


def _provider_readiness_release_check(report: dict[str, Any]) -> ReleaseCheck:
    ok = bool(report.get("ok"))
    return ReleaseCheck(
        id="provider_readiness",
        status="pass" if ok else "fail",
        severity="high",
        summary="Real provider/model readiness checks passed."
        if ok
        else "Real provider/model readiness checks failed.",
        evidence=report,
        actions=[
            "Run `crupier verify --provider ...` with real environment keys, fix blocked providers or allowed models, then retry `crupier release check --verify-providers`."
        ]
        if not ok
        else [],
    )


def _strict_public_release_check(report: Any) -> ReleaseCheck:
    warning_ids = [check.id for check in report.checks if check.status == "warn"]
    build_skipped = bool(report.build.get("skipped"))
    ok = not warning_ids and not build_skipped
    actions: list[str] = []
    if warning_ids:
        actions.append("Resolve release warnings before publishing publicly: " + ", ".join(warning_ids) + ".")
    if build_skipped:
        actions.append("Run the strict public release gate without --skip-build so wheel/sdist install smokes execute.")
    return ReleaseCheck(
        id="strict_public",
        status="pass" if ok else "fail",
        severity="high",
        summary="Strict public release gate passed."
        if ok
        else "Strict public release gate failed because warnings remain or build checks were skipped.",
        evidence={"warning_ids": warning_ids, "build_skipped": build_skipped},
        actions=actions,
    )


def _verify_provider(
    client: Crupier,
    provider: str,
    *,
    run_smoke: bool,
    all_models: bool,
) -> dict[str, Any]:
    settings = client.config.providers.get(provider)
    item: dict[str, Any] = {
        "provider": provider,
        "status": "unknown",
        "enabled": bool(settings and settings.enabled),
        "adapter_configured": provider in client.adapters,
        "env": _provider_env_status(settings, provider),
        "allowed_models": [],
        "discovered_count": None,
        "discovered_sample": [],
        "readiness": None,
        "smoke": [],
        "issues": [],
    }

    if settings is None:
        item["issues"].append(f"Provider {provider!r} is not configured in crupier.toml.")
    elif not settings.enabled:
        item["issues"].append(f"Provider {provider!r} is disabled in crupier.toml.")
    if provider not in client.adapters:
        item["issues"].append(f"No adapter is configured for provider {provider!r}.")
    env = item["env"]
    if env["required"] and not env["present"]:
        item["issues"].append(f"Missing required environment variable {env['key']}.")

    model_refs = _smoke_model_refs(client.config, provider=provider, explicit=None, all_models=all_models)
    item["allowed_models"] = model_refs
    if not model_refs:
        item["issues"].append(f"No allowed models found for provider {provider!r}.")

    if item["issues"]:
        item["status"] = "blocked"
        return item

    try:
        discovered = client.models.discover(provider=provider)
        item["discovered_count"] = len(discovered)
        item["discovered_sample"] = [model.model_ref for model in discovered[:5]]
    except Exception as exc:  # noqa: BLE001 - verification reports provider boundary failures
        item["issues"].append(_redact_secrets(f"Discovery failed: {exc}"))

    try:
        readiness = client.capabilities.readiness(model_refs)
        item["readiness"] = readiness.to_dict()
    except Exception as exc:  # noqa: BLE001 - verification should keep reporting other checks
        item["issues"].append(_redact_secrets(f"Readiness check failed: {exc}"))

    if run_smoke:
        item["smoke"] = _run_smoke_checks(client, model_refs)
    else:
        item["smoke_skipped"] = True

    item["status"] = _provider_verify_status(item, run_smoke=run_smoke)
    return item


def _run_smoke_checks(client: Crupier, model_refs: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for model_ref in model_refs:
        card = client.registry.get(model_ref)
        if card.model_kind == "embedding" or (card.supports_embeddings and card.modalities_output == ["embedding"]):
            results.append(_run_embedding_smoke(client, model_ref))
            continue
        try:
            result = client.deal(
                task='Smoke test. Reply with exactly: "crupier-ok"',
                mode="fast",
                strategy="single",
                constraints={
                    "force_model": model_ref,
                    "max_output_tokens": 16,
                    "store_prompt": False,
                    "store_response": False,
                },
                dry_run=False,
                trace="summary",
            )
            output_text = result.output_text.strip()
            results.append(
                {
                    "ok": "crupier-ok" in output_text.lower(),
                    "model": model_ref,
                    "latency_ms": result.latency_ms,
                    "provider": ModelRef.parse(model_ref).provider,
                }
            )
        except Exception as exc:  # noqa: BLE001 - smoke should report per-model failures
            results.append(
                {
                    "ok": False,
                    "model": model_ref,
                    "provider": ModelRef.parse(model_ref).provider,
                    "error_type": exc.__class__.__name__,
                    "error": _redact_secrets(str(exc)),
                }
            )
    return results


def _run_embedding_smoke(client: Crupier, model_ref: str) -> dict[str, Any]:
    ref = ModelRef.parse(model_ref)
    adapter = client.adapters.get(ref.provider)
    try:
        if adapter is None or not hasattr(adapter, "embed"):
            raise RuntimeError(f"No embedding adapter configured for provider {ref.provider!r}.")
        response = adapter.embed(model=ref.model, input=["crupier embedding smoke"])
        dimensions = len(response.embeddings[0]) if response.embeddings else 0
        return {
            "ok": dimensions > 0,
            "model": model_ref,
            "provider": ref.provider,
            "kind": "embeddings",
            "embedding_dimensions": dimensions,
        }
    except Exception as exc:  # noqa: BLE001 - verification reports per-model failures
        return {
            "ok": False,
            "model": model_ref,
            "provider": ref.provider,
            "kind": "embeddings",
            "error_type": exc.__class__.__name__,
            "error": _redact_secrets(str(exc)),
        }


def _provider_verify_status(item: dict[str, Any], *, run_smoke: bool) -> str:
    if item["issues"]:
        return "failed"
    readiness = item.get("readiness") or {}
    readiness_summary = readiness.get("summary") or {}
    if readiness_summary.get("failed"):
        return "failed"
    if run_smoke and any(not result.get("ok") for result in item.get("smoke", [])):
        return "failed"
    if readiness_summary.get("needs_probes"):
        return "needs_probes"
    return "ready"


def _provider_env_status(settings: Any, provider: str) -> dict[str, Any]:
    env_key = getattr(settings, "env_key", None) or DEFAULT_PROVIDER_ENV_KEYS.get(provider)
    host = getattr(settings, "host", None)
    required = provider in {"openai", "anthropic", "google"} or (provider == "ollama" and _ollama_cloud_host(host))
    if provider == "google":
        present = google_env_present(settings)
        env_key = google_env_label(settings)
    else:
        present = bool(env_key and os.environ.get(env_key))
    return {
        "key": env_key,
        "required": required,
        "present": present,
        "host": host,
    }


def _ollama_cloud_host(host: str | None) -> bool:
    if not host:
        return False
    lowered = host.lower()
    return "ollama.com" in lowered and not lowered.startswith("http://localhost") and not lowered.startswith(
        "http://127.0.0.1"
    )


def _status_summary(items: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for item in items:
        status = item["status"]
        summary[status] = summary.get(status, 0) + 1
    return summary


def _print_verify_report(report: dict[str, Any]) -> None:
    print("verify: " + ("ready" if report["ok"] else "not-ready"))
    print("providers: " + ", ".join(report["providers"]))
    print("summary: " + " ".join(f"{name}={count}" for name, count in sorted(report["summary"].items())))
    for item in report["items"]:
        line = f"{item['status']}\t{item['provider']}"
        if item["allowed_models"]:
            line += f"\tmodels={len(item['allowed_models'])}"
        if item["discovered_count"] is not None:
            line += f"\tdiscovered={item['discovered_count']}"
        print(line)
        env = item["env"]
        env_state = "set" if env["present"] else "missing"
        env_required = "required" if env["required"] else "optional"
        print(f"  env: {env['key']}={env_state} ({env_required})")
        for issue in item["issues"]:
            print(f"  issue: {issue}")
        readiness = item.get("readiness")
        if readiness:
            summary = readiness.get("summary", {})
            print("  readiness: " + " ".join(f"{name}={count}" for name, count in sorted(summary.items())))
        for smoke in item.get("smoke", []):
            smoke_status = "ok" if smoke.get("ok") else "failed"
            smoke_line = f"  smoke: {smoke_status} {smoke['model']}"
            if smoke.get("kind"):
                smoke_line += f" kind={smoke['kind']}"
            if smoke.get("embedding_dimensions"):
                smoke_line += f" dimensions={smoke['embedding_dimensions']}"
            if smoke.get("latency_ms") is not None:
                smoke_line += f" latency_ms={smoke['latency_ms']}"
            if smoke.get("error"):
                smoke_line += f" error={smoke['error']}"
            print(smoke_line)


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


def _capability_probe_model_refs(
    client: Crupier,
    *,
    provider: str | None,
    explicit: list[str] | None,
    all_models: bool,
) -> list[str]:
    if explicit:
        refs = [ModelRef.parse(model).key for model in explicit]
        return [model for model in refs if provider is None or ModelRef.parse(model).provider == provider]

    cards = client.registry.list(allowed_only=not all_models)
    refs: list[str] = []
    for card in cards:
        if provider is not None and card.model_ref.provider != provider:
            continue
        refs.append(card.model_ref.key)
    return refs


if __name__ == "__main__":
    raise SystemExit(main())
