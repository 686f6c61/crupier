"""Human feedback storage and scoring.

This layer captures project-local human judgement without storing prompts or
responses. It lets maintainers turn "the code passed, but the answer was not
good enough" into a routing signal.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .errors import CrupierError
from .models import ModelRef


VERDICTS = {"accept", "reject", "needs_work", "unknown"}


@dataclass(slots=True)
class HumanFeedbackRecord:
    feedback_id: str
    created_at: str
    project: str
    rating: int
    verdict: str = "unknown"
    models: list[str] = field(default_factory=list)
    mode: str | None = None
    strategy: str | None = None
    trace_id: str | None = None
    tags: list[str] = field(default_factory=list)
    note: str = ""
    reviewer_hash: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HumanFeedbackRecord":
        return cls(
            feedback_id=str(data["feedback_id"]),
            created_at=str(data["created_at"]),
            project=str(data.get("project", "")),
            rating=int(data["rating"]),
            verdict=str(data.get("verdict", "unknown")),
            models=[ModelRef.parse(str(model)).key for model in data.get("models", [])],
            mode=data.get("mode"),
            strategy=data.get("strategy"),
            trace_id=data.get("trace_id"),
            tags=[str(tag) for tag in data.get("tags", [])],
            note=str(data.get("note", "")),
            reviewer_hash=data.get("reviewer_hash"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in (None, "", [])}


@dataclass(slots=True)
class HumanReviewItem:
    id: str
    variant: str
    models: list[str]
    mode: str | None = None
    strategy: str | None = None
    case_id: str | None = None
    task: str = ""
    status: str = ""
    recommended: bool = False
    dry_run: bool = True
    estimated_cost_usd: float | None = None
    actual_cost_usd: float | None = None
    estimated_latency_ms: int | None = None
    latency_ms: int | None = None
    failed_checks: list[str] = field(default_factory=list)
    human_questions: list[str] = field(default_factory=list)
    output_preview: str = ""
    feedback_commands: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in (None, "", [])}


@dataclass(slots=True)
class HumanReviewPacket:
    source_path: str
    source_type: str
    dry_run: bool
    total_items: int
    recommended_items: int
    items: list[HumanReviewItem]
    warnings: list[str] = field(default_factory=list)
    written_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "dry_run": self.dry_run,
            "total_items": self.total_items,
            "recommended_items": self.recommended_items,
            "items": [item.to_dict() for item in self.items],
            "warnings": self.warnings,
            "written_files": self.written_files,
        }


@dataclass(slots=True)
class HumanDecisionImportResult:
    decision_path: str
    dry_run: bool
    imported: int
    skipped: list[dict[str, Any]] = field(default_factory=list)
    records: list[HumanFeedbackRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_path": self.decision_path,
            "dry_run": self.dry_run,
            "imported": self.imported,
            "skipped": self.skipped,
            "records": [record.to_dict() for record in self.records],
        }


class HumanFeedbackStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.path = self.root / "feedback.jsonl"

    def record(
        self,
        *,
        project: str,
        rating: int,
        verdict: str = "unknown",
        trace_id: str | None = None,
        models: list[str] | None = None,
        mode: str | None = None,
        strategy: str | None = None,
        tags: list[str] | None = None,
        note: str = "",
        reviewer_hash: str | None = None,
        trace_store: Any | None = None,
    ) -> HumanFeedbackRecord:
        rating = _validate_rating(rating)
        verdict = _validate_verdict(verdict)
        derived = self._derive_from_trace(trace_id, trace_store) if trace_id and trace_store else {}
        model_refs = _normalize_models(models or derived.get("models") or [])
        if not model_refs:
            raise CrupierError("Feedback requires at least one --model or a trace_id with route models.")
        record = HumanFeedbackRecord(
            feedback_id=f"hfb_{uuid4().hex[:16]}",
            created_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            project=project,
            rating=rating,
            verdict=verdict,
            models=model_refs,
            mode=mode or derived.get("mode"),
            strategy=strategy or derived.get("strategy"),
            trace_id=trace_id,
            tags=_normalize_tags(tags or []),
            note=_redact(_truncate_note(note)),
            reviewer_hash=reviewer_hash,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"schema_version": 1, **record.to_dict()}, sort_keys=True) + "\n")
        return record

    def list(self) -> list[HumanFeedbackRecord]:
        if not self.path.exists():
            return []
        records: list[HumanFeedbackRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                records.append(HumanFeedbackRecord.from_dict(data))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return records

    def summary(self, *, model: str | None = None, mode: str | None = None) -> dict[str, Any]:
        model_key = ModelRef.parse(model).key if model else None
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for record in self.list():
            if mode and record.mode != mode:
                continue
            for item in record.models:
                if model_key and item != model_key:
                    continue
                group_mode = record.mode or "overall"
                group = groups.setdefault(
                    (item, group_mode),
                    {
                        "model": item,
                        "mode": group_mode,
                        "count": 0,
                        "dry_run_source_count": 0,
                        "rating_sum": 0,
                        "verdicts": {"accept": 0, "reject": 0, "needs_work": 0, "unknown": 0},
                        "tags": {},
                    },
                )
                group["count"] += 1
                if "dry_run_source" in record.tags:
                    group["dry_run_source_count"] += 1
                group["rating_sum"] += record.rating
                group["verdicts"][record.verdict] = group["verdicts"].get(record.verdict, 0) + 1
                for tag in record.tags:
                    group["tags"][tag] = group["tags"].get(tag, 0) + 1

        items: list[dict[str, Any]] = []
        for group in groups.values():
            count = int(group["count"])
            avg_rating = float(group["rating_sum"]) / count if count else 0.0
            score_delta = _score_delta(avg_rating, group["verdicts"])
            items.append(
                {
                    "model": group["model"],
                    "mode": group["mode"],
                    "count": count,
                    "dry_run_source_count": int(group["dry_run_source_count"]),
                    "avg_rating": round(avg_rating, 2),
                    "score_delta": score_delta,
                    "status": _score_status(score_delta),
                    "verdicts": group["verdicts"],
                    "top_tags": _top_tags(group["tags"]),
                }
            )
        items.sort(key=lambda item: (item["model"], item["mode"]))
        dry_run_source_count = sum(int(item["dry_run_source_count"]) for item in items)
        return {
            "count": sum(item["count"] for item in items),
            "dry_run_source_count": dry_run_source_count,
            "production_feedback_count": sum(item["count"] for item in items) - dry_run_source_count,
            "groups": items,
        }

    def apply_to_registry(self, registry: Any, *, min_count: int = 1, dry_run: bool = False) -> dict[str, Any]:
        summary = self.summary()
        updated: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        written_files: list[str] = []
        for group in summary["groups"]:
            if int(group["count"]) < min_count:
                skipped.append({"model": group["model"], "reason": f"count below min_count={min_count}"})
                continue
            try:
                card = registry.get(group["model"])
            except Exception as exc:  # noqa: BLE001 - report missing cards without aborting the whole apply
                skipped.append({"model": group["model"], "reason": str(exc)})
                continue
            key = f"human:{group['mode']}"
            old_score = card.local_eval_scores.get(key)
            card.local_eval_scores[key] = group["score_delta"]
            if not dry_run:
                path = registry.save_card(card, dry_run=False)
                if path:
                    written_files.append(path)
            updated.append(
                {
                    "model": group["model"],
                    "mode": group["mode"],
                    "score_key": key,
                    "old_score": old_score,
                    "new_score": group["score_delta"],
                    "count": group["count"],
                }
            )
        return {
            "dry_run": dry_run,
            "min_count": min_count,
            "updated": updated,
            "skipped": skipped,
            "written_files": written_files,
        }

    @staticmethod
    def _derive_from_trace(trace_id: str | None, trace_store: Any) -> dict[str, Any]:
        if not trace_id:
            return {}
        record = trace_store.read(trace_id)
        route = record.get("result", {}).get("route") or {}
        return {
            "models": _route_models(route),
            "mode": record.get("request", {}).get("mode"),
            "strategy": route.get("strategy"),
        }


def build_human_review_packet(
    project_root: str | Path,
    *,
    report_path: str,
    case_id: str | None = None,
    variant: str | None = None,
    include_output_preview: bool = True,
) -> HumanReviewPacket:
    project_root_path = Path(project_root)
    resolved_path = _resolve_report_path(project_root_path, report_path)
    data = _load_review_report(resolved_path)
    source_type = _review_source_type(data)
    items, dry_run = _review_items_from_report(
        data,
        report_path=str(resolved_path),
        case_id=case_id,
        variant=variant,
        include_output_preview=include_output_preview,
    )
    warnings: list[str] = []
    if dry_run:
        warnings.append("This compare report is a dry run; run a real --no-dry-run comparison before production feedback.")
    if not items:
        warnings.append("No review items matched the selected case/variant.")
    return HumanReviewPacket(
        source_path=str(resolved_path),
        source_type=source_type,
        dry_run=dry_run,
        total_items=len(items),
        recommended_items=sum(1 for item in items if item.recommended),
        items=items,
        warnings=warnings,
    )


def write_human_review_packet(root: str | Path, packet: HumanReviewPacket) -> list[Path]:
    reviews_dir = Path(root) / ".crupier" / "feedback" / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = reviews_dir / f"human_review_{timestamp}.json"
    md_path = reviews_dir / f"human_review_{timestamp}.md"
    json_path.write_text(json.dumps(packet.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(format_human_review_markdown(packet), encoding="utf-8")
    packet.written_files = [str(json_path), str(md_path)]
    json_path.write_text(json.dumps(packet.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [json_path, md_path]


def build_human_decision_template(
    packet: HumanReviewPacket,
    *,
    reviewer_hash: str | None = None,
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for item in packet.items:
        tags = ["human_review", f"compare_variant:{item.variant}"]
        if item.recommended:
            tags.append("recommended_variant")
        if item.case_id:
            tags.append(f"compare_case:{item.case_id}")
        decisions.append(
            {
                "id": item.id,
                "case_id": item.case_id,
                "variant": item.variant,
                "models": item.models,
                "mode": item.mode,
                "strategy": item.strategy,
                "recommended": item.recommended,
                "status": item.status,
                "human_questions": item.human_questions,
                "record": False,
                "rating": None,
                "verdict": "unknown",
                "tags": tags,
                "note": "",
                "reviewer_hash": reviewer_hash,
            }
        )
    return {
        "schema_version": 1,
        "kind": "crupier_human_decisions",
        "source_review": packet.source_path,
        "source_type": packet.source_type,
        "source_dry_run": packet.dry_run,
        "instructions": [
            "Set record=true only for variants a human reviewer has judged.",
            "For recorded decisions, set rating to an integer from 1 to 5 and verdict to accept, needs_work, reject, or unknown.",
            "Notes are optional and redacted on import. Do not paste prompts, responses, credentials, or private user data.",
        ],
        "decisions": decisions,
    }


def write_human_decision_template(
    root: str | Path,
    packet: HumanReviewPacket,
    *,
    reviewer_hash: str | None = None,
) -> Path:
    decisions_dir = Path(root) / ".crupier" / "feedback" / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = decisions_dir / f"human_decisions_{timestamp}.json"
    template = build_human_decision_template(packet, reviewer_hash=reviewer_hash)
    path.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def import_human_decisions(
    store: HumanFeedbackStore,
    *,
    project: str,
    decision_path: str | Path,
    dry_run: bool = False,
    reviewer_hash: str | None = None,
    allow_dry_run_source: bool = False,
) -> HumanDecisionImportResult:
    path = Path(decision_path).expanduser()
    if not path.exists():
        raise CrupierError(f"Human decision file not found: {decision_path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CrupierError(f"Human decision file is not valid JSON: {decision_path}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("decisions"), list):
        raise CrupierError("Human decision file must contain a decisions list.")

    source_dry_run = bool(data.get("source_dry_run", False))
    records: list[HumanFeedbackRecord] = []
    skipped: list[dict[str, Any]] = []
    for index, item in enumerate(data["decisions"]):
        if not isinstance(item, dict):
            skipped.append({"index": index, "reason": "decision is not an object"})
            continue
        should_record = bool(item.get("record", item.get("apply", False)))
        if not should_record:
            skipped.append({"id": str(item.get("id") or index), "reason": "record=false"})
            continue
        if source_dry_run and not dry_run and not allow_dry_run_source:
            raise CrupierError(
                "Human decision template was generated from a dry-run compare report. "
                "Run a real comparison with --no-dry-run before production feedback, or pass "
                "--allow-dry-run-source for non-production calibration."
            )
        models = [str(model) for model in item.get("models") or []]
        if not models:
            raise CrupierError(f"Decision {item.get('id') or index!r} has no models to score.")
        raw_rating = item.get("rating")
        if not isinstance(raw_rating, int):
            raise CrupierError(f"Decision {item.get('id') or index!r} rating must be an integer from 1 to 5.")
        rating = _validate_rating(raw_rating)
        verdict = _validate_verdict(str(item.get("verdict") or "unknown"))
        tags = [str(tag) for tag in item.get("tags") or []]
        if "decision_import" not in tags:
            tags.append("decision_import")
        if source_dry_run and "dry_run_source" not in tags:
            tags.append("dry_run_source")
        note = str(item.get("note") or "")
        selected_reviewer = str(item.get("reviewer_hash") or reviewer_hash or "") or None
        if dry_run:
            preview = HumanFeedbackRecord(
                feedback_id=f"dry_run:{item.get('id') or index}",
                created_at="dry-run",
                project=project,
                rating=rating,
                verdict=verdict,
                models=_normalize_models(models),
                mode=item.get("mode"),
                strategy=item.get("strategy"),
                tags=_normalize_tags(tags),
                note=_redact(_truncate_note(note)),
                reviewer_hash=selected_reviewer,
            )
            records.append(preview)
            continue
        records.append(
            store.record(
                project=project,
                models=models,
                mode=item.get("mode"),
                strategy=item.get("strategy"),
                rating=rating,
                verdict=verdict,
                tags=tags,
                note=note,
                reviewer_hash=selected_reviewer,
            )
        )
    return HumanDecisionImportResult(
        decision_path=str(path),
        dry_run=dry_run,
        imported=len(records),
        skipped=skipped,
        records=records,
    )


def format_human_review_markdown(packet: HumanReviewPacket) -> str:
    lines = [
        "# Crupier Human Review",
        "",
        f"- source: `{packet.source_path}`",
        f"- type: `{packet.source_type}`",
        f"- dry_run: `{str(packet.dry_run).lower()}`",
        f"- review_items: `{packet.total_items}`",
        f"- recommended_items: `{packet.recommended_items}`",
    ]
    if packet.warnings:
        lines.extend(["", "## Warnings"])
        for warning in packet.warnings:
            lines.append(f"- {warning}")
    lines.append("")
    lines.append("## Review Items")
    for item in packet.items:
        title = item.id
        if item.recommended:
            title += " (recommended)"
        lines.extend(["", f"### {title}", ""])
        if item.task:
            lines.append(f"- task: {item.task}")
        if item.case_id:
            lines.append(f"- case: `{item.case_id}`")
        lines.append(f"- variant: `{item.variant}`")
        lines.append(f"- status: `{item.status}`")
        if item.mode:
            lines.append(f"- mode: `{item.mode}`")
        if item.strategy:
            lines.append(f"- strategy: `{item.strategy}`")
        if item.models:
            lines.append(f"- models: `{', '.join(item.models)}`")
        cost = item.actual_cost_usd if item.actual_cost_usd is not None else item.estimated_cost_usd
        latency = item.latency_ms if item.latency_ms is not None else item.estimated_latency_ms
        if cost is not None:
            lines.append(f"- cost_usd: `{cost}`")
        if latency is not None:
            lines.append(f"- latency_ms: `{latency}`")
        for failed in item.failed_checks:
            lines.append(f"- failed_check: {failed}")
        for question in item.human_questions:
            lines.append(f"- human_check: {question}")
        if item.output_preview:
            lines.extend(["", "Preview:", "", "```text", item.output_preview, "```"])
        if item.feedback_commands:
            lines.extend(["", "Feedback commands:"])
            for name, command in item.feedback_commands.items():
                lines.append(f"- {name}: `{command}`")
    lines.append("")
    return "\n".join(lines)


def _validate_rating(rating: int) -> int:
    try:
        value = int(rating)
    except (TypeError, ValueError) as exc:
        raise CrupierError("Feedback rating must be an integer from 1 to 5.") from exc
    if value < 1 or value > 5:
        raise CrupierError("Feedback rating must be an integer from 1 to 5.")
    return value


def _resolve_report_path(project_root: Path, report_path: str) -> Path:
    path = Path(report_path).expanduser()
    if not path.is_absolute() and not path.exists():
        path = project_root / path
    if not path.exists():
        raise CrupierError(f"Compare report not found: {report_path}")
    return path.resolve()


def _load_review_report(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CrupierError(f"Compare report is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise CrupierError("Compare report must be a JSON object.")
    return data


def _review_source_type(data: dict[str, Any]) -> str:
    if isinstance(data.get("cases"), list):
        return "compare_dataset"
    if isinstance(data.get("variants"), list):
        return "compare"
    raise CrupierError("Review requires an eval compare or eval compare-dataset report.")


def _review_items_from_report(
    data: dict[str, Any],
    *,
    report_path: str,
    case_id: str | None,
    variant: str | None,
    include_output_preview: bool,
) -> tuple[list[HumanReviewItem], bool]:
    source_type = _review_source_type(data)
    if source_type == "compare":
        dry_run = bool(data.get("dry_run", True))
        comparison = data
        comparison_items = _review_items_from_comparison(
            comparison,
            report_path=report_path,
            case_id=None,
            task=str(comparison.get("task") or ""),
            winner=comparison.get("winner"),
            dry_run=dry_run,
            variant=variant,
            include_output_preview=include_output_preview,
        )
        return comparison_items, dry_run

    cases = data.get("cases") or []
    if not isinstance(cases, list):
        raise CrupierError("Compare-dataset report has invalid cases.")
    dry_run = bool(data.get("dry_run", True))
    items: list[HumanReviewItem] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        current_case_id = str(case.get("id") or "")
        if case_id and current_case_id != case_id:
            continue
        case_comparison = case.get("comparison")
        if not isinstance(case_comparison, dict):
            continue
        items.extend(
            _review_items_from_comparison(
                case_comparison,
                report_path=report_path,
                case_id=current_case_id,
                task=str(case.get("task") or case_comparison.get("task") or ""),
                winner=case.get("winner") or case_comparison.get("winner"),
                dry_run=bool(case_comparison.get("dry_run", dry_run)),
                variant=variant,
                include_output_preview=include_output_preview,
            )
        )
    return items, dry_run


def _review_items_from_comparison(
    comparison: dict[str, Any],
    *,
    report_path: str,
    case_id: str | None,
    task: str,
    winner: Any,
    dry_run: bool,
    variant: str | None,
    include_output_preview: bool,
) -> list[HumanReviewItem]:
    variants = comparison.get("variants") or []
    if not isinstance(variants, list):
        raise CrupierError("Compare report has invalid variants.")
    items: list[HumanReviewItem] = []
    for raw_variant in variants:
        if not isinstance(raw_variant, dict):
            continue
        name = str(raw_variant.get("name") or "variant")
        if variant and variant not in {name, *[str(model) for model in raw_variant.get("models", []) or []]}:
            continue
        item_id = f"{case_id}:{name}" if case_id else name
        models = [ModelRef.parse(str(model)).key for model in raw_variant.get("models", []) or []]
        recommended = bool(winner and str(winner) == name)
        item = HumanReviewItem(
            id=item_id,
            case_id=case_id,
            variant=name,
            task=task,
            status=str(raw_variant.get("status") or ""),
            recommended=recommended,
            dry_run=dry_run,
            models=models,
            mode=raw_variant.get("mode") or comparison.get("mode"),
            strategy=raw_variant.get("strategy"),
            estimated_cost_usd=_optional_float(raw_variant.get("estimated_cost_usd")),
            actual_cost_usd=_optional_float(raw_variant.get("actual_cost_usd")),
            estimated_latency_ms=_optional_int(raw_variant.get("estimated_latency_ms")),
            latency_ms=_optional_int(raw_variant.get("latency_ms")),
            failed_checks=[str(check) for check in raw_variant.get("failed_checks", [])],
            human_questions=[str(question) for question in raw_variant.get("human_questions", [])],
            output_preview=str(raw_variant.get("output_preview") or "") if include_output_preview else "",
        )
        item.feedback_commands = _feedback_commands(
            report_path=report_path,
            case_id=case_id,
            variant=name,
            recommended=recommended,
            dry_run=dry_run,
        )
        items.append(item)
    return items


def _feedback_commands(
    *,
    report_path: str,
    case_id: str | None,
    variant: str,
    recommended: bool,
    dry_run: bool,
) -> dict[str, str]:
    base = ["crupier", "feedback", "record", "--compare-report", report_path, "--variant", variant]
    if case_id:
        base.extend(["--case-id", case_id])
    if dry_run:
        base.append("--allow-dry-run-source")
    base.extend(["--tag", "human_review"])
    if recommended:
        base.extend(["--tag", "recommended_variant"])
    return {
        "accept": shlex.join([*base, "--rating", "5", "--verdict", "accept"]),
        "needs_work": shlex.join([*base, "--rating", "2", "--verdict", "needs_work"]),
        "reject": shlex.join([*base, "--rating", "1", "--verdict", "reject"]),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_verdict(verdict: str) -> str:
    value = str(verdict or "unknown")
    if value not in VERDICTS:
        raise CrupierError("Feedback verdict must be one of: " + ", ".join(sorted(VERDICTS)))
    return value


def _normalize_models(models: list[str]) -> list[str]:
    normalized: list[str] = []
    for model in models:
        key = ModelRef.parse(str(model)).key
        if key not in normalized:
            normalized.append(key)
    return normalized


def _normalize_tags(tags: list[str]) -> list[str]:
    normalized: list[str] = []
    for tag in tags:
        clean = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(tag).strip())[:64]
        if clean and clean not in normalized:
            normalized.append(clean)
    return normalized


def _score_delta(avg_rating: float, verdicts: dict[str, int]) -> float:
    rating_delta = (avg_rating - 3.0) * 2.0
    judged = int(verdicts.get("accept", 0) + verdicts.get("reject", 0) + verdicts.get("needs_work", 0))
    verdict_delta = 0.0
    if judged:
        verdict_delta = (
            verdicts.get("accept", 0) - verdicts.get("reject", 0) - (0.5 * verdicts.get("needs_work", 0))
        ) / judged * 2.0
    return round(max(-8.0, min(8.0, rating_delta + verdict_delta)), 2)


def _score_status(score_delta: float) -> str:
    if score_delta >= 1.0:
        return "positive"
    if score_delta <= -1.0:
        return "negative"
    return "neutral"


def _top_tags(tags: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"tag": tag, "count": count}
        for tag, count in sorted(tags.items(), key=lambda item: (-item[1], item[0]))[:10]
    ]


def _route_models(route: dict[str, Any]) -> list[str]:
    models: list[str] = []
    for step in route.get("steps", []) or []:
        for model in [step.get("model"), *(step.get("models") or [])]:
            if model and model not in models:
                models.append(str(model))
    return models


def _truncate_note(note: str) -> str:
    compact = " ".join(str(note or "").split())
    return compact[:1000]


def _redact(text: str) -> str:
    redacted = text
    for pattern, replacement in _SECRET_REPLACERS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


_SECRET_REPLACERS = (
    (re.compile(("s" + "k-") + r"[A-Za-z0-9_\-]{10,}"), "[redacted]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{12,}", re.IGNORECASE), r"\1[redacted]"),
    (re.compile(r"([A-Z][A-Z0-9_]*_API_KEY=)[^\s]+"), r"\1[redacted]"),
)
