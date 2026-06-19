"""Routing eval runner.

Evals are intentionally product-facing: they check whether a planned route
matches expectations a human maintainer would care about, not only whether the
code executes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import RoutePlan


BUILTIN_ROUTING_EVALS: list[dict[str, Any]] = [
    {
        "id": "fast_uses_single",
        "task": "Answer this short product question in one sentence.",
        "mode": "fast",
        "expect": {"strategy": "single", "max_models": 1},
    },
    {
        "id": "research_uses_fusion_when_possible",
        "task": "Compare two agent architectures, find tradeoffs, and identify blind spots.",
        "mode": "research",
        "expect": {"strategy_in": ["fusion", "single"], "min_models": 1, "max_models": 5},
    },
    {
        "id": "private_keeps_local_first_strategy",
        "task": "Route a confidential internal planning request.",
        "mode": "private",
        "expect": {"strategy": "local_first", "roles_include": ["primary"]},
    },
    {
        "id": "structured_prefers_cascade",
        "task": "Extract these fields as JSON: name, date, total.",
        "mode": "structured",
        "expect": {"strategy_in": ["cascade", "single"], "roles_include": ["primary"]},
    },
    {
        "id": "agentic_high_risk_uses_critique",
        "task": "Plan a code-changing agent step with tool use and rollback risks.",
        "mode": "agentic",
        "constraints": {"risk_level": "high"},
        "expect": {"strategy_in": ["critique_repair", "single"], "min_models": 1},
    },
]


@dataclass(slots=True)
class EvalCase:
    id: str
    task: str
    input: Any = None
    mode: str | None = None
    strategy: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    expect: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        return cls(
            id=str(data["id"]),
            task=str(data["task"]),
            input=data.get("input"),
            mode=data.get("mode"),
            strategy=data.get("strategy"),
            constraints=dict(data.get("constraints", {})),
            expect=dict(data.get("expect", {})),
            notes=data.get("notes", ""),
        )


@dataclass(slots=True)
class EvalCaseResult:
    id: str
    status: str
    ok: bool
    strategy: str | None = None
    models: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    reason: str = ""
    summary: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalRunReport:
    name: str
    orchestrator_mode: str
    total: int
    passed: int
    failed: int
    results: list[EvalCaseResult]
    warnings: list[str] = field(default_factory=list)
    written_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        data["results"] = [item.to_dict() for item in self.results]
        return data


@dataclass(slots=True)
class CompareVariant:
    name: str
    mode: str | None = None
    strategy: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompareVariant":
        return cls(
            name=str(data.get("name") or data.get("model") or "variant"),
            mode=data.get("mode"),
            strategy=data.get("strategy"),
            constraints=dict(data.get("constraints", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CompareVariantResult:
    name: str
    status: str
    ok: bool
    mode: str | None = None
    strategy: str | None = None
    models: list[str] = field(default_factory=list)
    estimated_cost_usd: float | None = None
    actual_cost_usd: float | None = None
    estimated_latency_ms: int | None = None
    latency_ms: int | None = None
    output_preview: str = ""
    checks: dict[str, Any] = field(default_factory=dict)
    failed_checks: list[str] = field(default_factory=list)
    route_reason: str = ""
    human_questions: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CompareRunReport:
    task: str
    dry_run: bool
    total: int
    passed: int
    failed: int
    variants: list[CompareVariantResult]
    winner: str | None = None
    recommendation: str = ""
    warnings: list[str] = field(default_factory=list)
    written_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        data["variants"] = [item.to_dict() for item in self.variants]
        return data


@dataclass(slots=True)
class CompareDatasetCaseResult:
    id: str
    ok: bool
    winner: str | None
    task: str
    mode: str | None = None
    comparison: CompareRunReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ok": self.ok,
            "winner": self.winner,
            "task": self.task,
            "mode": self.mode,
            "comparison": self.comparison.to_dict() if self.comparison else None,
        }


@dataclass(slots=True)
class CompareDatasetModelScore:
    model: str
    mode: str
    appearances: int
    passed: int
    wins: int
    runs: int = 1
    avg_estimated_cost_usd: float | None = None
    avg_actual_cost_usd: float | None = None
    avg_estimated_latency_ms: int | None = None
    avg_latency_ms: int | None = None
    score_delta: float = 0.0
    score_key: str = ""
    confidence: str = "low"
    trend: str = "current"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CompareDatasetReport:
    name: str
    dry_run: bool
    total_cases: int
    passed_cases: int
    failed_cases: int
    cases: list[CompareDatasetCaseResult]
    model_scores: list[CompareDatasetModelScore]
    apply_report: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    history_path: str | None = None
    written_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.failed_cases == 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        data["cases"] = [item.to_dict() for item in self.cases]
        data["model_scores"] = [item.to_dict() for item in self.model_scores]
        return data


@dataclass(slots=True)
class CompareHistoryReport:
    total_runs: int
    model_scores: list[CompareDatasetModelScore]
    last_run_at: str | None = None
    apply_report: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_runs": self.total_runs,
            "last_run_at": self.last_run_at,
            "model_scores": [item.to_dict() for item in self.model_scores],
            "apply_report": self.apply_report,
            "warnings": self.warnings,
        }


class RoutingEvalRunner:
    def __init__(self, client: Any):
        self.client = client

    def run(
        self,
        *,
        dataset: str | Path | None = None,
        write_report: bool = False,
    ) -> EvalRunReport:
        name, cases = load_eval_cases(dataset)
        results = [self._run_case(case) for case in cases]
        passed = sum(1 for item in results if item.ok)
        failed = len(results) - passed
        report = EvalRunReport(
            name=name,
            orchestrator_mode=self.client.config.orchestrator.mode,
            total=len(results),
            passed=passed,
            failed=failed,
            results=results,
            warnings=[],
        )
        if write_report:
            report.written_path = str(write_eval_report(self.client.config.evals_dir, report))
        return report

    def compare(
        self,
        *,
        task: str,
        input: Any = None,
        mode: str | None = None,
        strategy: str | None = None,
        constraints: dict[str, Any] | None = None,
        variants: list[CompareVariant] | None = None,
        response_schema: Any = None,
        expect_contains: list[str] | None = None,
        dry_run: bool = True,
        write_report: bool = False,
    ) -> CompareRunReport:
        base_constraints = dict(constraints or {})
        base_constraints.setdefault("store_prompt", False)
        base_constraints.setdefault("store_response", False)
        variants = variants or [CompareVariant(name="default")]
        results = [
            self._run_compare_variant(
                task=task,
                input=input,
                base_mode=mode,
                base_strategy=strategy,
                base_constraints=base_constraints,
                variant=variant,
                response_schema=response_schema,
                expect_contains=expect_contains or [],
                dry_run=dry_run,
            )
            for variant in variants
        ]
        passed = sum(1 for item in results if item.ok)
        failed = len(results) - passed
        winner, recommendation = recommend_compare_winner(results)
        report = CompareRunReport(
            task=task,
            dry_run=dry_run,
            total=len(results),
            passed=passed,
            failed=failed,
            variants=results,
            winner=winner,
            recommendation=recommendation,
            warnings=[],
        )
        if write_report:
            report.written_path = str(write_compare_report(self.client.config.evals_dir, report))
        return report

    def compare_dataset(
        self,
        *,
        dataset: str | Path | None = None,
        variants: list[CompareVariant] | None = None,
        constraints: dict[str, Any] | None = None,
        expect_contains: list[str] | None = None,
        dry_run: bool = True,
        apply: bool = False,
        min_count: int = 1,
        min_confidence: str = "low",
        record_history: bool = False,
        write_report: bool = False,
    ) -> CompareDatasetReport:
        name, cases = load_eval_cases(dataset)
        case_results: list[CompareDatasetCaseResult] = []
        for case in cases:
            merged_constraints = {**dict(case.constraints), **dict(constraints or {})}
            comparison = self.compare(
                task=case.task,
                input=case.input,
                mode=case.mode,
                strategy=case.strategy,
                constraints=merged_constraints,
                variants=variants,
                expect_contains=expect_contains,
                dry_run=dry_run,
                write_report=False,
            )
            case_results.append(
                CompareDatasetCaseResult(
                    id=case.id,
                    ok=bool(comparison.winner),
                    winner=comparison.winner,
                    task=case.task,
                    mode=case.mode,
                    comparison=comparison,
                )
            )

        model_scores = aggregate_compare_scores(case_results)
        passed_cases = sum(1 for item in case_results if item.ok)
        report = CompareDatasetReport(
            name=name,
            dry_run=dry_run,
            total_cases=len(case_results),
            passed_cases=passed_cases,
            failed_cases=len(case_results) - passed_cases,
            cases=case_results,
            model_scores=model_scores,
            warnings=[],
        )
        if record_history:
            report.history_path = str(write_compare_history(self.client.config.evals_dir, report))
        if apply:
            report.apply_report = apply_compare_scores_to_registry(
                model_scores,
                self.client.registry,
                min_count=min_count,
                min_confidence=min_confidence,
                dry_run=False,
            )
        if write_report:
            report.written_path = str(write_compare_dataset_report(self.client.config.evals_dir, report))
        return report

    def history(
        self,
        *,
        model: str | None = None,
        mode: str | None = None,
        apply: bool = False,
        min_count: int = 3,
        min_confidence: str = "medium",
        dry_run: bool = True,
    ) -> CompareHistoryReport:
        report = summarize_compare_history(self.client.config.evals_dir, model=model, mode=mode)
        if apply:
            report.apply_report = apply_compare_scores_to_registry(
                report.model_scores,
                self.client.registry,
                min_count=min_count,
                min_confidence=min_confidence,
                dry_run=dry_run,
            )
        return report

    def _run_case(self, case: EvalCase) -> EvalCaseResult:
        try:
            result = self.client.deal(
                task=case.task,
                input=case.input,
                mode=case.mode,
                strategy=case.strategy,
                constraints=case.constraints,
                trace="summary",
                dry_run=True,
            )
        except Exception as exc:  # noqa: BLE001 - eval reports should capture route failures
            return EvalCaseResult(
                id=case.id,
                status="fail",
                ok=False,
                failed_checks=["route_error"],
                error=str(exc),
            )

        plan = result.route
        if plan is None:
            return EvalCaseResult(
                id=case.id,
                status="fail",
                ok=False,
                failed_checks=["no_route_plan"],
            )
        failed_checks = evaluate_expectations(plan, case.expect)
        return EvalCaseResult(
            id=case.id,
            status="pass" if not failed_checks else "fail",
            ok=not failed_checks,
            strategy=plan.strategy,
            models=plan.models,
            failed_checks=failed_checks,
            reason=plan.reason,
            summary=plan.summary,
        )

    def _run_compare_variant(
        self,
        *,
        task: str,
        input: Any,
        base_mode: str | None,
        base_strategy: str | None,
        base_constraints: dict[str, Any],
        variant: CompareVariant,
        response_schema: Any,
        expect_contains: list[str],
        dry_run: bool,
    ) -> CompareVariantResult:
        constraints = {**base_constraints, **variant.constraints}
        try:
            result = self.client.deal(
                task=task,
                input=input,
                mode=variant.mode or base_mode,
                strategy=variant.strategy or base_strategy,
                constraints=constraints,
                response_schema=response_schema,
                trace="summary",
                dry_run=dry_run,
            )
        except Exception as exc:  # noqa: BLE001 - compare should report every variant boundary
            return CompareVariantResult(
                name=variant.name,
                status="fail",
                ok=False,
                mode=variant.mode or base_mode,
                failed_checks=["route_or_execution_error"],
                error=str(exc),
                human_questions=_human_compare_questions(dry_run=dry_run),
            )

        plan = result.route
        if plan is None:
            return CompareVariantResult(
                name=variant.name,
                status="fail",
                ok=False,
                mode=variant.mode or base_mode,
                failed_checks=["no_route_plan"],
                output_preview=_preview(result.output_text),
                human_questions=_human_compare_questions(dry_run=dry_run),
            )
        checks = compare_output_checks(result.output_text, expect_contains=expect_contains, dry_run=dry_run)
        failed_checks = [key for key, value in checks.items() if value is False]
        return CompareVariantResult(
            name=variant.name,
            status="pass" if not failed_checks else "fail",
            ok=not failed_checks,
            mode=variant.mode or base_mode,
            strategy=plan.strategy,
            models=plan.models,
            estimated_cost_usd=plan.estimated_cost.estimated_usd,
            actual_cost_usd=None if dry_run else result.cost.actual_usd,
            estimated_latency_ms=plan.estimated_latency_ms,
            latency_ms=None if dry_run else result.latency_ms,
            output_preview=_preview(result.output_text),
            checks=checks,
            failed_checks=failed_checks,
            route_reason=plan.reason,
            human_questions=_human_compare_questions(dry_run=dry_run),
        )


def load_eval_cases(dataset: str | Path | None) -> tuple[str, list[EvalCase]]:
    if dataset is None:
        return "builtin-routing", [EvalCase.from_dict(item) for item in BUILTIN_ROUTING_EVALS]

    path = Path(dataset)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return path.stem, [EvalCase.from_dict(item) for item in rows]

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return path.stem, [EvalCase.from_dict(item) for item in data]
    return str(data.get("name", path.stem)), [EvalCase.from_dict(item) for item in data.get("cases", [])]


def evaluate_expectations(plan: RoutePlan, expect: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if not expect:
        return failed

    if "strategy" in expect and plan.strategy != expect["strategy"]:
        failed.append(f"strategy expected {expect['strategy']!r}, got {plan.strategy!r}")
    if "strategy_in" in expect and plan.strategy not in set(expect["strategy_in"]):
        failed.append(f"strategy expected one of {expect['strategy_in']!r}, got {plan.strategy!r}")
    if "risk_level" in expect and plan.risk_level != expect["risk_level"]:
        failed.append(f"risk_level expected {expect['risk_level']!r}, got {plan.risk_level!r}")

    models = plan.models
    providers = {model.split(":", 1)[0] for model in models}
    if "models_include" in expect:
        for model in expect["models_include"]:
            if model not in models:
                failed.append(f"missing expected model {model!r}")
    if "models_exclude" in expect:
        for model in expect["models_exclude"]:
            if model in models:
                failed.append(f"unexpected model {model!r}")
    if "providers_include" in expect:
        for provider in expect["providers_include"]:
            if provider not in providers:
                failed.append(f"missing expected provider {provider!r}")
    if "providers_exclude" in expect:
        for provider in expect["providers_exclude"]:
            if provider in providers:
                failed.append(f"unexpected provider {provider!r}")
    if "min_models" in expect and len(models) < int(expect["min_models"]):
        failed.append(f"expected at least {expect['min_models']} models, got {len(models)}")
    if "max_models" in expect and len(models) > int(expect["max_models"]):
        failed.append(f"expected at most {expect['max_models']} models, got {len(models)}")

    roles = {step.role for step in plan.steps}
    if "roles_include" in expect:
        for role in expect["roles_include"]:
            if role not in roles:
                failed.append(f"missing expected role {role!r}")
    if "roles_exclude" in expect:
        for role in expect["roles_exclude"]:
            if role in roles:
                failed.append(f"unexpected role {role!r}")
    return failed


def compare_output_checks(output: str, *, expect_contains: list[str], dry_run: bool) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "route_planned": True,
    }
    if dry_run:
        checks["provider_execution"] = "skipped"
    else:
        checks["non_empty_output"] = bool(output.strip())
    for item in expect_contains:
        checks[f"contains:{item}"] = item.lower() in output.lower()
    return checks


def recommend_compare_winner(results: list[CompareVariantResult]) -> tuple[str | None, str]:
    passed = [item for item in results if item.ok]
    if not passed:
        return None, "No variant passed the deterministic checks; inspect failures before choosing."

    def sort_key(item: CompareVariantResult) -> tuple[float, int, int, str]:
        cost = item.actual_cost_usd if item.actual_cost_usd is not None else item.estimated_cost_usd
        latency = item.latency_ms if item.latency_ms is not None else item.estimated_latency_ms
        return (
            float(cost if cost is not None else 999.0),
            int(latency if latency is not None else 999_999),
            len(item.models),
            item.name,
        )

    winner = sorted(passed, key=sort_key)[0]
    cost = winner.actual_cost_usd if winner.actual_cost_usd is not None else winner.estimated_cost_usd
    latency = winner.latency_ms if winner.latency_ms is not None else winner.estimated_latency_ms
    return (
        winner.name,
        f"{winner.name} passed checks with cost={cost} latency_ms={latency}; human review should confirm output quality.",
    )


def aggregate_compare_scores(case_results: list[CompareDatasetCaseResult]) -> list[CompareDatasetModelScore]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for case in case_results:
        if case.comparison is None:
            continue
        for variant in case.comparison.variants:
            mode = variant.mode or case.mode or "overall"
            for model in variant.models:
                group = groups.setdefault(
                    (model, mode),
                    {
                        "model": model,
                        "mode": mode,
                        "appearances": 0,
                        "passed": 0,
                        "wins": 0,
                        "estimated_costs": [],
                        "actual_costs": [],
                        "estimated_latencies": [],
                        "latencies": [],
                    },
                )
                group["appearances"] += 1
                if variant.ok:
                    group["passed"] += 1
                if case.winner == variant.name:
                    group["wins"] += 1
                _append_number(group["estimated_costs"], variant.estimated_cost_usd)
                _append_number(group["actual_costs"], variant.actual_cost_usd)
                _append_number(group["estimated_latencies"], variant.estimated_latency_ms)
                _append_number(group["latencies"], variant.latency_ms)

    scores: list[CompareDatasetModelScore] = []
    for group in groups.values():
        appearances = int(group["appearances"])
        passed = int(group["passed"])
        wins = int(group["wins"])
        score_delta = _compare_score_delta(appearances=appearances, passed=passed, wins=wins)
        mode = str(group["mode"])
        scores.append(
            CompareDatasetModelScore(
                model=str(group["model"]),
                mode=mode,
                appearances=appearances,
                passed=passed,
                wins=wins,
                avg_estimated_cost_usd=_avg_float(group["estimated_costs"]),
                avg_actual_cost_usd=_avg_float(group["actual_costs"]),
                avg_estimated_latency_ms=_avg_int(group["estimated_latencies"]),
                avg_latency_ms=_avg_int(group["latencies"]),
                score_delta=score_delta,
                score_key=f"eval:{mode}",
                confidence=_confidence_from_appearances(appearances),
                trend="current",
            )
        )
    return sorted(scores, key=lambda item: (item.model, item.mode))


def apply_compare_scores_to_registry(
    model_scores: list[CompareDatasetModelScore],
    registry: Any,
    *,
    min_count: int = 1,
    min_confidence: str = "low",
    dry_run: bool = False,
) -> dict[str, Any]:
    min_confidence = _validate_confidence(min_confidence)
    updated: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    written_files: list[str] = []
    for score in model_scores:
        if score.appearances < min_count:
            skipped.append({"model": score.model, "reason": f"appearances below min_count={min_count}"})
            continue
        if _confidence_rank(score.confidence) < _confidence_rank(min_confidence):
            skipped.append(
                {
                    "model": score.model,
                    "reason": f"confidence {score.confidence} below min_confidence={min_confidence}",
                }
            )
            continue
        try:
            card = registry.get(score.model)
        except Exception as exc:  # noqa: BLE001 - aggregate apply should report all missing cards
            skipped.append({"model": score.model, "reason": str(exc)})
            continue
        old_score = card.local_eval_scores.get(score.score_key)
        card.local_eval_scores[score.score_key] = score.score_delta
        if not dry_run:
            path = registry.save_card(card, dry_run=False)
            if path:
                written_files.append(path)
        updated.append(
            {
                "model": score.model,
                "mode": score.mode,
                "score_key": score.score_key,
                "old_score": old_score,
                "new_score": score.score_delta,
                "appearances": score.appearances,
                "passed": score.passed,
                "wins": score.wins,
                "confidence": score.confidence,
                "trend": score.trend,
            }
        )
    return {
        "dry_run": dry_run,
        "min_count": min_count,
        "min_confidence": min_confidence,
        "updated": updated,
        "skipped": skipped,
        "written_files": written_files,
    }


def write_eval_report(root: Path, report: EvalRunReport) -> Path:
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = runs_dir / f"routing_{timestamp}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_compare_report(root: Path, report: CompareRunReport) -> Path:
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = runs_dir / f"compare_{timestamp}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_compare_dataset_report(root: Path, report: CompareDatasetReport) -> Path:
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = runs_dir / f"compare_dataset_{timestamp}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_compare_history(root: Path, report: CompareDatasetReport) -> Path:
    history_dir = root / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / "compare_runs.jsonl"
    record = {
        "schema_version": 1,
        "run_id": f"cmp_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset": report.name,
        "dry_run": report.dry_run,
        "total_cases": report.total_cases,
        "passed_cases": report.passed_cases,
        "failed_cases": report.failed_cases,
        "case_ids": [case.id for case in report.cases],
        "model_scores": [score.to_dict() for score in report.model_scores],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def summarize_compare_history(root: Path, *, model: str | None = None, mode: str | None = None) -> CompareHistoryReport:
    history_path = root / "history" / "compare_runs.jsonl"
    records = _read_compare_history(history_path)
    if not records:
        return CompareHistoryReport(total_runs=0, model_scores=[], warnings=["No compare history found."])

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    last_run_at: str | None = None
    for record in records:
        created_at = record.get("created_at")
        if created_at and (last_run_at is None or str(created_at) > last_run_at):
            last_run_at = str(created_at)
        for raw_score in record.get("model_scores", []):
            item_model = str(raw_score.get("model", ""))
            item_mode = str(raw_score.get("mode", "overall"))
            if model and item_model != model:
                continue
            if mode and item_mode != mode:
                continue
            group = groups.setdefault(
                (item_model, item_mode),
                {
                    "model": item_model,
                    "mode": item_mode,
                    "runs": 0,
                    "appearances": 0,
                    "passed": 0,
                    "wins": 0,
                    "estimated_costs": [],
                    "actual_costs": [],
                    "estimated_latencies": [],
                    "latencies": [],
                    "score_values": [],
                },
            )
            appearances = int(raw_score.get("appearances", 0) or 0)
            group["runs"] += 1
            group["appearances"] += appearances
            group["passed"] += int(raw_score.get("passed", 0) or 0)
            group["wins"] += int(raw_score.get("wins", 0) or 0)
            _append_weighted(group["estimated_costs"], raw_score.get("avg_estimated_cost_usd"), appearances)
            _append_weighted(group["actual_costs"], raw_score.get("avg_actual_cost_usd"), appearances)
            _append_weighted(group["estimated_latencies"], raw_score.get("avg_estimated_latency_ms"), appearances)
            _append_weighted(group["latencies"], raw_score.get("avg_latency_ms"), appearances)
            _append_number(group["score_values"], raw_score.get("score_delta"))

    scores: list[CompareDatasetModelScore] = []
    for group in groups.values():
        score_values = list(group["score_values"])
        score_delta = round(sum(score_values) / len(score_values), 2) if score_values else 0.0
        appearances = int(group["appearances"])
        mode_value = str(group["mode"])
        scores.append(
            CompareDatasetModelScore(
                model=str(group["model"]),
                mode=mode_value,
                appearances=appearances,
                passed=int(group["passed"]),
                wins=int(group["wins"]),
                runs=int(group["runs"]),
                avg_estimated_cost_usd=_avg_weighted(group["estimated_costs"]),
                avg_actual_cost_usd=_avg_weighted(group["actual_costs"]),
                avg_estimated_latency_ms=_avg_weighted_int(group["estimated_latencies"]),
                avg_latency_ms=_avg_weighted_int(group["latencies"]),
                score_delta=score_delta,
                score_key=f"eval:{mode_value}",
                confidence=_confidence_from_appearances(appearances),
                trend=_trend(score_values),
            )
        )
    scores.sort(key=lambda item: (item.model, item.mode))
    return CompareHistoryReport(total_runs=len(records), model_scores=scores, last_run_at=last_run_at)


def _human_compare_questions(*, dry_run: bool) -> list[str]:
    questions = [
        "Would a maintainer ship this result for the project?",
        "Is the cost/latency tradeoff justified against the next-best variant?",
    ]
    if dry_run:
        questions.append("Should this variant be executed with --no-dry-run before recording feedback?")
    else:
        questions.append("Did the real output satisfy the project-specific quality bar?")
    return questions


def _preview(text: str, limit: int = 320) -> str:
    compact = " ".join((text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _append_number(values: list[float], value: Any) -> None:
    if isinstance(value, int | float):
        values.append(float(value))


def _avg_float(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 8)


def _avg_int(values: list[float]) -> int | None:
    if not values:
        return None
    return int(round(sum(values) / len(values)))


def _compare_score_delta(*, appearances: int, passed: int, wins: int) -> float:
    if appearances <= 0:
        return 0.0
    pass_rate = passed / appearances
    win_rate = wins / appearances
    score = ((pass_rate - 0.5) * 4.0) + (win_rate * 4.0)
    return round(max(-8.0, min(8.0, score)), 2)


def _read_compare_history(path: Path) -> list[dict[str, Any]]:
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


def _append_weighted(values: list[tuple[float, int]], value: Any, weight: int) -> None:
    if isinstance(value, int | float) and weight > 0:
        values.append((float(value), weight))


def _avg_weighted(values: list[tuple[float, int]]) -> float | None:
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return None
    total = sum(value * weight for value, weight in values)
    return round(total / total_weight, 8)


def _avg_weighted_int(values: list[tuple[float, int]]) -> int | None:
    value = _avg_weighted(values)
    return int(round(value)) if value is not None else None


def _confidence_from_appearances(appearances: int) -> str:
    if appearances >= 10:
        return "high"
    if appearances >= 3:
        return "medium"
    return "low"


def _confidence_rank(confidence: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(confidence, 0)


def _validate_confidence(confidence: str) -> str:
    value = str(confidence or "low").lower()
    if value not in {"low", "medium", "high"}:
        raise ValueError("confidence must be low, medium, or high")
    return value


def _trend(score_values: list[float]) -> str:
    if len(score_values) < 2:
        return "insufficient"
    previous = sum(score_values[:-1]) / (len(score_values) - 1)
    delta = score_values[-1] - previous
    if delta >= 0.75:
        return "improving"
    if delta <= -0.75:
        return "declining"
    return "stable"
