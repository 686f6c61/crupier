"""Production shadow and canary routing experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from statistics import NormalDist, mean
from threading import RLock
from time import monotonic, sleep
from typing import Any
from uuid import uuid4

from .config import ExperimentSettings
from .errors import CrupierError
from .models import CrupierResult, ExperimentObservation
from .state import SQLiteStateStore
from .tools import normalize_tools

ExperimentEvaluator = Callable[
    [CrupierResult | None, CrupierResult],
    dict[str, Any],
]


class _ExperimentStateUnavailable(CrupierError):
    """Internal signal used to fail closed to baseline traffic."""


@dataclass(slots=True)
class PromotionReport:
    experiment: str
    eligible: bool
    sample_count: int
    gates: dict[str, bool | None]
    metrics: dict[str, Any]
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperimentReport:
    experiment: str
    status: str
    observations: int
    sampled: int
    completed: int
    failed: int
    cohorts: dict[str, int]
    metrics: dict[str, Any]
    promotion: PromotionReport

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["promotion"] = self.promotion.to_dict()
        return data


class ExperimentManager:
    def __init__(
        self,
        client: Any,
        path: str,
        *,
        evaluator: ExperimentEvaluator | None = None,
    ):
        self.client = client
        self.store = SQLiteStateStore(path)
        self.evaluator = evaluator
        workers = max(
            [settings.max_concurrency for settings in client.config.experiments.values()]
            or [2]
        )
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="crupier-shadow",
        )
        self._futures: dict[str, Future[Any]] = {}
        self._lock = RLock()

    def run(self, experiment: str, request_kwargs: dict[str, Any]) -> CrupierResult:
        settings = self._settings(experiment)
        try:
            runtime_status = self._runtime_status(experiment)
        except Exception as exc:  # noqa: BLE001 - the control plane must not break live traffic
            return self._serve_baseline_without_state(settings, request_kwargs, exc)
        if not settings.enabled or runtime_status in {"paused", "rolled_back"}:
            result = self.client.deal(**request_kwargs)
            try:
                result.experiment = self._record_control(
                    settings,
                    result,
                    sampled=False,
                    status="disabled",
                )
            except _ExperimentStateUnavailable as exc:
                result.experiment = self._transient_observation(settings, exc)
                result.warnings.append(result.experiment.warnings[-1])
            return result

        sticky_value = _sticky_value(settings, request_kwargs)
        sampled = runtime_status == "promoted" or _sample(
            experiment,
            sticky_value,
            settings.sample_rate,
        )
        try:
            if runtime_status == "promoted":
                return self._run_promoted(settings, request_kwargs, sticky_value)
            if settings.traffic == "canary":
                return self._run_canary(
                    settings,
                    request_kwargs,
                    sampled=sampled,
                    sticky_value=sticky_value,
                )
            return self._run_shadow(
                settings,
                request_kwargs,
                sampled=sampled,
                sticky_value=sticky_value,
            )
        except _ExperimentStateUnavailable as exc:
            return self._serve_baseline_without_state(settings, request_kwargs, exc)

    def observation(self, observation_id: str) -> ExperimentObservation:
        record = self.store.get("experiment_observation", observation_id)
        return _observation(dict(record.payload["observation"]))

    def wait(self, observation_id: str, *, timeout_seconds: float = 30.0) -> ExperimentObservation:
        deadline = monotonic() + max(0.0, timeout_seconds)
        while monotonic() <= deadline:
            observation = self.observation(observation_id)
            if observation.status not in {"scheduled", "running"}:
                return observation
            sleep(0.01)
        return self.observation(observation_id)

    def record_evaluation(
        self,
        observation_id: str,
        checks: dict[str, Any],
        *,
        actor: str,
    ) -> ExperimentObservation:
        if not actor.strip():
            raise CrupierError("Evaluation actor cannot be empty.")
        if not checks:
            raise CrupierError("Evaluation checks cannot be empty.")
        if any(not isinstance(key, str) or not key.strip() for key in checks):
            raise CrupierError("Evaluation check names must be non-empty strings.")
        record = self.store.get("experiment_observation", observation_id)
        observation = _observation(dict(record.payload["observation"]))
        if observation.status in {"running", "scheduled"}:
            raise CrupierError("A running experiment observation cannot be evaluated yet.")
        observation.checks.update(_validated_checks(checks))
        self.store.transition(
            kind="experiment_observation",
            record_id=observation_id,
            expected_statuses={record.status},
            status=record.status,
            payload={
                "experiment": observation.experiment,
                "observation": observation.to_dict(),
            },
            expires_at=None,
            event="evaluated",
            actor=actor.strip(),
        )
        return observation

    def report(self, experiment: str) -> ExperimentReport:
        settings = self._settings(experiment)
        records = [
            record
            for record in self.store.list("experiment_observation", limit=10_000)
            if record.payload.get("experiment") == experiment
        ]
        observations = [_observation(dict(record.payload["observation"])) for record in records]
        completed = [
            item
            for item in observations
            if item.status in {"completed", "control_completed", "candidate_completed"}
        ]
        failed = [item for item in observations if item.status == "failed"]
        cohorts: dict[str, int] = {}
        for item in observations:
            cohorts[item.cohort] = cohorts.get(item.cohort, 0) + 1
        evidence = [*completed, *failed]
        metrics = _aggregate_metrics(evidence)
        promotion = _promotion_report(settings, evidence, metrics)
        return ExperimentReport(
            experiment=experiment,
            status=self._runtime_status(experiment),
            observations=len(observations),
            sampled=sum(item.sampled for item in observations),
            completed=len(completed),
            failed=len(failed),
            cohorts=cohorts,
            metrics=metrics,
            promotion=promotion,
        )

    def promote(
        self,
        experiment: str,
        *,
        actor: str,
        force: bool = False,
    ) -> ExperimentReport:
        if not actor.strip():
            raise CrupierError("Promotion actor cannot be empty.")
        report = self.report(experiment)
        if not force and not report.promotion.eligible:
            raise CrupierError(
                "Experiment is not eligible for promotion: "
                + "; ".join(report.promotion.reasons)
            )
        current = self._runtime_record(experiment)
        payload = dict(current.payload)
        payload.update(
            {
                "promoted_by": actor.strip(),
                "promotion_report": report.promotion.to_dict(),
            }
        )
        self.store.transition(
            kind="experiment",
            record_id=experiment,
            expected_statuses={"active", "rolled_back", "paused"},
            status="promoted",
            payload=payload,
            expires_at=None,
            event="promoted",
            actor=actor.strip(),
        )
        return self.report(experiment)

    def rollback(self, experiment: str, *, actor: str, reason: str) -> ExperimentReport:
        if not actor.strip():
            raise CrupierError("Rollback actor cannot be empty.")
        current = self._runtime_record(experiment)
        payload = dict(current.payload)
        payload.update(
            {
                "rolled_back_by": actor.strip(),
                "rollback_reason": reason.strip()[:2_000],
            }
        )
        self.store.transition(
            kind="experiment",
            record_id=experiment,
            expected_statuses={"promoted", "active", "paused"},
            status="rolled_back",
            payload=payload,
            expires_at=None,
            event="rolled_back",
            actor=actor.strip(),
        )
        return self.report(experiment)

    def pause(self, experiment: str, *, actor: str) -> ExperimentReport:
        if not actor.strip():
            raise CrupierError("Pause actor cannot be empty.")
        current = self._runtime_record(experiment)
        self.store.transition(
            kind="experiment",
            record_id=experiment,
            expected_statuses={"active"},
            status="paused",
            payload=dict(current.payload),
            expires_at=None,
            event="paused",
            actor=actor.strip(),
        )
        return self.report(experiment)

    def resume(self, experiment: str, *, actor: str) -> ExperimentReport:
        if not actor.strip():
            raise CrupierError("Resume actor cannot be empty.")
        current = self._runtime_record(experiment)
        self.store.transition(
            kind="experiment",
            record_id=experiment,
            expected_statuses={"paused", "rolled_back"},
            status="active",
            payload=dict(current.payload),
            expires_at=None,
            event="resumed",
            actor=actor.strip(),
        )
        return self.report(experiment)

    def close(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def _run_shadow(
        self,
        settings: ExperimentSettings,
        request_kwargs: dict[str, Any],
        *,
        sampled: bool,
        sticky_value: str,
    ) -> CrupierResult:
        observation = self._create_observation(
            settings,
            cohort="shadow" if sampled else "control",
            sampled=sampled,
            sticky_value=sticky_value,
        )
        if not sampled:
            try:
                primary = self.client.deal(**request_kwargs)
            except Exception as exc:
                self._fail_observation(observation, exc)
                raise
            completed = self._complete_control(observation, settings, primary)
            primary.experiment = completed
            return primary

        candidate_kwargs, warnings = self._candidate_kwargs(
            settings,
            request_kwargs,
            shadow=True,
        )
        candidate_future = self._executor.submit(self.client.deal, **candidate_kwargs)
        try:
            primary = self.client.deal(**request_kwargs)
        except Exception as exc:
            self._fail_observation(observation, exc)
            raise
        if settings.execution == "async":
            scheduled = observation
            scheduled.status = "scheduled"
            scheduled.primary = _result_metrics(primary, store_output=False)
            scheduled.warnings.extend(warnings)
            self._update_observation(scheduled, expected={"running"}, status="scheduled")
            future = self._executor.submit(
                self._finish_shadow_async,
                scheduled,
                settings,
                primary,
                candidate_future,
            )
            with self._lock:
                self._futures[observation.observation_id] = future
            primary.experiment = scheduled
            return primary
        try:
            candidate = candidate_future.result()
        except Exception as exc:  # noqa: BLE001 - canary failures must fall back regardless of provider exception type
            failed = self._fail_shadow_candidate(
                observation,
                settings,
                primary,
                exc,
                warnings,
            )
            primary.experiment = failed
            return primary
        completed = self._complete_pair(
            observation,
            settings,
            primary,
            candidate,
            warnings=warnings,
        )
        primary.experiment = completed
        if warning := self._maybe_auto_promote(settings):
            primary.warnings.append(warning)
            primary.experiment.warnings.append(warning)
        return primary

    def _run_canary(
        self,
        settings: ExperimentSettings,
        request_kwargs: dict[str, Any],
        *,
        sampled: bool,
        sticky_value: str,
    ) -> CrupierResult:
        observation = self._create_observation(
            settings,
            cohort="candidate" if sampled else "control",
            sampled=sampled,
            sticky_value=sticky_value,
        )
        if not sampled:
            try:
                primary = self.client.deal(**request_kwargs)
            except Exception as exc:
                self._fail_observation(observation, exc)
                raise
            completed = self._complete_control(observation, settings, primary)
            primary.experiment = completed
            return primary

        candidate_kwargs, warnings = self._candidate_kwargs(
            settings,
            request_kwargs,
            shadow=False,
        )
        try:
            candidate = self.client.deal(**candidate_kwargs)
        except Exception as exc:  # noqa: BLE001 - promoted candidates retain the same baseline fallback contract
            try:
                primary = self.client.deal(**request_kwargs)
            except Exception as primary_exc:
                self._fail_observation(
                    observation,
                    primary_exc,
                    role="both",
                    secondary_error=exc,
                )
                raise
            failed = self._fail_canary_candidate(
                observation,
                settings,
                primary,
                exc,
                warnings,
            )
            primary.experiment = failed
            primary.warnings.append(
                "Canary candidate failed and baseline served the request: "
                f"{_redact(str(exc))}"
            )
            return primary
        completed = self._complete_candidate(
            observation,
            settings,
            candidate,
            warnings=warnings,
        )
        candidate.experiment = completed
        if warning := self._maybe_auto_promote(settings):
            candidate.warnings.append(warning)
            candidate.experiment.warnings.append(warning)
        return candidate

    def _run_promoted(
        self,
        settings: ExperimentSettings,
        request_kwargs: dict[str, Any],
        sticky_value: str,
    ) -> CrupierResult:
        observation = self._create_observation(
            settings,
            cohort="promoted",
            sampled=True,
            sticky_value=sticky_value,
        )
        candidate_kwargs, warnings = self._candidate_kwargs(
            settings,
            request_kwargs,
            shadow=False,
        )
        try:
            candidate = self.client.deal(**candidate_kwargs)
        except Exception as exc:  # noqa: BLE001 - async provider failures are recorded instead of escaping the worker
            try:
                primary = self.client.deal(**request_kwargs)
            except Exception as primary_exc:
                self._fail_observation(
                    observation,
                    primary_exc,
                    role="both",
                    secondary_error=exc,
                )
                raise
            failed = self._fail_canary_candidate(
                observation,
                settings,
                primary,
                exc,
                warnings,
            )
            primary.experiment = failed
            primary.warnings.append(
                "Promoted candidate failed and baseline served the request: "
                f"{_redact(str(exc))}"
            )
            return primary
        completed = self._complete_candidate(
            observation,
            settings,
            candidate,
            warnings=warnings,
        )
        candidate.experiment = completed
        return candidate

    def _candidate_kwargs(
        self,
        settings: ExperimentSettings,
        request_kwargs: dict[str, Any],
        *,
        shadow: bool,
    ) -> tuple[dict[str, Any], list[str]]:
        candidate = dict(request_kwargs)
        constraints = {
            **dict(request_kwargs.get("constraints") or {}),
            **settings.candidate_constraints,
            "store_prompt": False,
            "store_response": False,
        }
        if settings.candidate_models:
            constraints["allowed_models"] = list(settings.candidate_models)
            if len(settings.candidate_models) == 1:
                constraints["force_model"] = settings.candidate_models[0]
        if settings.max_shadow_cost_usd is not None and shadow:
            current = constraints.get("max_cost_usd")
            constraints["max_cost_usd"] = (
                settings.max_shadow_cost_usd
                if current is None
                else min(float(current), settings.max_shadow_cost_usd)
            )
        candidate["constraints"] = constraints
        if settings.candidate_strategy:
            candidate["strategy"] = settings.candidate_strategy
        warnings: list[str] = []
        execution = settings.execution
        tools = normalize_tools(list(request_kwargs.get("tools") or []))
        side_effecting = any(tool.side_effects or tool.requires_approval for tool in tools)
        if shadow and side_effecting and not settings.allow_side_effecting_tools:
            execution = "plan_only"
            warnings.append(
                "Shadow execution was reduced to plan_only because the request has "
                "side-effecting or approval-bound tools."
            )
        if shadow and constraints.get("requires_human_approval"):
            execution = "plan_only"
            warnings.append(
                "Shadow execution was reduced to plan_only because the route requires human approval."
            )
        if execution == "plan_only":
            candidate["dry_run"] = True
        candidate.pop("experiment", None)
        candidate.pop("approval_token", None)
        metadata = dict(candidate.get("metadata") or {})
        metadata["_crupier_experiment_role"] = "shadow" if shadow else "canary"
        candidate["metadata"] = metadata
        return candidate, warnings

    def _finish_shadow_async(
        self,
        observation: ExperimentObservation,
        settings: ExperimentSettings,
        primary: CrupierResult,
        candidate_future: Future[CrupierResult],
    ) -> None:
        try:
            candidate = candidate_future.result()
        except Exception as exc:  # noqa: BLE001 - background candidate failures must be isolated
            self._fail_shadow_candidate(
                observation,
                settings,
                primary,
                exc,
                list(observation.warnings),
            )
        else:
            self._complete_pair(
                observation,
                settings,
                primary,
                candidate,
                warnings=list(observation.warnings),
                expected={"scheduled"},
            )
            self._maybe_auto_promote(settings)
        finally:
            with self._lock:
                self._futures.pop(observation.observation_id, None)

    def _create_observation(
        self,
        settings: ExperimentSettings,
        *,
        cohort: str,
        sampled: bool,
        sticky_value: str,
    ) -> ExperimentObservation:
        observation = ExperimentObservation(
            observation_id=f"obs_{uuid4().hex[:16]}",
            experiment=settings.name,
            traffic=settings.traffic,
            cohort=cohort,
            sampled=sampled,
            status="running",
            checks={
                "sticky_key_hash": _hash(sticky_value)[:16],
                "config_hash": _hash(
                    json.dumps(asdict(settings), sort_keys=True, default=str)
                )[:16],
            },
        )
        try:
            self.store.create(
                kind="experiment_observation",
                record_id=observation.observation_id,
                status="running",
                payload={
                    "experiment": settings.name,
                    "observation": observation.to_dict(),
                },
                event="started",
            )
        except Exception as exc:
            raise _ExperimentStateUnavailable(
                "Experiment state could not be initialized."
            ) from exc
        return observation

    def _record_control(
        self,
        settings: ExperimentSettings,
        result: CrupierResult,
        *,
        sampled: bool,
        status: str,
    ) -> ExperimentObservation:
        observation = self._create_observation(
            settings,
            cohort="control",
            sampled=sampled,
            sticky_value="disabled",
        )
        observation.status = status
        observation.primary = _result_metrics(result, store_output=False)
        return self._update_observation(
            observation,
            expected={"running"},
            status=status,
        )

    def _complete_control(
        self,
        observation: ExperimentObservation,
        settings: ExperimentSettings,
        primary: CrupierResult,
    ) -> ExperimentObservation:
        observation.status = "control_completed"
        observation.primary = _result_metrics(primary, store_output=settings.store_outputs)
        observation.checks.update(_single_checks(primary))
        return self._update_observation(
            observation,
            expected={"running"},
            status="control_completed",
        )

    def _complete_candidate(
        self,
        observation: ExperimentObservation,
        settings: ExperimentSettings,
        candidate: CrupierResult,
        *,
        warnings: list[str],
    ) -> ExperimentObservation:
        observation.status = "candidate_completed"
        observation.candidate = _result_metrics(
            candidate,
            store_output=settings.store_outputs,
        )
        observation.checks.update(_single_checks(candidate))
        if self.evaluator is not None:
            try:
                evaluated = self.evaluator(None, candidate)
                observation.checks.update(_validated_checks(evaluated))
            except Exception as exc:  # noqa: BLE001 - experiment failures cannot break live traffic
                observation.warnings.append(
                    f"Experiment evaluator failed: {_redact(str(exc))}"
                )
        _extend_unique(observation.warnings, warnings)
        return self._update_observation(
            observation,
            expected={"running"},
            status="candidate_completed",
        )

    def _complete_pair(
        self,
        observation: ExperimentObservation,
        settings: ExperimentSettings,
        primary: CrupierResult,
        candidate: CrupierResult,
        *,
        warnings: list[str],
        expected: set[str] | None = None,
    ) -> ExperimentObservation:
        observation.status = "completed"
        observation.primary = _result_metrics(primary, store_output=settings.store_outputs)
        observation.candidate = _result_metrics(
            candidate,
            store_output=settings.store_outputs,
        )
        observation.diffs = _metric_diffs(observation.primary, observation.candidate)
        observation.checks.update(_pair_checks(primary, candidate))
        if self.evaluator is not None:
            try:
                evaluated = self.evaluator(primary, candidate)
                observation.checks.update(_validated_checks(evaluated))
            except Exception as exc:  # noqa: BLE001 - experiment failures cannot break live traffic
                observation.warnings.append(
                    f"Experiment evaluator failed: {_redact(str(exc))}"
                )
        _extend_unique(observation.warnings, warnings)
        return self._update_observation(
            observation,
            expected=expected or {"running"},
            status="completed",
        )

    def _fail_shadow_candidate(
        self,
        observation: ExperimentObservation,
        settings: ExperimentSettings,
        primary: CrupierResult,
        exc: Exception,
        warnings: list[str],
    ) -> ExperimentObservation:
        observation.status = "failed"
        observation.primary = _result_metrics(primary, store_output=settings.store_outputs)
        observation.candidate = {"error": True}
        observation.error = _redact(str(exc))
        observation.checks["failure_role"] = "candidate"
        _extend_unique(observation.warnings, warnings)
        return self._update_observation(
            observation,
            expected={"running", "scheduled"},
            status="failed",
        )

    def _fail_canary_candidate(
        self,
        observation: ExperimentObservation,
        settings: ExperimentSettings,
        primary: CrupierResult,
        exc: Exception,
        warnings: list[str],
    ) -> ExperimentObservation:
        observation.status = "failed"
        observation.primary = _result_metrics(primary, store_output=settings.store_outputs)
        observation.candidate = {"error": True}
        observation.error = _redact(str(exc))
        observation.checks["failure_role"] = "candidate"
        _extend_unique(observation.warnings, warnings)
        return self._update_observation(
            observation,
            expected={"running"},
            status="failed",
        )

    def _fail_observation(
        self,
        observation: ExperimentObservation,
        exc: Exception,
        *,
        role: str = "primary",
        secondary_error: Exception | None = None,
    ) -> ExperimentObservation:
        observation.status = "failed"
        observation.error = _redact(str(exc))
        observation.checks["failure_role"] = role
        if role in {"primary", "both"}:
            observation.primary = {"error": True}
        if role in {"candidate", "both"}:
            observation.candidate = {"error": True}
        if secondary_error is not None:
            observation.warnings.append(
                "Candidate failed before baseline fallback also failed: "
                f"{_redact(str(secondary_error))}"
            )
        return self._update_observation(
            observation,
            expected={"running", "scheduled"},
            status="failed",
        )

    def _update_observation(
        self,
        observation: ExperimentObservation,
        *,
        expected: set[str],
        status: str,
    ) -> ExperimentObservation:
        try:
            self.store.transition(
                kind="experiment_observation",
                record_id=observation.observation_id,
                expected_statuses=expected,
                status=status,
                payload={
                    "experiment": observation.experiment,
                    "observation": observation.to_dict(),
                },
                expires_at=None,
                event=status,
            )
        except Exception as exc:  # noqa: BLE001 - experiment telemetry is never on the live path
            observation.warnings.append(
                "Experiment state update failed after execution: "
                f"{_redact(str(exc))}"
            )
        return observation

    def _serve_baseline_without_state(
        self,
        settings: ExperimentSettings,
        request_kwargs: dict[str, Any],
        exc: Exception,
    ) -> CrupierResult:
        result = self.client.deal(**request_kwargs)
        result.experiment = self._transient_observation(settings, exc)
        result.warnings.append(result.experiment.warnings[-1])
        return result

    def _transient_observation(
        self,
        settings: ExperimentSettings,
        exc: Exception,
    ) -> ExperimentObservation:
        return ExperimentObservation(
            observation_id=f"obs_unpersisted_{uuid4().hex[:12]}",
            experiment=settings.name,
            traffic=settings.traffic,
            cohort="control",
            sampled=False,
            status="control_plane_unavailable",
            warnings=[
                (
                    "Experiment control plane was unavailable; baseline served and "
                    f"candidate execution was skipped: {_redact(str(exc))}"
                )
            ],
        )

    def _settings(self, name: str) -> ExperimentSettings:
        settings = self.client.config.experiments.get(name)
        if settings is None:
            raise CrupierError(f"Experiment {name!r} is not configured.")
        return settings

    def _runtime_record(self, experiment: str):
        self._settings(experiment)
        try:
            return self.store.get("experiment", experiment)
        except CrupierError:
            try:
                return self.store.create(
                    kind="experiment",
                    record_id=experiment,
                    status="active",
                    payload={"experiment": experiment},
                    event="activated",
                )
            except CrupierError:
                return self.store.get("experiment", experiment)

    def _runtime_status(self, experiment: str) -> str:
        return self._runtime_record(experiment).status

    def _maybe_auto_promote(self, settings: ExperimentSettings) -> str | None:
        if settings.promotion.action != "auto":
            return None
        try:
            report = self.report(settings.name)
            if report.promotion.eligible and report.status != "promoted":
                self.promote(settings.name, actor="automatic-policy")
        except Exception as exc:  # noqa: BLE001 - promotion must not break live traffic
            return "Automatic promotion failed closed: " + _redact(str(exc))
        return None


def _sticky_value(settings: ExperimentSettings, request_kwargs: dict[str, Any]) -> str:
    metadata = dict(request_kwargs.get("metadata") or {})
    key = settings.sticky_by
    value = metadata.get(key)
    if value is None:
        for fallback in ("session_id", "tenant_id", "request_id", "user_id_hash"):
            if metadata.get(fallback) is not None:
                value = metadata[fallback]
                break
    if value is None:
        value = {
            "task": request_kwargs.get("task"),
            "mode": request_kwargs.get("mode"),
            "input": request_kwargs.get("input"),
        }
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _sample(experiment: str, sticky_value: str, sample_rate: float) -> bool:
    if sample_rate <= 0:
        return False
    if sample_rate >= 1:
        return True
    digest = hashlib.sha256(f"{experiment}:{sticky_value}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return bucket < sample_rate


def _result_metrics(result: CrupierResult, *, store_output: bool) -> dict[str, Any]:
    output = result.output_text or ""
    metrics: dict[str, Any] = {
        "strategy": result.route.strategy if result.route else None,
        "models": result.route.models if result.route else [],
        "estimated_cost_usd": result.cost.estimated_usd,
        "actual_cost_usd": result.cost.actual_usd,
        "latency_ms": result.latency_ms,
        "output_chars": len(output),
        "output_hash": _hash(output),
        "quality_score": 1.0 if output.strip() else 0.0,
        "error": False,
        "dry_run": bool(result.provider_metadata.get("dry_run", False)),
    }
    if metrics["dry_run"]:
        metrics["quality_score"] = None
    if store_output:
        metrics["output"] = output
        try:
            encoded = json.dumps(
                result.output_json,
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError):
            metrics["output_json"] = None
            metrics["output_json_serialization_error"] = True
        else:
            metrics["output_json"] = json.loads(encoded)
    return metrics


def _single_checks(result: CrupierResult) -> dict[str, Any]:
    return {
        "route_valid": result.route is not None,
        "output_nonempty": bool(result.output_text.strip()),
    }


def _pair_checks(primary: CrupierResult, candidate: CrupierResult) -> dict[str, Any]:
    return {
        "primary_route_valid": primary.route is not None,
        "candidate_route_valid": candidate.route is not None,
        "primary_output_nonempty": bool(primary.output_text.strip()),
        "candidate_output_nonempty": bool(candidate.output_text.strip()),
    }


def _metric_diffs(primary: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "estimated_cost_usd": _difference(
            candidate.get("estimated_cost_usd"),
            primary.get("estimated_cost_usd"),
        ),
        "actual_cost_usd": _difference(
            candidate.get("actual_cost_usd"),
            primary.get("actual_cost_usd"),
        ),
        "latency_ms": _difference(
            candidate.get("latency_ms"),
            primary.get("latency_ms"),
        ),
        "output_chars": _difference(
            candidate.get("output_chars"),
            primary.get("output_chars"),
        ),
        "same_output_hash": candidate.get("output_hash") == primary.get("output_hash"),
    }


def _difference(candidate: Any, primary: Any) -> float | None:
    if not isinstance(candidate, int | float) or not isinstance(primary, int | float):
        return None
    return float(candidate) - float(primary)


def _aggregate_metrics(observations: list[ExperimentObservation]) -> dict[str, Any]:
    primary = [item.primary for item in observations if item.primary]
    candidate = [item.candidate for item in observations if item.candidate]
    candidate_errors = sum(bool(item.get("error")) for item in candidate)
    primary_errors = sum(1 for item in observations if bool(item.primary.get("error")))
    return {
        "primary": _cohort_metrics(primary, primary_errors),
        "candidate": _cohort_metrics(candidate, candidate_errors),
        "paired_count": sum(bool(item.primary and item.candidate) for item in observations),
    }


def _cohort_metrics(items: list[dict[str, Any]], errors: int) -> dict[str, Any]:
    costs: list[float] = []
    for item in items:
        actual = item.get("actual_cost_usd")
        estimated = item.get("estimated_cost_usd")
        value = actual if isinstance(actual, int | float) else estimated
        if isinstance(value, int | float):
            costs.append(float(value))
    latencies = [
        float(item["latency_ms"])
        for item in items
        if isinstance(item.get("latency_ms"), int | float)
    ]
    quality = [
        float(item["quality_score"])
        for item in items
        if isinstance(item.get("quality_score"), int | float)
    ]
    count = len(items)
    return {
        "count": count,
        "errors": errors,
        "error_rate": errors / count if count else None,
        "avg_cost_usd": mean(costs) if costs else None,
        "avg_latency_ms": mean(latencies) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "avg_quality": mean(quality) if quality else None,
    }


def _promotion_report(
    settings: ExperimentSettings,
    observations: list[ExperimentObservation],
    metrics: dict[str, Any],
) -> PromotionReport:
    promotion = settings.promotion
    candidate = metrics["candidate"]
    primary = metrics["primary"]
    sample_count = int(candidate.get("count") or 0)
    reasons: list[str] = []
    gates: dict[str, bool | None] = {}
    gates["min_samples"] = sample_count >= promotion.min_samples
    if not gates["min_samples"]:
        reasons.append(f"candidate samples {sample_count} < {promotion.min_samples}")

    candidate_error = candidate.get("error_rate")
    primary_error = primary.get("error_rate")
    gates["max_error_rate"] = (
        isinstance(candidate_error, int | float)
        and _wilson_upper(
            int(candidate.get("errors") or 0),
            sample_count,
            promotion.confidence,
        )
        <= promotion.max_error_rate
    )
    if not gates["max_error_rate"]:
        reasons.append("candidate error-rate confidence bound exceeds policy")
    gates["max_error_rate_delta"] = (
        isinstance(candidate_error, int | float)
        and isinstance(primary_error, int | float)
        and float(candidate_error) - float(primary_error)
        <= promotion.max_error_rate_delta
    )
    if not gates["max_error_rate_delta"]:
        reasons.append("candidate error-rate delta exceeds policy")

    quality_values = [
        float(item.checks[promotion.quality_check])
        for item in observations
        if isinstance(item.checks.get(promotion.quality_check), int | float)
    ]
    quality_delta = mean(quality_values) if quality_values else None
    metrics["quality_evidence"] = {
        "check": promotion.quality_check,
        "count": len(quality_values),
        "average_delta": quality_delta,
    }
    if promotion.require_quality_evaluator:
        gates["quality"] = (
            quality_delta is not None
            and quality_delta >= promotion.min_quality_delta
        )
    else:
        candidate_quality = candidate.get("avg_quality")
        primary_quality = primary.get("avg_quality")
        gates["quality"] = (
            isinstance(candidate_quality, int | float)
            and isinstance(primary_quality, int | float)
            and float(candidate_quality) - float(primary_quality)
            >= promotion.min_quality_delta
        )
    if not gates["quality"]:
        reasons.append("quality evidence is missing or below policy")

    gates["cost"] = _ratio_gate(
        candidate.get("avg_cost_usd"),
        primary.get("avg_cost_usd"),
        promotion.max_cost_ratio,
    )
    if gates["cost"] is False:
        reasons.append("candidate cost ratio exceeds policy")
    gates["latency"] = _ratio_gate(
        candidate.get("p95_latency_ms"),
        primary.get("p95_latency_ms"),
        promotion.max_p95_latency_ratio,
    )
    if gates["latency"] is False:
        reasons.append("candidate p95 latency ratio exceeds policy")
    if any(item.candidate.get("dry_run") for item in observations if item.candidate):
        gates["live_execution_evidence"] = False
        reasons.append("plan_only observations cannot authorize promotion")
    else:
        gates["live_execution_evidence"] = True
    eligible = all(value is not False and value is not None for value in gates.values())
    return PromotionReport(
        experiment=settings.name,
        eligible=eligible,
        sample_count=sample_count,
        gates=gates,
        metrics=metrics,
        reasons=reasons,
    )


def _ratio_gate(candidate: Any, primary: Any, maximum: float | None) -> bool | None:
    if maximum is None:
        return True
    if not isinstance(candidate, int | float) or not isinstance(primary, int | float):
        return None
    if float(primary) == 0:
        return float(candidate) == 0
    return float(candidate) / float(primary) <= maximum


def _wilson_upper(errors: int, total: int, confidence: float) -> float:
    if total <= 0:
        return 1.0
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    p = errors / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (centre + margin) / denominator


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _observation(data: dict[str, Any]) -> ExperimentObservation:
    return ExperimentObservation(
        observation_id=str(data["observation_id"]),
        experiment=str(data["experiment"]),
        traffic=str(data["traffic"]),
        cohort=str(data["cohort"]),
        sampled=bool(data["sampled"]),
        status=str(data["status"]),
        primary=dict(data.get("primary", {})),
        candidate=dict(data.get("candidate", {})),
        diffs=dict(data.get("diffs", {})),
        checks=dict(data.get("checks", {})),
        warnings=list(data.get("warnings", [])),
        error=data.get("error"),
    )


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _extend_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _validated_checks(checks: Any) -> dict[str, Any]:
    if not isinstance(checks, dict):
        raise CrupierError("Experiment evaluator must return a dictionary.")
    try:
        encoded = json.dumps(checks, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CrupierError(
            "Experiment checks must contain finite JSON-compatible values."
        ) from exc
    decoded = json.loads(encoded)
    return dict(decoded)


_SECRET_REPLACERS = (
    (re.compile(("s" + "k-") + r"[A-Za-z0-9_\-]{10,}"), "[redacted]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{12,}", re.IGNORECASE), r"\1[redacted]"),
    (re.compile(r"([A-Z][A-Z0-9_]*_API_KEY=)[^\s]+"), r"\1[redacted]"),
)


def _redact(message: str) -> str:
    redacted = message
    for pattern, replacement in _SECRET_REPLACERS:
        redacted = pattern.sub(replacement, redacted)
    return redacted[:4_000]
