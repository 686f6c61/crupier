"""Multi-turn sessions with compatible route stickiness and bounded state."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any
from uuid import uuid4

from .errors import CrupierBudgetExceededError, CrupierError, CrupierExecutionLimitError
from .models import CrupierResult, RoutePlan
from .multimodal import normalize_files, plan_file_representations
from .state import SQLiteStateStore, StateRecord
from .tools import normalize_tools

STICKINESS_MODES = {"none", "compatible"}


@dataclass(slots=True)
class RouteTransition:
    turn: int
    reason: str
    reused: bool
    strategy: str
    models: list[str]
    trace_id: str | None = None
    estimated_cost_usd: float | None = None
    latency_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SessionInfo:
    session_id: str
    status: str
    mode: str
    stickiness: str
    turns: int
    cumulative_cost_usd: float
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CrupierSession:
    def __init__(
        self,
        client: Any,
        *,
        session_id: str,
        mode: str,
        stickiness: str = "compatible",
        persist: bool = False,
        constraints: dict[str, Any] | None = None,
        max_turns: int = 100,
        max_history_chars: int = 200_000,
        max_session_cost_usd: float | None = None,
        compactor: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
        store: SQLiteStateStore | None = None,
        state_record: StateRecord | None = None,
    ):
        if stickiness not in STICKINESS_MODES:
            raise CrupierError(
                f"Session stickiness must be one of: {', '.join(sorted(STICKINESS_MODES))}."
            )
        if max_turns < 1:
            raise CrupierError("Session max_turns must be at least 1.")
        if max_history_chars < 1_000:
            raise CrupierError("Session max_history_chars must be at least 1000.")
        self.client = client
        self.session_id = session_id
        self.mode = mode
        self.stickiness = stickiness
        self.persist = persist
        self.base_constraints = dict(constraints or {})
        self.max_turns = int(max_turns)
        self.max_history_chars = int(max_history_chars)
        self.max_session_cost_usd = (
            float(max_session_cost_usd) if max_session_cost_usd is not None else None
        )
        self.compactor = compactor
        self._store = store
        self._lock = RLock()
        self._version = state_record.version if state_record else 0
        self.status = state_record.status if state_record else "active"
        payload = state_record.payload if state_record else {}
        self.messages: list[dict[str, Any]] = list(payload.get("messages", []))
        self.route_history: list[RouteTransition] = [
            RouteTransition(**item) for item in payload.get("route_history", [])
        ]
        previous_plan = payload.get("previous_plan")
        self._previous_plan = (
            RoutePlan.from_dict(previous_plan) if isinstance(previous_plan, dict) else None
        )
        self._previous_signature: dict[str, Any] | None = payload.get("previous_signature")
        self._tool_ledger: list[dict[str, Any]] = list(payload.get("tool_ledger", []))
        self.turns = int(payload.get("turns", 0))
        self.cumulative_cost_usd = float(payload.get("cumulative_cost_usd", 0.0))

    def deal(
        self,
        task: str,
        input: Any = None,
        *,
        mode: str | None = None,
        strategy: str | None = None,
        constraints: dict[str, Any] | None = None,
        files: list[Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
        response_schema: Any = None,
        metadata: dict[str, Any] | None = None,
        trace: bool | str = False,
        dry_run: bool | None = None,
        approval_token: str | None = None,
        experiment: str | None = None,
    ) -> CrupierResult:
        if approval_token:
            return self.execute_approved(
                approval_token,
                tools=tools,
                trace=trace,
            )
        with self._lock:
            self._require_active()
            if self.turns >= self.max_turns:
                raise CrupierExecutionLimitError(
                    f"Session {self.session_id} exhausted max_turns={self.max_turns}."
                )
            turn_mode = mode or self.mode
            turn_constraints = {**self.base_constraints, **dict(constraints or {})}
            all_messages = [*self.messages, *list(messages or [])]
            signature = self._signature(
                mode=turn_mode,
                strategy=strategy,
                constraints=turn_constraints,
                files=files,
                tools=tools,
                response_schema=response_schema,
            )
            self._apply_remaining_budget(turn_constraints)
            reason = self._replan_reason(signature)
            turn_metadata = dict(metadata or {})
            turn_metadata.update(
                {
                    "session_id": self.session_id,
                    "session_turn": self.turns + 1,
                    "_crupier_previous_tool_executions": list(self._tool_ledger),
                }
            )
            if reason is None and self._previous_plan is not None:
                turn_metadata["_crupier_sticky_plan"] = self._previous_plan
                reason = "compatible_route_retained"

            result = self.client.deal(
                task=task,
                input=input,
                mode=turn_mode,
                strategy=strategy,
                constraints=turn_constraints,
                files=files,
                messages=all_messages,
                tools=tools,
                response_schema=response_schema,
                metadata=turn_metadata,
                trace="debug",
                dry_run=dry_run,
                approval_token=approval_token,
                experiment=experiment,
            )
            self._record_turn(
                task=task,
                input=input,
                result=result,
                signature=signature,
                reason=reason or "route_replanned",
                dry_run=bool(result.provider_metadata.get("dry_run", False)),
            )
            if not trace:
                result.trace = None
            elif trace == "summary" and result.trace is not None:
                # The trace object remains typed; callers choose summary serialization via to_dict().
                pass
            return result

    def execute_approved(
        self,
        approval_token: str,
        *,
        tools: list[Any] | None = None,
        trace: bool | str = False,
    ) -> CrupierResult:
        with self._lock:
            self._require_active()
            if self.turns >= self.max_turns:
                raise CrupierExecutionLimitError(
                    f"Session {self.session_id} exhausted max_turns={self.max_turns}."
                )
            prepared = self.client.approvals.consume(
                approval_token,
                tools=tools,
                expected_metadata={"session_id": self.session_id},
            )
            request = prepared.request
            signature = self._signature(
                mode=request.mode or self.mode,
                strategy=request.strategy,
                constraints=dict(request.constraints),
                files=(
                    list(request.file_plan.assets)
                    if request.file_plan is not None
                    else list(request.files)
                ),
                tools=list(request.tools),
                response_schema=request.response_schema,
            )
            reason = self._replan_reason(signature)
            prepared.dry_run = False
            result = self.client.execute(prepared, trace="debug")
            self._record_turn(
                task=request.task,
                input=request.input,
                result=result,
                signature=signature,
                reason=reason or "route_replanned",
                dry_run=False,
            )
            if not trace:
                result.trace = None
            return result

    def close(self) -> SessionInfo:
        with self._lock:
            if self.status == "closed":
                return self.info()
            self.status = "closed"
            self._persist(event="closed", status="closed")
            return self.info()

    def info(self) -> SessionInfo:
        return SessionInfo(
            session_id=self.session_id,
            status=self.status,
            mode=self.mode,
            stickiness=self.stickiness,
            turns=self.turns,
            cumulative_cost_usd=round(self.cumulative_cost_usd, 8),
        )

    def _record_turn(
        self,
        *,
        task: str,
        input: Any,
        result: CrupierResult,
        signature: dict[str, Any],
        reason: str,
        dry_run: bool,
    ) -> None:
        route = result.route
        if route is None:
            raise CrupierError("Session turn completed without a route.")
        trace_id = result.trace.trace_id if result.trace else None
        reused = bool(
            result.trace
            and result.trace.final_quality_signals.get("sticky_route_reused")
        )
        if reason == "compatible_route_retained" and not reused:
            reason = "route_invalidated_by_policy_or_budget"
        self.turns += 1
        self.messages.append(
            {
                "role": "user",
                "content": {"task": task, "input": input},
            }
        )
        self.messages.append({"role": "assistant", "content": result.output_text})
        self._compact_history()
        self._previous_plan = RoutePlan.from_dict(route.to_dict())
        self._previous_signature = signature
        self._tool_ledger = list(result.provider_metadata.get("tool_calls", self._tool_ledger))
        if not dry_run:
            actual = result.cost.actual_usd
            estimated = result.cost.estimated_usd
            self.cumulative_cost_usd += float(actual if actual is not None else estimated)
        self.route_history.append(
            RouteTransition(
                turn=self.turns,
                reason=reason,
                reused=reused,
                strategy=route.strategy,
                models=route.models,
                trace_id=trace_id,
                estimated_cost_usd=route.estimated_cost.estimated_usd,
                latency_ms=result.latency_ms,
            )
        )
        self._persist(event="turn_completed", status="active")

    def _signature(
        self,
        *,
        mode: str,
        strategy: str | None,
        constraints: dict[str, Any],
        files: list[Any] | None,
        tools: list[Any] | None,
        response_schema: Any,
    ) -> dict[str, Any]:
        assets = normalize_files(files)
        file_plan = plan_file_representations(assets, constraints=constraints)
        tool_catalog = [item.public_dict() for item in normalize_tools(list(tools or []))]
        return {
            "mode": mode,
            "strategy": strategy,
            "risk_level": constraints.get("risk_level"),
            "file_modalities": file_plan.required_model_modalities if file_plan else [],
            "file_capabilities": file_plan.required_model_capabilities if file_plan else [],
            "file_representations": (
                [item.representation for item in file_plan.representations] if file_plan else []
            ),
            "tools_hash": _digest(tool_catalog),
            "response_schema_hash": _digest(response_schema),
            "budget": {
                key: constraints.get(key)
                for key in ("max_cost_usd", "max_latency_ms", "max_calls", "max_output_tokens")
            },
            "requires_tools": bool(constraints.get("requires_tools") or tools),
            "streaming": bool(
                constraints.get("stream") or constraints.get("require_streaming")
            ),
        }

    def _replan_reason(self, signature: dict[str, Any]) -> str | None:
        if self._previous_plan is None or self._previous_signature is None:
            return "initial_route"
        if self.stickiness == "none":
            return "stickiness_disabled"
        previous = self._previous_signature
        if signature["mode"] != previous.get("mode"):
            return "mode_changed"
        if signature["strategy"] != previous.get("strategy"):
            return "strategy_changed"
        if signature["risk_level"] != previous.get("risk_level"):
            return "risk_changed"
        for key in (
            "file_modalities",
            "file_capabilities",
            "file_representations",
            "tools_hash",
            "response_schema_hash",
            "requires_tools",
            "streaming",
        ):
            if signature[key] != previous.get(key):
                return "capability_changed"
        if signature["budget"] != previous.get("budget"):
            return "budget_changed"
        if self._context_pressure():
            return "context_pressure"
        return None

    def _context_pressure(self) -> bool:
        if self._previous_plan is None:
            return False
        windows = []
        for model in self._previous_plan.models:
            try:
                window = self.client.registry.get(model).context_window
            except Exception:  # noqa: BLE001,S112 - missing refreshed card forces normal policy revalidation
                continue
            if isinstance(window, int) and window > 0:
                windows.append(window)
        if not windows:
            return False
        estimated_tokens = len(json.dumps(self.messages, ensure_ascii=False, default=str)) // 4
        return estimated_tokens >= int(min(windows) * 0.75)

    def _apply_remaining_budget(self, constraints: dict[str, Any]) -> None:
        if self.max_session_cost_usd is None:
            return
        remaining = self.max_session_cost_usd - self.cumulative_cost_usd
        if remaining <= 0:
            raise CrupierBudgetExceededError(
                f"Session {self.session_id} exhausted max_session_cost_usd="
                f"{self.max_session_cost_usd:.4f}."
            )
        turn_limit = constraints.get("max_cost_usd")
        constraints["max_cost_usd"] = (
            remaining if turn_limit is None else min(float(turn_limit), remaining)
        )

    def _compact_history(self) -> None:
        encoded = json.dumps(self.messages, ensure_ascii=False, default=str)
        if len(encoded) <= self.max_history_chars:
            return
        if self.compactor is not None:
            compacted = self.compactor(list(self.messages))
            if not isinstance(compacted, list):
                raise CrupierError("Session compactor must return a list of messages.")
            if len(json.dumps(compacted, ensure_ascii=False, default=str)) > self.max_history_chars:
                raise CrupierError(
                    "Session compactor output exceeds max_history_chars."
                )
            self.messages = compacted
            return
        omitted = 0
        while len(self.messages) > 2:
            self.messages = self.messages[2:]
            omitted += 2
            if len(json.dumps(self.messages, ensure_ascii=False, default=str)) <= self.max_history_chars:
                break
        self.messages.insert(
            0,
            {
                "role": "system",
                "content": f"{omitted} older session messages were compacted by configured limits.",
            },
        )

    def _require_active(self) -> None:
        if self.status != "active":
            raise CrupierError(f"Session {self.session_id} is {self.status!r}.")

    def _persist(self, *, event: str, status: str) -> None:
        if not self.persist or self._store is None:
            self.status = status
            return
        record = self._store.transition(
            kind="session",
            record_id=self.session_id,
            expected_statuses={"active"},
            status=status,
            payload=self._payload(),
            expires_at=None,
            event=event,
            expected_version=self._version,
        )
        self._version = record.version
        self.status = record.status

    def _payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "stickiness": self.stickiness,
            "base_constraints": self.base_constraints,
            "max_turns": self.max_turns,
            "max_history_chars": self.max_history_chars,
            "max_session_cost_usd": self.max_session_cost_usd,
            "turns": self.turns,
            "cumulative_cost_usd": self.cumulative_cost_usd,
            "messages": self.messages,
            "route_history": [item.to_dict() for item in self.route_history],
            "previous_plan": self._previous_plan.to_dict() if self._previous_plan else None,
            "previous_signature": self._previous_signature,
            "tool_ledger": self._tool_ledger,
        }


class SessionManager:
    def __init__(self, client: Any, path: str):
        self.client = client
        self.store = SQLiteStateStore(path)

    def create(
        self,
        *,
        mode: str | None = None,
        sticky: bool | None = None,
        stickiness: str | None = None,
        persist: bool = False,
        constraints: dict[str, Any] | None = None,
        max_turns: int = 100,
        max_history_chars: int = 200_000,
        max_session_cost_usd: float | None = None,
        compactor: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    ) -> CrupierSession:
        resolved_stickiness = stickiness or ("compatible" if sticky is not False else "none")
        session_id = f"ses_{uuid4().hex[:16]}"
        session = CrupierSession(
            self.client,
            session_id=session_id,
            mode=mode or self.client.config.project.default_profile,
            stickiness=resolved_stickiness,
            persist=persist,
            constraints=constraints,
            max_turns=max_turns,
            max_history_chars=max_history_chars,
            max_session_cost_usd=max_session_cost_usd,
            compactor=compactor,
            store=self.store,
        )
        if persist:
            record = self.store.create(
                kind="session",
                record_id=session_id,
                status="active",
                payload=session._payload(),
                event="created",
            )
            session._version = record.version
        return session

    def resume(self, session_id: str) -> CrupierSession:
        record = self.store.get("session", session_id)
        payload = record.payload
        return CrupierSession(
            self.client,
            session_id=session_id,
            mode=str(payload.get("mode") or self.client.config.project.default_profile),
            stickiness=str(payload.get("stickiness", "compatible")),
            persist=True,
            constraints=dict(payload.get("base_constraints", {})),
            max_turns=int(payload.get("max_turns", 100)),
            max_history_chars=int(payload.get("max_history_chars", 200_000)),
            max_session_cost_usd=payload.get("max_session_cost_usd"),
            store=self.store,
            state_record=record,
        )

    def list(self, *, status: str | None = None) -> list[SessionInfo]:
        return [
            SessionInfo(
                session_id=record.id,
                status=record.status,
                mode=str(record.payload.get("mode", "")),
                stickiness=str(record.payload.get("stickiness", "compatible")),
                turns=int(record.payload.get("turns", 0)),
                cumulative_cost_usd=float(record.payload.get("cumulative_cost_usd", 0.0)),
                created_at=record.created_at,
                updated_at=record.updated_at,
            )
            for record in self.store.list("session", status=status)
        ]


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
