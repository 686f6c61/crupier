"""Durable approvals bound to frozen route and request fingerprints."""

from __future__ import annotations

import builtins
import hashlib
import hmac
import json
import secrets
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from .errors import CrupierError
from .models import (
    DecisionTrace,
    FileAsset,
    FileRoutingPlan,
    PreparedDeal,
    RequestEnvelope,
    RoutePlan,
)
from .state import SQLiteStateStore, StateRecord
from .tools import normalize_tools

APPROVAL_STATUSES = {"pending", "granted", "rejected", "expired", "revoked", "consumed"}
ApprovalReviewerVerifier = Callable[[str, dict[str, Any]], dict[str, Any]]


@dataclass(slots=True)
class ApprovalRecord:
    approval_id: str
    trace_id: str
    status: str
    plan_hash: str
    request_fingerprint: str
    created_at: str
    updated_at: str
    expires_at: str | None = None
    reviewer: str | None = None
    reviewer_verified: bool = False
    reviewer_attestation: dict[str, Any] | None = None
    reason: str = ""
    scope: str = "route"
    token: str | None = None

    def to_dict(self, *, include_token: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_token:
            data.pop("token", None)
        return {key: value for key, value in data.items() if value not in (None, "")}


class ApprovalManager:
    def __init__(
        self,
        path: str | Path,
        *,
        reviewer_verifier: ApprovalReviewerVerifier | None = None,
    ):
        self.store = SQLiteStateStore(path)
        self.reviewer_verifier = reviewer_verifier

    def create(
        self,
        prepared: PreparedDeal,
        *,
        ttl_s: int = 86_400,
        scope: str = "route",
    ) -> ApprovalRecord:
        if ttl_s <= 0:
            raise CrupierError("Approval ttl_s must be greater than zero.")
        if scope != "route":
            raise CrupierError("Approval scope must currently be 'route'.")
        approval_id = f"apr_{uuid4().hex[:16]}"
        expires_at = _after(ttl_s)
        payload = {
            "trace_id": prepared.trace.trace_id,
            "scope": scope,
            "plan_hash": plan_hash(prepared.plan),
            "request_fingerprint": request_fingerprint(prepared.request),
            "prepared": serialize_prepared(prepared),
            "reviewer": None,
            "reviewer_verified": False,
            "reviewer_attestation": None,
            "reason": "",
            "token_hash": None,
        }
        record = self.store.create(
            kind="approval",
            record_id=approval_id,
            status="pending",
            payload=payload,
            expires_at=expires_at,
            event="requested",
        )
        return _approval(record)

    def list(self, *, status: str | None = None, limit: int = 200) -> list[ApprovalRecord]:
        if status is not None and status not in APPROVAL_STATUSES:
            raise CrupierError(f"Unknown approval status {status!r}.")
        records = self.store.list("approval", status=status, limit=limit)
        return [self._expire_if_needed(record) for record in records]

    def get(self, approval_id: str) -> ApprovalRecord:
        return self._expire_if_needed(self.store.get("approval", approval_id))

    def resolve(self, approval_or_trace_id: str) -> ApprovalRecord:
        if approval_or_trace_id.startswith("apr_"):
            return self.get(approval_or_trace_id)
        matches = [
            item
            for item in self.list(limit=10_000)
            if item.trace_id == approval_or_trace_id
        ]
        if not matches:
            raise CrupierError(
                f"No approval was found for {approval_or_trace_id!r}."
            )
        pending = [item for item in matches if item.status == "pending"]
        if len(pending) == 1:
            return pending[0]
        if len(matches) == 1:
            return matches[0]
        raise CrupierError(
            f"Multiple approvals match trace {approval_or_trace_id!r}; use an approval_id."
        )

    def events(self, approval_id: str) -> builtins.list[dict[str, Any]]:
        return self.store.events("approval", approval_id)

    def grant(
        self,
        approval_id: str,
        *,
        reviewer: str,
        ttl_s: int = 3_600,
        reason: str = "",
        reviewer_context: dict[str, Any] | None = None,
    ) -> ApprovalRecord:
        if not reviewer.strip():
            raise CrupierError("Approval reviewer cannot be empty.")
        if ttl_s <= 0:
            raise CrupierError("Approval ttl_s must be greater than zero.")
        current = self.store.get("approval", approval_id)
        current_approval = self._expire_if_needed(current)
        if current_approval.status != "pending":
            raise CrupierError(
                f"Approval {approval_id} is {current_approval.status!r}; only pending approvals can be granted."
            )
        secret = secrets.token_urlsafe(32)
        token = f"{approval_id}.{secret}"
        attestation = self._verify_reviewer(
            reviewer.strip(),
            dict(reviewer_context or {}),
        )
        payload = dict(current.payload)
        payload.update(
            {
                "reviewer": reviewer.strip(),
                "reviewer_verified": bool(attestation.get("verified", False)),
                "reviewer_attestation": attestation or None,
                "reason": reason.strip()[:2_000],
                "token_hash": _token_hash(token),
            }
        )
        expires_at = min(_parse(current.expires_at), _parse(_after(ttl_s))).isoformat()
        record = self.store.transition(
            kind="approval",
            record_id=approval_id,
            expected_statuses={"pending"},
            status="granted",
            payload=payload,
            expires_at=expires_at,
            event="granted",
            actor=reviewer.strip(),
        )
        approval = _approval(record)
        approval.token = token
        return approval

    def reject(self, approval_id: str, *, reviewer: str, reason: str) -> ApprovalRecord:
        return self._decision(
            approval_id,
            reviewer=reviewer,
            reason=reason,
            status="rejected",
            event="rejected",
        )

    def revoke(self, approval_id: str, *, reviewer: str, reason: str) -> ApprovalRecord:
        return self._decision(
            approval_id,
            reviewer=reviewer,
            reason=reason,
            status="revoked",
            event="revoked",
            expected={"pending", "granted"},
        )

    def consume(
        self,
        token: str,
        *,
        tools: builtins.list[Any] | None = None,
        expected_metadata: dict[str, Any] | None = None,
    ) -> PreparedDeal:
        approval_id = _approval_id_from_token(token)
        current = self.store.get("approval", approval_id)
        approval = self._expire_if_needed(current)
        if approval.status != "granted":
            raise CrupierError(
                f"Approval {approval_id} is {approval.status!r}; a live granted approval is required."
            )
        expected_hash = str(current.payload.get("token_hash") or "")
        if not expected_hash or not hmac.compare_digest(expected_hash, _token_hash(token)):
            raise CrupierError("Approval token is invalid.")
        prepared = deserialize_prepared(dict(current.payload["prepared"]))
        if plan_hash(prepared.plan) != current.payload.get("plan_hash"):
            raise CrupierError("Frozen approval plan hash no longer matches its stored payload.")
        if request_fingerprint(prepared.request) != current.payload.get("request_fingerprint"):
            raise CrupierError("Frozen approval request fingerprint no longer matches its stored payload.")
        for key, expected in dict(expected_metadata or {}).items():
            if prepared.request.metadata.get(key) != expected:
                raise CrupierError(
                    f"Approval {approval_id} is not bound to the expected {key!r} context."
                )
        stored_tools = [item.public_dict() for item in normalize_tools(prepared.request.tools)]
        supplied_tools = [item.public_dict() for item in normalize_tools(list(tools or []))]
        if stored_tools and not tools:
            raise CrupierError(
                "Approved route requires the original runtime tool handlers when it is executed."
            )
        if supplied_tools != stored_tools:
            raise CrupierError("Runtime tools do not match the tool catalog bound to this approval.")
        if tools:
            prepared.request.tools = list(tools)
        payload = dict(current.payload)
        payload["token_hash"] = None
        self.store.transition(
            kind="approval",
            record_id=approval_id,
            expected_statuses={"granted"},
            status="consumed",
            payload=payload,
            expires_at=current.expires_at,
            event="consumed",
            actor=str(current.payload.get("reviewer") or ""),
        )
        prepared.request.constraints["human_approval_granted"] = True
        approval_bound_tools = {
            tool.name
            for tool in normalize_tools(prepared.request.tools)
            if tool.requires_approval or tool.side_effects
        }
        approval_bound_tools.update(
            str(name)
            for name in prepared.request.constraints.get("require_approval_for", [])
        )
        if approval_bound_tools:
            prepared.request.constraints["approved_tools"] = sorted(approval_bound_tools)
        prepared.request.metadata["_crupier_approval"] = {
            "approval_id": approval_id,
            "reviewer": current.payload.get("reviewer"),
            "reviewer_verified": bool(current.payload.get("reviewer_verified")),
            "reviewer_attestation": current.payload.get("reviewer_attestation"),
            "plan_hash": current.payload.get("plan_hash"),
        }
        return prepared

    def _verify_reviewer(
        self,
        reviewer: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if self.reviewer_verifier is None:
            return {}
        attestation = self.reviewer_verifier(reviewer, context)
        if not isinstance(attestation, dict):
            raise CrupierError("Approval reviewer verifier must return a dictionary.")
        if not attestation.get("verified"):
            raise CrupierError("Approval reviewer identity could not be verified.")
        return _jsonable(attestation)

    def _decision(
        self,
        approval_id: str,
        *,
        reviewer: str,
        reason: str,
        status: str,
        event: str,
        expected: set[str] | None = None,
    ) -> ApprovalRecord:
        if not reviewer.strip():
            raise CrupierError("Approval reviewer cannot be empty.")
        current = self.store.get("approval", approval_id)
        payload = dict(current.payload)
        payload.update(
            {
                "reviewer": reviewer.strip(),
                "reason": reason.strip()[:2_000],
                "token_hash": None,
            }
        )
        record = self.store.transition(
            kind="approval",
            record_id=approval_id,
            expected_statuses=expected or {"pending"},
            status=status,
            payload=payload,
            expires_at=current.expires_at,
            event=event,
            actor=reviewer.strip(),
        )
        return _approval(record)

    def _expire_if_needed(self, record: StateRecord) -> ApprovalRecord:
        if record.status in {"pending", "granted"} and _parse(record.expires_at) <= datetime.now(UTC):
            payload = dict(record.payload)
            payload["token_hash"] = None
            record = self.store.transition(
                kind="approval",
                record_id=record.id,
                expected_statuses={record.status},
                status="expired",
                payload=payload,
                expires_at=record.expires_at,
                event="expired",
            )
        return _approval(record)


def plan_hash(plan: RoutePlan) -> str:
    return _digest(plan.to_dict())


def request_fingerprint(request: RequestEnvelope) -> str:
    return _digest(_request_payload(request, include_runtime=False))


def serialize_prepared(prepared: PreparedDeal) -> dict[str, Any]:
    return {
        "request": _request_payload(prepared.request, include_runtime=True),
        "plan": prepared.plan.to_dict(),
        "trace": prepared.trace.to_dict(),
        "dry_run": prepared.dry_run,
        "warnings": list(prepared.warnings),
        "planning_calls": _jsonable(prepared.planning_calls),
        "execution_budget_snapshot": _jsonable(prepared.execution_budget_snapshot),
    }


def deserialize_prepared(data: dict[str, Any]) -> PreparedDeal:
    request_data = dict(data["request"])
    file_plan_data = request_data.get("file_plan")
    request = RequestEnvelope(
        task=str(request_data["task"]),
        input=request_data.get("input"),
        messages=list(request_data.get("messages", [])),
        files=[FileAsset.from_dict(item) for item in request_data.get("files", [])],
        file_plan=FileRoutingPlan.from_dict(file_plan_data) if isinstance(file_plan_data, dict) else None,
        tools=list(request_data.get("tools", [])),
        response_schema=request_data.get("response_schema"),
        mode=request_data.get("mode"),
        strategy=request_data.get("strategy"),
        constraints=dict(request_data.get("constraints", {})),
        metadata=dict(request_data.get("metadata", {})),
        tenant_id=request_data.get("tenant_id"),
        user_id_hash=request_data.get("user_id_hash"),
    )
    plan = RoutePlan.from_dict(dict(data["plan"]))
    trace_data = dict(data["trace"])
    trace = DecisionTrace(
        trace_id=str(trace_data["trace_id"]),
        request_summary=str(trace_data.get("request_summary", "")),
        candidate_models=list(trace_data.get("candidate_models", [])),
        excluded_models=list(trace_data.get("excluded_models", [])),
        policy_filters=list(trace_data.get("policy_filters", [])),
        orchestrator_model=trace_data.get("orchestrator_model"),
        route_plan=plan,
        provider_calls=list(trace_data.get("provider_calls", [])),
        fallbacks=list(trace_data.get("fallbacks", [])),
        errors=list(trace_data.get("errors", [])),
        storage_decision=dict(trace_data.get("storage_decision", {})),
        final_quality_signals=dict(trace_data.get("final_quality_signals", {})),
    )
    return PreparedDeal(
        request=request,
        plan=plan,
        trace=trace,
        dry_run=bool(data.get("dry_run", False)),
        warnings=list(data.get("warnings", [])),
        planning_calls=list(data.get("planning_calls", [])),
        execution_budget_snapshot=dict(data.get("execution_budget_snapshot", {})),
    )


def _request_payload(request: RequestEnvelope, *, include_runtime: bool) -> dict[str, Any]:
    tools = [tool.public_dict() for tool in normalize_tools(request.tools)]
    files = [asset.to_dict(include_uri=include_runtime) for asset in request.files]
    source_file_digests = []
    for asset in request.file_plan.assets if request.file_plan else request.files:
        item = asset.to_dict(include_uri=include_runtime)
        item["sha256"] = _asset_digest(asset)
        source_file_digests.append(item)
    metadata = {
        key: value
        for key, value in request.metadata.items()
        if not key.startswith("_crupier_")
    }
    return _jsonable(
        {
            "task": request.task,
            "input": request.input,
            "messages": request.messages,
            "files": files,
            "source_files": source_file_digests,
            "file_plan": request.file_plan.to_dict(include_uri=include_runtime)
            if request.file_plan
            else None,
            "tools": tools,
            "response_schema": request.response_schema,
            "mode": request.mode,
            "strategy": request.strategy,
            "constraints": {
                key: value
                for key, value in request.constraints.items()
                if key not in {"human_approval_granted"}
            },
            "metadata": metadata,
            "tenant_id": request.tenant_id,
            "user_id_hash": request.user_id_hash,
        }
    )


def _asset_digest(asset: FileAsset) -> str | None:
    if not asset.uri:
        raise CrupierError(
            f"Approval-bound file {asset.name or '<unnamed>'!r} needs a stable URI."
        )
    if asset.uri.startswith(("http://", "https://")):
        expected_digest = str(asset.metadata.get("sha256") or "").lower()
        if len(expected_digest) != 64 or any(
            character not in "0123456789abcdef" for character in expected_digest
        ):
            raise CrupierError(
                f"Remote approval-bound file {asset.name or asset.uri!r} requires metadata.sha256."
            )
        return expected_digest
    if asset.uri.startswith("data:"):
        return hashlib.sha256(asset.uri.encode("utf-8")).hexdigest()
    path = Path(asset.uri).expanduser()
    if not path.exists() or not path.is_file():
        raise CrupierError(
            f"Approval-bound file {asset.name or str(path)!r} must exist when the route is frozen."
        )
    content_digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            content_digest.update(chunk)
    return content_digest.hexdigest()


def _approval(record: StateRecord) -> ApprovalRecord:
    payload = record.payload
    return ApprovalRecord(
        approval_id=record.id,
        trace_id=str(payload.get("trace_id", "")),
        status=record.status,
        plan_hash=str(payload.get("plan_hash", "")),
        request_fingerprint=str(payload.get("request_fingerprint", "")),
        created_at=record.created_at,
        updated_at=record.updated_at,
        expires_at=record.expires_at,
        reviewer=payload.get("reviewer"),
        reviewer_verified=bool(payload.get("reviewer_verified", False)),
        reviewer_attestation=(
            dict(payload["reviewer_attestation"])
            if isinstance(payload.get("reviewer_attestation"), dict)
            else None
        ),
        reason=str(payload.get("reason", "")),
        scope=str(payload.get("scope", "route")),
    )


def _approval_id_from_token(token: str) -> str:
    approval_id, separator, secret = token.partition(".")
    if not separator or not approval_id.startswith("apr_") or len(secret) < 20:
        raise CrupierError("Approval token is invalid.")
    return approval_id


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _digest(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        values = [_jsonable(item) for item in value]
        return sorted(values, key=repr) if isinstance(value, set) else values
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    raise CrupierError(
        f"Approval payload contains unsupported value type {type(value).__name__}; "
        "use JSON-compatible values or an object with to_dict()."
    )


def _after(seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=int(seconds))).isoformat()


def _parse(value: str | None) -> datetime:
    if not value:
        return datetime.max.replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
