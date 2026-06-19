"""Opt-in persistent trace storage.

Trace persistence is intentionally conservative: metadata can be stored without
raw prompts or responses, while replay requires explicit prompt storage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import CrupierError
from .models import CrupierResult, RequestEnvelope


@dataclass(slots=True)
class StoredTraceRef:
    trace_id: str
    path: Path
    created_at: str | None = None
    strategy: str | None = None
    models: list[str] | None = None
    replayable: bool = False
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "path": str(self.path),
            "created_at": self.created_at,
            "strategy": self.strategy,
            "models": self.models or [],
            "replayable": self.replayable,
            "summary": self.summary,
        }


class TraceStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def should_store(self, storage_decision: dict[str, Any]) -> bool:
        return bool(storage_decision.get("store_trace", False))

    def write(
        self,
        *,
        project: str,
        request: RequestEnvelope,
        result: CrupierResult,
        dry_run: bool,
        trace_level: bool | str,
    ) -> Path | None:
        if result.trace is None:
            return None
        decision = result.trace.storage_decision
        if not self.should_store(decision):
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        trace_id = result.trace.trace_id
        path = self.root / f"{_safe_trace_id(trace_id)}.json"
        record = self._record(project=project, request=request, result=result, dry_run=dry_run, trace_level=trace_level)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def list(self) -> list[StoredTraceRef]:
        refs: list[StoredTraceRef] = []
        if not self.root.exists():
            return refs
        for path in sorted(self.root.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            route = data.get("result", {}).get("route") or {}
            refs.append(
                StoredTraceRef(
                    trace_id=str(data.get("trace_id") or path.stem),
                    path=path,
                    created_at=data.get("created_at"),
                    strategy=route.get("strategy"),
                    models=_route_models(route),
                    replayable=bool(data.get("replayable", False)),
                    summary=str(data.get("request", {}).get("summary", "")),
                )
            )
        return refs

    def read(self, trace_id: str) -> dict[str, Any]:
        path = self._path(trace_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def delete(self, trace_id: str) -> Path:
        path = self._path(trace_id)
        path.unlink()
        return path

    def replay(
        self,
        trace_id: str,
        client: Any,
        *,
        dry_run: bool = True,
        trace: bool | str = "summary",
    ) -> CrupierResult:
        record = self.read(trace_id)
        if not record.get("replayable"):
            raise CrupierError("Trace is not replayable because prompt/input storage was disabled.")
        request = record.get("request", {})
        if request.get("has_tools"):
            raise CrupierError("Trace replay for tool-bearing requests is not supported yet.")
        constraints = dict(request.get("constraints", {}))
        constraints["store_trace"] = False
        return client.deal(
            task=str(request["task"]),
            input=request.get("input"),
            messages=list(request.get("messages", [])),
            files=[item.get("uri") for item in request.get("files", []) if item.get("uri")],
            mode=request.get("mode"),
            strategy=request.get("strategy"),
            response_schema=request.get("response_schema"),
            constraints=constraints,
            dry_run=dry_run,
            trace=trace,
        )

    def _record(
        self,
        *,
        project: str,
        request: RequestEnvelope,
        result: CrupierResult,
        dry_run: bool,
        trace_level: bool | str,
    ) -> dict[str, Any]:
        assert result.trace is not None
        decision = result.trace.storage_decision
        store_prompt = bool(decision.get("store_prompt", False))
        store_response = bool(decision.get("store_response", False))
        replayable = store_prompt and not request.tools
        return {
            "schema_version": 1,
            "trace_id": result.trace.trace_id,
            "created_at": datetime.now(UTC).isoformat(),
            "project": project,
            "dry_run": dry_run,
            "trace_level": trace_level,
            "replayable": replayable,
            "storage_decision": dict(decision),
            "request": _request_record(request, store_prompt=store_prompt),
            "result": _result_record(result, store_response=store_response, store_prompt=store_prompt),
            "trace": _jsonable(_redact_value(result.trace.to_dict(summary=trace_level != "debug"))),
        }

    def _path(self, trace_id: str) -> Path:
        safe = _safe_trace_id(trace_id)
        path = self.root / f"{safe}.json"
        if not path.exists():
            raise CrupierError(f"Trace {trace_id!r} was not found.")
        return path


def _request_record(request: RequestEnvelope, *, store_prompt: bool) -> dict[str, Any]:
    data: dict[str, Any] = {
        "summary": _redact(_summarize(request.task)),
        "mode": request.mode,
        "strategy": request.strategy,
                "constraints": _jsonable(_safe_constraints(request.constraints)),
        "response_schema": request.response_schema if isinstance(request.response_schema, dict) else None,
        "has_tools": bool(request.tools),
        "files": [asset.to_dict(include_uri=store_prompt) for asset in (request.file_plan.assets if request.file_plan else request.files)],
        "file_plan": request.file_plan.to_dict(include_uri=store_prompt) if request.file_plan else None,
    }
    file_context = request.metadata.get("extracted_file_context") if request.metadata else None
    if isinstance(file_context, dict):
        data["file_context"] = {
            "files": file_context.get("files", []),
            "warnings": file_context.get("warnings", []),
            "max_chars": file_context.get("max_chars"),
        }
    if store_prompt:
        data.update(
            {
                "task": _redact(request.task),
                "input": _jsonable(_redact_value(request.input)),
                "messages": _jsonable(_redact_value(request.messages)),
            }
        )
    return data


def _result_record(result: CrupierResult, *, store_response: bool, store_prompt: bool) -> dict[str, Any]:
    metadata = dict(result.provider_metadata)
    if "tool_calls" in metadata:
        metadata["tool_calls"] = [_safe_tool_call(call, store_prompt=store_prompt, store_response=store_response) for call in metadata["tool_calls"]]
    data: dict[str, Any] = {
        "route": result.route.to_dict() if result.route else None,
        "cost": result.cost.to_dict(),
        "latency_ms": result.latency_ms,
        "warnings": result.warnings,
        "provider_metadata": _jsonable(metadata),
    }
    if store_response:
        data["output_text"] = _redact(result.output_text)
        data["output_json"] = _jsonable(_redact_value(result.output_json))
    return data


def _safe_constraints(constraints: dict[str, Any]) -> dict[str, Any]:
    blocked = {"api_key", "authorization", "headers"}
    return {
        key: _redact_value(value)
        for key, value in constraints.items()
        if key not in blocked and not key.lower().endswith("_api_key")
    }


def _safe_tool_call(call: dict[str, Any], *, store_prompt: bool, store_response: bool) -> dict[str, Any]:
    data = {
        "idempotency_key": call.get("idempotency_key"),
        "name": call.get("name"),
        "status": call.get("status"),
        "requires_approval": call.get("requires_approval", False),
        "error": call.get("error"),
    }
    if store_prompt and "arguments" in call:
        data["arguments"] = _redact_value(call["arguments"])
    if store_response and "result" in call:
        data["result"] = _redact_value(call["result"])
    return {key: value for key, value in data.items() if value is not None}


def _route_models(route: dict[str, Any]) -> list[str]:
    models: list[str] = []
    for step in route.get("steps", []) or []:
        for model in [step.get("model"), *(step.get("models") or [])]:
            if model and model not in models:
                models.append(model)
    return models


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    return value


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [_jsonable(item) for item in value]
        return repr(value)


def _redact(text: str) -> str:
    redacted = text
    for pattern, replacement in _SECRET_REPLACERS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _summarize(text: str) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= 180 else compact[:177] + "..."


def _safe_trace_id(trace_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", trace_id)


_SECRET_REPLACERS = (
    (re.compile(("s" + "k-") + r"[A-Za-z0-9_\-]{10,}"), "[redacted]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{12,}", re.IGNORECASE), r"\1[redacted]"),
    (re.compile(r"([A-Z][A-Z0-9_]*_API_KEY=)[^\s]+"), r"\1[redacted]"),
)
