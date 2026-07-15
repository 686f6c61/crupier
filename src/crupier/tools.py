"""Normalized local tool execution for agent routes.

The first production-safe tool loop is provider-agnostic: Crupier asks the
selected model for a JSON tool plan, executes approved local tools, then asks
the model for a final answer using the tool results. Provider-native tool call
mapping can later optimize this path without changing the public SDK contract.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable

from .adapters.common import build_prompt
from .errors import CrupierModelUnsupportedError, CrupierRouteValidationError, CrupierToolApprovalRequired
from .models import RequestEnvelope


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    handler: Callable[..., Any] | None = None
    requires_approval: bool = False
    side_effects: bool = False

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_approval": self.requires_approval,
            "side_effects": self.side_effects,
        }


@dataclass(slots=True)
class ToolCallRequest:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecution:
    idempotency_key: str
    name: str
    arguments: dict[str, Any]
    status: str
    result: Any = None
    error: str | None = None
    requires_approval: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = {
            "idempotency_key": self.idempotency_key,
            "name": self.name,
            "arguments": self.arguments,
            "status": self.status,
            "result": _jsonable(self.result),
            "error": self.error,
            "requires_approval": self.requires_approval,
        }
        return {key: value for key, value in data.items() if value is not None}


def normalize_tools(tools: list[Any]) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for item in tools:
        if callable(item):
            specs.append(_spec_from_callable(item))
        elif isinstance(item, dict):
            specs.append(_spec_from_dict(item))
        else:
            raise CrupierModelUnsupportedError(f"Unsupported tool definition type: {type(item).__name__}")
    seen: set[str] = set()
    for spec in specs:
        if not spec.name:
            raise CrupierModelUnsupportedError("Tool name cannot be empty.")
        if spec.name in seen:
            raise CrupierModelUnsupportedError(f"Duplicate tool name {spec.name!r}.")
        seen.add(spec.name)
    return specs


def build_tool_planning_prompt(
    request: RequestEnvelope,
    tools: list[ToolSpec],
    *,
    response_schema: dict[str, Any] | None = None,
) -> str:
    tool_catalog = [tool.public_dict() for tool in tools]
    extra = (
        "Available tools:\n"
        + json.dumps(tool_catalog, ensure_ascii=False, sort_keys=True)
        + "\n\nReturn only JSON with this shape:\n"
        '{"tool_calls":[{"name":"tool_name","arguments":{}}],"final":"answer if no tool is needed"}\n'
        "Use tool_calls only when a tool result is needed. Do not invent tool names or arguments."
    )
    if response_schema:
        extra += (
            "\nThe final answer must ultimately be JSON matching this schema:\n"
            + json.dumps(response_schema, ensure_ascii=False, sort_keys=True)
        )
    return build_prompt(request, extra=extra)


def build_tool_final_prompt(
    request: RequestEnvelope,
    executions: list[ToolExecution],
    *,
    response_schema: dict[str, Any] | None = None,
) -> str:
    extra = (
        "Tool execution results:\n"
        + json.dumps([execution.to_dict() for execution in executions], ensure_ascii=False, sort_keys=True)
        + "\n\nProduce the final answer for the user. Do not request more tools."
    )
    if response_schema:
        extra += (
            "\nReturn only valid JSON matching this schema:\n"
            + json.dumps(response_schema, ensure_ascii=False, sort_keys=True)
        )
    return build_prompt(request, extra=extra)


def build_tool_replanning_prompt(
    request: RequestEnvelope,
    tools: list[ToolSpec],
    executions: list[ToolExecution],
    *,
    response_schema: dict[str, Any] | None = None,
) -> str:
    tool_catalog = [tool.public_dict() for tool in tools]
    extra = (
        "Available tools:\n"
        + json.dumps(tool_catalog, ensure_ascii=False, sort_keys=True)
        + "\n\nTool execution results so far:\n"
        + json.dumps([execution.to_dict() for execution in executions], ensure_ascii=False, sort_keys=True)
        + "\n\nReturn only JSON with this shape:\n"
        '{"tool_calls":[{"name":"tool_name","arguments":{}}],"final":"answer if enough information is available"}\n'
        "Request more tool_calls only if the previous results are insufficient. Do not repeat completed tool calls."
    )
    if response_schema:
        extra += (
            "\nThe final answer must ultimately be JSON matching this schema:\n"
            + json.dumps(response_schema, ensure_ascii=False, sort_keys=True)
        )
    return build_prompt(request, extra=extra)


def parse_tool_plan(text: str) -> tuple[list[ToolCallRequest], str | None]:
    try:
        data = _extract_json_object(text)
    except CrupierRouteValidationError:
        return [], text

    raw_calls = data.get("tool_calls") or data.get("tools") or []
    calls: list[ToolCallRequest] = []
    if isinstance(raw_calls, list):
        for item in raw_calls:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            arguments = item.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    raise CrupierRouteValidationError(
                        f"Tool {name or '<unknown>'!r} arguments must be a JSON object."
                    )
            if name:
                if not isinstance(arguments, dict):
                    raise CrupierRouteValidationError(f"Tool {name!r} arguments must be a JSON object.")
                calls.append(ToolCallRequest(name=str(name), arguments=dict(arguments)))
    final = data.get("final") if isinstance(data.get("final"), str) else None
    return calls, final


def extract_final_answer(text: str) -> str:
    opening = "<final_answer>"
    closing = "</final_answer>"
    if opening in text:
        remainder = text.split(opening, 1)[1]
        final = remainder.split(closing, 1)[0] if closing in remainder else remainder
        return final.strip()
    _, parsed_final = parse_tool_plan(text)
    return parsed_final.strip() if parsed_final and parsed_final.strip() else text.strip()


def execute_tool_plan(
    calls: list[ToolCallRequest],
    tools: list[ToolSpec],
    request: RequestEnvelope,
    *,
    previous_executions: list[ToolExecution] | None = None,
    max_result_chars: int = 50_000,
) -> list[ToolExecution]:
    by_name = {tool.name: tool for tool in tools}
    allowed_tools = _name_set(request.constraints.get("allowed_tools"), default=set(by_name))
    approved_tools = _name_set(request.constraints.get("approved_tools"), default=set())
    approve_all = bool(request.constraints.get("approve_tool_calls", False))
    require_approval_for = _name_set(request.constraints.get("require_approval_for"), default=set())

    executions: list[ToolExecution] = []
    completed_by_key: dict[str, ToolExecution] = {
        execution.idempotency_key: execution
        for execution in previous_executions or []
        if execution.status in {"completed", "skipped_duplicate"}
    }
    for call in calls:
        if call.name not in by_name:
            raise CrupierModelUnsupportedError(f"Model requested unknown tool {call.name!r}.")
        if call.name not in allowed_tools:
            raise CrupierToolApprovalRequired(f"Tool {call.name!r} is not in allowed_tools.")
        spec = by_name[call.name]
        key = idempotency_key(call.name, call.arguments)
        if key in completed_by_key:
            previous = completed_by_key[key]
            executions.append(
                ToolExecution(
                    idempotency_key=key,
                    name=call.name,
                    arguments=call.arguments,
                    status="skipped_duplicate",
                    result=previous.result,
                    requires_approval=previous.requires_approval,
                )
            )
            continue
        requires_approval = spec.requires_approval or spec.side_effects or call.name in require_approval_for
        if requires_approval and not (approve_all or call.name in approved_tools):
            raise CrupierToolApprovalRequired(
                f"Tool {call.name!r} requires approval. Pass constraints.approved_tools or approve_tool_calls."
            )
        if spec.handler is None:
            raise CrupierModelUnsupportedError(f"Tool {call.name!r} has no local handler.")
        try:
            result = _bounded_tool_result(spec.handler(**call.arguments), max_chars=max_result_chars)
            execution = ToolExecution(
                idempotency_key=key,
                name=call.name,
                arguments=call.arguments,
                status="completed",
                result=result,
                requires_approval=requires_approval,
            )
        except Exception as exc:  # noqa: BLE001 - tool exceptions become ledger entries
            execution = ToolExecution(
                idempotency_key=key,
                name=call.name,
                arguments=call.arguments,
                status="failed",
                error=_truncate(str(exc), 4000),
                requires_approval=requires_approval,
            )
        executions.append(execution)
        completed_by_key[key] = execution
    return executions


def idempotency_key(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _spec_from_callable(func: Callable[..., Any]) -> ToolSpec:
    return ToolSpec(
        name=getattr(func, "__name__", "tool"),
        description=(inspect.getdoc(func) or "")[:500],
        parameters=_schema_from_signature(func),
        handler=func,
    )


def _spec_from_dict(data: dict[str, Any]) -> ToolSpec:
    function_data = data.get("function") if data.get("type") == "function" else data
    if not isinstance(function_data, dict):
        raise CrupierModelUnsupportedError("Function tool definition must be an object.")
    handler = function_data.get("handler") or data.get("handler") or data.get("callable")
    if handler is not None and not callable(handler):
        handler = None
    parameters = function_data.get("parameters", {"type": "object", "properties": {}})
    if not isinstance(parameters, dict):
        raise CrupierModelUnsupportedError("Function tool parameters must be a JSON Schema object.")
    return ToolSpec(
        name=str(function_data.get("name", "")),
        description=str(function_data.get("description", "")),
        parameters=dict(parameters),
        handler=handler,
        requires_approval=bool(function_data.get("requires_approval", data.get("requires_approval", False))),
        side_effects=bool(function_data.get("side_effects", data.get("side_effects", False))),
    )


def _schema_from_signature(func: Callable[..., Any]) -> dict[str, Any]:
    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
            continue
        properties[name] = {"type": _json_type(parameter.annotation)}
        if parameter.default is parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def _json_type(annotation: Any) -> str:
    if annotation in {int}:
        return "integer"
    if annotation in {float}:
        return "number"
    if annotation in {bool}:
        return "boolean"
    if annotation in {list}:
        return "array"
    if annotation in {dict}:
        return "object"
    return "string"


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise CrupierRouteValidationError("Tool plan did not contain a JSON object.")
        try:
            data = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise CrupierRouteValidationError(f"Tool plan JSON parse failed: {exc}") from exc
    if not isinstance(data, dict):
        raise CrupierRouteValidationError("Tool plan must be a JSON object.")
    return data


def _name_set(value: Any, *, default: set[str]) -> set[str]:
    if value is None:
        return set(default)
    if isinstance(value, str):
        return {value}
    try:
        return {str(item) for item in value}
    except TypeError as exc:
        raise CrupierModelUnsupportedError("Tool allow/approval constraints must be a string or list of names.") from exc


def _bounded_tool_result(value: Any, *, max_chars: int) -> Any:
    safe = _jsonable(value)
    encoded = json.dumps(safe, ensure_ascii=False, sort_keys=True)
    limit = max(256, int(max_chars))
    if len(encoded) <= limit:
        return safe
    return {
        "truncated": True,
        "original_chars": len(encoded),
        "preview": encoded[:limit],
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    return repr(value)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
