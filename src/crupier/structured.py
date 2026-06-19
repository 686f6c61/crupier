"""Structured output parsing and lightweight JSON Schema validation."""

from __future__ import annotations

import json
from typing import Any

from .adapters.common import build_prompt
from .errors import CrupierStructuredOutputError
from .models import RequestEnvelope


def schema_from_request(request: RequestEnvelope) -> dict[str, Any] | None:
    raw_schema = request.constraints.get("response_schema") or request.response_schema
    if raw_schema is None:
        return None
    if isinstance(raw_schema, dict):
        return raw_schema
    model_json_schema = getattr(raw_schema, "model_json_schema", None)
    if callable(model_json_schema):
        return dict(model_json_schema())
    schema = getattr(raw_schema, "schema", None)
    if callable(schema):
        return dict(schema())
    if raw_schema is object:
        return {"type": "object"}
    raise CrupierStructuredOutputError(f"Unsupported response_schema type: {type(raw_schema).__name__}")


def build_structured_prompt(request: RequestEnvelope, schema: dict[str, Any]) -> str:
    return build_prompt(
        request,
        extra=(
            "Return only a valid JSON value matching this JSON Schema. "
            "Do not include markdown, prose, comments, or hidden reasoning.\n"
            "JSON Schema:\n"
            + json.dumps(schema, ensure_ascii=False, sort_keys=True)
        ),
    )


def build_repair_prompt(request: RequestEnvelope, schema: dict[str, Any], *, bad_output: str, error: str) -> str:
    return build_prompt(
        request,
        extra=(
            "The previous output was invalid for the requested JSON Schema.\n"
            f"Validation error: {error}\n"
            "Previous output:\n"
            f"{_truncate(bad_output, 3000)}\n\n"
            "Return only repaired valid JSON matching this JSON Schema:\n"
            + json.dumps(schema, ensure_ascii=False, sort_keys=True)
        ),
    )


def parse_and_validate_json(text: str, schema: dict[str, Any]) -> Any:
    data = _extract_json(text)
    _validate_json_schema(data, schema, path="$")
    return data


def _extract_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start_candidates = [index for index in [stripped.find("{"), stripped.find("[")] if index >= 0]
        if not start_candidates:
            raise CrupierStructuredOutputError("Structured output did not contain JSON.")
        start = min(start_candidates)
        end = max(stripped.rfind("}"), stripped.rfind("]"))
        if end <= start:
            raise CrupierStructuredOutputError("Structured output did not contain a complete JSON value.")
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise CrupierStructuredOutputError(f"Structured output JSON parse failed: {exc}") from exc


def _validate_json_schema(value: Any, schema: dict[str, Any], *, path: str) -> None:
    expected_type = schema.get("type")
    if expected_type:
        _validate_type(value, expected_type, path=path)

    if expected_type == "object" or "properties" in schema:
        if not isinstance(value, dict):
            raise CrupierStructuredOutputError(f"{path} must be an object.")
        for key in schema.get("required", []):
            if key not in value:
                raise CrupierStructuredOutputError(f"{path}.{key} is required.")
        properties = schema.get("properties", {})
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, dict):
                _validate_json_schema(value[key], child_schema, path=f"{path}.{key}")
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(properties)
            if extras:
                raise CrupierStructuredOutputError(f"{path} contains unexpected keys: {', '.join(sorted(extras))}.")

    if expected_type == "array" or "items" in schema:
        if not isinstance(value, list):
            raise CrupierStructuredOutputError(f"{path} must be an array.")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_json_schema(item, item_schema, path=f"{path}[{index}]")

    if "enum" in schema and value not in schema["enum"]:
        raise CrupierStructuredOutputError(f"{path} must be one of {schema['enum']!r}.")


def _validate_type(value: Any, expected_type: str | list[str], *, path: str) -> None:
    if isinstance(expected_type, list):
        errors = []
        for item in expected_type:
            try:
                _validate_type(value, item, path=path)
                return
            except CrupierStructuredOutputError as exc:
                errors.append(str(exc))
        raise CrupierStructuredOutputError("; ".join(errors))

    checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "number": lambda item: isinstance(item, int | float) and not isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    check = checks.get(expected_type)
    if check is None:
        return
    if not check(value):
        raise CrupierStructuredOutputError(f"{path} must be {expected_type}.")


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
