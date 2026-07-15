import pytest

from crupier.errors import CrupierStructuredOutputError
from crupier.models import RequestEnvelope
from crupier.structured import (
    build_repair_prompt,
    build_structured_prompt,
    parse_and_validate_json,
    schema_from_request,
)


def test_schema_from_request_normalizes_openai_json_schema_format():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    request = RequestEnvelope(
        task="x",
        response_schema={
            "type": "json_schema",
            "json_schema": {"name": "person", "schema": schema, "strict": True},
        },
    )

    assert schema_from_request(request) == schema


def test_schema_from_request_normalizes_json_object_format():
    request = RequestEnvelope(task="x", response_schema={"type": "json_object"})

    assert schema_from_request(request) == {"type": "object"}


def test_schema_from_request_supports_direct_and_response_format_schemas():
    direct = {"type": "array", "items": {"type": "integer"}}

    assert schema_from_request(RequestEnvelope(task="x", response_schema=direct)) is direct
    assert schema_from_request(
        RequestEnvelope(task="x", response_schema={"type": "json_schema", "schema": direct})
    ) == direct
    assert schema_from_request(
        RequestEnvelope(task="x", response_schema={"type": "json_schema", "json_schema": direct})
    ) == direct
    assert schema_from_request(RequestEnvelope(task="x")) is None


def test_schema_from_request_prefers_constraint_and_supports_model_classes():
    class ModernModel:
        @classmethod
        def model_json_schema(cls):
            return {"type": "object", "title": cls.__name__}

    class LegacyModel:
        @classmethod
        def schema(cls):
            return {"type": "string", "title": cls.__name__}

    constrained = RequestEnvelope(
        task="x",
        response_schema={"type": "string"},
        constraints={"response_schema": ModernModel},
    )

    assert schema_from_request(constrained) == {"type": "object", "title": "ModernModel"}
    assert schema_from_request(RequestEnvelope(task="x", response_schema=LegacyModel)) == {
        "type": "string",
        "title": "LegacyModel",
    }
    assert schema_from_request(RequestEnvelope(task="x", response_schema=object)) == {"type": "object"}


def test_schema_from_request_rejects_unsupported_schema_type():
    with pytest.raises(CrupierStructuredOutputError, match="Unsupported response_schema type: int"):
        schema_from_request(RequestEnvelope(task="x", response_schema=42))


def test_structured_prompts_include_schema_error_and_bounded_bad_output():
    request = RequestEnvelope(task="Extract", input="record")
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}

    prompt = build_structured_prompt(request, schema)
    repair = build_repair_prompt(request, schema, bad_output="x" * 4000, error="$.name must be string")

    assert "Return only a valid JSON value" in prompt
    assert '"name"' in prompt
    assert "$.name must be string" in repair
    assert "x" * 2997 + "..." in repair
    assert "x" * 3000 not in repair


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('```json\n{"ok": true}\n```', {"ok": True}),
        ('Result: {"ok": true} done.', {"ok": True}),
        ('Result: [1, 2, 3] done.', [1, 2, 3]),
    ],
)
def test_parse_and_validate_json_extracts_common_model_wrappers(text, expected):
    assert parse_and_validate_json(text, {}) == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("there is no value", "did not contain JSON"),
        ('prefix {"open": true', "complete JSON value"),
        ('prefix {not-json} suffix', "JSON parse failed"),
    ],
)
def test_parse_and_validate_json_reports_distinct_parse_failures(text, message):
    with pytest.raises(CrupierStructuredOutputError, match=message):
        parse_and_validate_json(text, {})


def test_json_schema_validates_nested_objects_arrays_and_closed_properties():
    schema = {
        "type": "object",
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["status"],
                    "properties": {"status": {"type": "string", "enum": ["ok", "failed"]}},
                    "additionalProperties": False,
                },
            }
        },
        "additionalProperties": False,
    }

    assert parse_and_validate_json('{"items":[{"status":"ok"}]}', schema) == {
        "items": [{"status": "ok"}]
    }
    with pytest.raises(CrupierStructuredOutputError, match=r"\$\.items is required"):
        parse_and_validate_json("{}", schema)
    with pytest.raises(CrupierStructuredOutputError, match="unexpected keys: extra"):
        parse_and_validate_json('{"items":[],"extra":1}', schema)
    with pytest.raises(CrupierStructuredOutputError, match=r"\$\.items\[0\]\.status must be one of"):
        parse_and_validate_json('{"items":[{"status":"unknown"}]}', schema)
    with pytest.raises(CrupierStructuredOutputError, match="unexpected keys: extra"):
        parse_and_validate_json('{"items":[{"status":"ok","extra":1}]}', schema)


@pytest.mark.parametrize(
    ("value", "schema", "message"),
    [
        ('"text"', {"type": "object"}, "must be object"),
        ('{"value":1}', {"type": "array"}, "must be array"),
        ("true", {"type": "integer"}, "must be integer"),
        ("1", {"type": "boolean"}, "must be boolean"),
        ("1.5", {"type": "integer"}, "must be integer"),
        ('"1"', {"type": "number"}, "must be number"),
        ("null", {"type": "string"}, "must be string"),
    ],
)
def test_json_schema_rejects_wrong_primitive_types(value, schema, message):
    with pytest.raises(CrupierStructuredOutputError, match=message):
        parse_and_validate_json(value, schema)


@pytest.mark.parametrize(
    ("value", "schema"),
    [
        ("1", {"type": "integer"}),
        ("1.5", {"type": "number"}),
        ("true", {"type": "boolean"}),
        ("null", {"type": "null"}),
        ('"x"', {"type": "custom-provider-type"}),
        ("null", {"type": ["object", "null"], "properties": {"x": {"type": "string"}}}),
        ('"x"', {"type": ["string", "null"]}),
    ],
)
def test_json_schema_accepts_primitive_unknown_and_union_types(value, schema):
    parse_and_validate_json(value, schema)


def test_json_schema_union_reports_each_failed_alternative():
    with pytest.raises(CrupierStructuredOutputError) as exc_info:
        parse_and_validate_json("true", {"type": ["string", "integer"]})

    assert "$ must be string" in str(exc_info.value)
    assert "$ must be integer" in str(exc_info.value)
