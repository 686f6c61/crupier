from crupier.models import RequestEnvelope
from crupier.structured import schema_from_request


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
