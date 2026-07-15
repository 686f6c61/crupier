from datetime import date, datetime, timezone

import pytest

from crupier.errors import CrupierModelUnsupportedError, CrupierRouteValidationError, CrupierToolApprovalRequired
from crupier.models import RequestEnvelope
from crupier.tools import (
    ToolCallRequest,
    ToolExecution,
    ToolSpec,
    build_tool_final_prompt,
    build_tool_planning_prompt,
    build_tool_replanning_prompt,
    execute_tool_plan,
    extract_final_answer,
    idempotency_key,
    normalize_tools,
    parse_tool_plan,
)


def test_normalize_callable_builds_schema_from_signature():
    def calculate(
        count: int,
        ratio: float = 1.0,
        enabled: bool = False,
        tags: list = (),
        payload: dict = None,
        label="default",
        **extras,
    ):
        """Calculate a result."""
        return count, ratio, enabled, tags, payload, label, extras

    spec = normalize_tools([calculate])[0]

    assert spec.name == "calculate"
    assert spec.description == "Calculate a result."
    assert spec.parameters == {
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "enabled": {"type": "boolean"},
            "tags": {"type": "array"},
            "payload": {"type": "object"},
            "label": {"type": "string"},
        },
        "required": ["count"],
    }
    assert spec.handler is calculate


def test_normalize_openai_function_wrapper_preserves_crupier_metadata():
    def search(query):
        return query

    spec = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search records",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                    "requires_approval": True,
                },
                "handler": search,
                "side_effects": True,
            }
        ]
    )[0]

    assert spec.public_dict() == {
        "name": "search",
        "description": "Search records",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        "requires_approval": True,
        "side_effects": True,
    }
    assert spec.handler is search


@pytest.mark.parametrize(
    ("tool", "message"),
    [
        (42, "Unsupported tool definition type: int"),
        ({"name": ""}, "Tool name cannot be empty"),
        ({"type": "function", "function": []}, "definition must be an object"),
        ({"name": "x", "parameters": []}, "parameters must be a JSON Schema object"),
    ],
)
def test_normalize_tools_rejects_invalid_definitions(tool, message):
    with pytest.raises(CrupierModelUnsupportedError, match=message):
        normalize_tools([tool])


def test_normalize_tools_rejects_duplicate_names_and_ignores_noncallable_handler():
    with pytest.raises(CrupierModelUnsupportedError, match="Duplicate tool name 'same'"):
        normalize_tools([{"name": "same"}, {"name": "same"}])

    assert normalize_tools([{"name": "safe", "handler": "not-callable"}])[0].handler is None


def test_tool_prompts_include_catalog_history_and_response_schema():
    request = RequestEnvelope(task="Find customer", input="Ada")
    tools = [ToolSpec(name="lookup", description="Lookup", handler=lambda: None)]
    schema = {"type": "object", "required": ["answer"]}
    execution = ToolExecution(
        idempotency_key="abc",
        name="lookup",
        arguments={},
        status="completed",
        result={"name": "Ada"},
    )

    planning = build_tool_planning_prompt(request, tools, response_schema=schema)
    replanning = build_tool_replanning_prompt(request, tools, [execution], response_schema=schema)
    final = build_tool_final_prompt(request, [execution], response_schema=schema)

    assert "Available tools" in planning and '"lookup"' in planning and '"answer"' in planning
    assert "Tool execution results so far" in replanning and '"Ada"' in replanning
    assert "Do not repeat completed tool calls" in replanning
    assert "Produce the final answer" in final and '"answer"' in final


def test_parse_tool_plan_accepts_fences_aliases_and_string_arguments():
    text = r"""```json
    {"tools":[null,{"name":"lookup","arguments":"{\"id\": 7}"},{"arguments":{}}],"final":"done"}
    ```"""

    calls, final = parse_tool_plan(text)

    assert calls == [ToolCallRequest(name="lookup", arguments={"id": 7})]
    assert final == "done"


def test_extract_final_answer_handles_closed_truncated_json_and_plain_envelopes():
    assert extract_final_answer("prefix <final_answer>clean</final_answer> internal") == "clean"
    assert extract_final_answer("<final_answer>truncated but usable") == "truncated but usable"
    assert extract_final_answer('{"final":"legacy"}\ninternal') == "legacy"
    assert extract_final_answer("plain answer") == "plain answer"


def test_parse_tool_plan_extracts_json_from_prose_and_falls_back_to_plain_answer():
    calls, final = parse_tool_plan('Plan: {"tool_calls":[{"name":"search","arguments":{}}]} end')
    assert calls == [ToolCallRequest(name="search", arguments={})]
    assert final is None

    assert parse_tool_plan("A normal final answer") == ([], "A normal final answer")
    assert parse_tool_plan('[{"name":"not-an-object-plan"}]') == ([], '[{"name":"not-an-object-plan"}]')
    assert parse_tool_plan("Plan: {not-json} end") == ([], "Plan: {not-json} end")


@pytest.mark.parametrize(
    "arguments",
    ['"not-json"', '"[1, 2]"', "[]"],
)
def test_parse_tool_plan_rejects_non_object_arguments(arguments):
    plan = '{"tool_calls":[{"name":"lookup","arguments":' + arguments + "}]}"

    with pytest.raises(CrupierRouteValidationError, match="arguments must be a JSON object"):
        parse_tool_plan(plan)


def test_execute_tool_plan_records_success_failure_and_duplicates():
    def double(value):
        return {"value": value * 2}

    def fail():
        raise RuntimeError("provider record unavailable")

    tools = [ToolSpec(name="double", handler=double), ToolSpec(name="fail", handler=fail)]
    calls = [
        ToolCallRequest(name="double", arguments={"value": 3}),
        ToolCallRequest(name="double", arguments={"value": 3}),
        ToolCallRequest(name="fail"),
    ]

    executions = execute_tool_plan(calls, tools, RequestEnvelope(task="x"))

    assert [item.status for item in executions] == ["completed", "skipped_duplicate", "failed"]
    assert executions[0].result == {"value": 6}
    assert executions[1].result == {"value": 6}
    assert executions[2].error == "provider record unavailable"


def test_execute_tool_plan_skips_completed_execution_from_previous_round():
    call = ToolCallRequest(name="lookup", arguments={"id": 1})
    previous = ToolExecution(
        idempotency_key=idempotency_key(call.name, call.arguments),
        name=call.name,
        arguments=call.arguments,
        status="completed",
        result="cached",
        requires_approval=True,
    )

    execution = execute_tool_plan(
        [call],
        [ToolSpec(name="lookup", handler=lambda id: id)],
        RequestEnvelope(task="x"),
        previous_executions=[previous],
    )[0]

    assert execution.status == "skipped_duplicate"
    assert execution.result == "cached"
    assert execution.requires_approval is True


@pytest.mark.parametrize(
    ("tools", "request_envelope", "message", "error_type"),
    [
        ([], RequestEnvelope(task="x"), "unknown tool 'missing'", CrupierModelUnsupportedError),
        (
            [ToolSpec(name="missing", handler=lambda: None)],
            RequestEnvelope(task="x", constraints={"allowed_tools": []}),
            "not in allowed_tools",
            CrupierToolApprovalRequired,
        ),
        (
            [ToolSpec(name="missing", handler=lambda: None, requires_approval=True)],
            RequestEnvelope(task="x"),
            "requires approval",
            CrupierToolApprovalRequired,
        ),
        (
            [ToolSpec(name="missing")],
            RequestEnvelope(task="x"),
            "has no local handler",
            CrupierModelUnsupportedError,
        ),
    ],
)
def test_execute_tool_plan_enforces_catalog_allowlist_approval_and_handler(
    tools, request_envelope, message, error_type
):
    with pytest.raises(error_type, match=message):
        execute_tool_plan([ToolCallRequest(name="missing")], tools, request_envelope)


def test_execute_tool_plan_honors_approval_modes_and_validates_constraints():
    tools = [ToolSpec(name="write", handler=lambda: "ok", side_effects=True)]

    approved = execute_tool_plan(
        [ToolCallRequest(name="write")],
        tools,
        RequestEnvelope(task="x", constraints={"approved_tools": "write"}),
    )
    approve_all = execute_tool_plan(
        [ToolCallRequest(name="write")],
        tools,
        RequestEnvelope(task="x", constraints={"approve_tool_calls": True, "require_approval_for": ["write"]}),
    )

    assert approved[0].status == "completed"
    assert approve_all[0].requires_approval is True
    with pytest.raises(CrupierModelUnsupportedError, match="string or list of names"):
        execute_tool_plan(
            [ToolCallRequest(name="write")],
            tools,
            RequestEnvelope(task="x", constraints={"allowed_tools": 7}),
        )


def test_execute_tool_plan_bounds_large_results_and_long_errors():
    large = execute_tool_plan(
        [ToolCallRequest(name="large")],
        [ToolSpec(name="large", handler=lambda: "x" * 1000)],
        RequestEnvelope(task="x"),
        max_result_chars=10,
    )[0]

    def fail_long():
        raise RuntimeError("e" * 5000)

    failed = execute_tool_plan(
        [ToolCallRequest(name="fail")],
        [ToolSpec(name="fail", handler=fail_long)],
        RequestEnvelope(task="x"),
    )[0]

    assert large.result["truncated"] is True
    assert len(large.result["preview"]) == 256
    assert large.result["original_chars"] > 256
    assert len(failed.error) == 4000
    assert failed.error.endswith("...")


def test_tool_execution_serializes_provider_objects_and_omits_none():
    class Modern:
        def model_dump(self):
            return {"when": date(2026, 7, 15), "values": {3, 1}}

    class Legacy:
        def to_dict(self):
            return {"at": datetime(2026, 7, 15, 12, tzinfo=timezone.utc), "items": (1, 2)}

    modern = ToolExecution("a", "modern", {}, "completed", result=Modern()).to_dict()
    legacy = ToolExecution("b", "legacy", {}, "completed", result=Legacy()).to_dict()
    fallback = ToolExecution("c", "fallback", {}, "completed", result=object()).to_dict()

    assert modern["result"] == {"when": "2026-07-15", "values": [1, 3]}
    assert legacy["result"] == {"at": "2026-07-15T12:00:00+00:00", "items": [1, 2]}
    assert fallback["result"].startswith("<object object at ")
    assert "error" not in fallback


def test_idempotency_key_is_stable_across_argument_order():
    first = idempotency_key("search", {"query": "Ada", "limit": 3})
    second = idempotency_key("search", {"limit": 3, "query": "Ada"})

    assert first == second
    assert len(first) == 16
