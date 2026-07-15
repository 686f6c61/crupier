from crupier import ModelRef
from crupier.models import (
    CapabilityCard,
    CostEstimate,
    DecisionTrace,
    FileAsset,
    FileRepresentation,
    FileRoutingPlan,
    OperationResult,
    PlanningContext,
    RequestEnvelope,
    RoutePlan,
    RouteStep,
    UpdateReport,
)


def test_model_ref_parse_keeps_ollama_tag_colons():
    ref = ModelRef.parse("ollama:qwen3.5:122b")

    assert ref.provider == "ollama"
    assert ref.model == "qwen3.5:122b"
    assert ref.key == "ollama:qwen3.5:122b"


def test_model_ref_detects_latest_alias():
    ref = ModelRef.parse("openai:gpt-latest")

    assert ref.stability == "latest"


def test_model_ref_normalizes_claude_provider_alias():
    ref = ModelRef.parse("claude:claude-opus-4-8")

    assert ref.provider == "anthropic"
    assert ref.key == "anthropic:claude-opus-4-8"


def test_model_ref_rejects_invalid_refs_and_detects_preview_experimental():
    for value in ["missing", ":model", "provider:"]:
        try:
            ModelRef.parse(value)
        except ValueError as exc:
            assert "provider:model" in str(exc)
        else:
            raise AssertionError(f"{value!r} should be rejected")

    assert ModelRef.parse("openai:model-preview").stability == "preview"
    assert ModelRef.parse("openai:model-experimental").stability == "experimental"
    assert str(ModelRef.parse("openai:model")) == "openai:model"


def test_file_models_round_trip_from_dict():
    asset_data = {
        "kind": "pdf",
        "name": "report.pdf",
        "uri": "/tmp/report.pdf",
        "metadata": {"source": "test"},
    }
    representation_data = {
        "asset_name": "report.pdf",
        "kind": "pdf",
        "representation": "native_pdf",
        "required_model_modalities": ["file"],
        "required_model_capabilities": ["file_input"],
        "pipeline": [],
        "warnings": [],
    }
    plan_data = {
        "assets": [asset_data],
        "representations": [representation_data],
        "required_model_modalities": ["file"],
        "extraction_required": False,
    }

    asset = FileAsset.from_dict(asset_data)
    representation = FileRepresentation.from_dict(representation_data)
    plan = FileRoutingPlan.from_dict(plan_data)

    assert asset.to_dict(include_uri=True)["uri"] == "/tmp/report.pdf"
    assert representation.to_dict()["representation"] == "native_pdf"
    assert plan.assets == [asset]
    assert plan.representations == [representation]


def test_planning_context_and_result_models_serialize_full_and_summary_views():
    card = CapabilityCard(model_ref=ModelRef.parse("openai:test"), last_updated="today")
    file_plan = FileRoutingPlan(assets=[FileAsset(kind="text", name="notes.txt")])
    request = RequestEnvelope(task="answer", file_plan=file_plan)
    context = PlanningContext(request=request, candidates=[card], metadata={"source": "test"})

    summary = context.to_dict(summary=True)
    full = context.to_dict(summary=False)

    assert summary["input_plan"]["files"]["assets"][0]["name"] == "notes.txt"
    assert "candidates" not in summary
    assert full["candidates"][0]["model_ref"]["model"] == "test"

    route = RoutePlan(
        strategy="single",
        steps=[RouteStep(role="primary", model="openai:test")],
        estimated_cost=CostEstimate(estimated_usd=0.1),
    )
    trace = DecisionTrace(
        trace_id="trace",
        request_summary="answer",
        route_plan=route,
        provider_calls=[{"model": "openai:test"}],
        errors=[{"message": "none"}],
    )
    assert "provider_calls" not in trace.to_dict(summary=True)
    assert trace.to_dict(summary=False)["provider_calls"] == [{"model": "openai:test"}]

    operation = OperationResult(operation="tts", model="openai:test", data=b"1234", route=route, trace=trace)
    assert operation.to_dict()["data"] == {"bytes": 4}
    assert UpdateReport(changed_models=["openai:test"]).to_dict()["changed_models"] == ["openai:test"]
