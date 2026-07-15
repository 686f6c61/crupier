from io import BytesIO
from pathlib import Path

import pytest

from crupier import Crupier
from crupier.adapters import AdapterResponse, EmbeddingResponse, OperationResponse
from crupier.config import CrupierConfig
from crupier.errors import CrupierModelUnsupportedError, CrupierPolicyError, CrupierRouteValidationError
from crupier.models import CapabilityCard, CostEstimate, ModelRef, RequestEnvelope, RoutePlan, RouteStep
from crupier.operations import (
    OperationRouter,
    _deterministic_operation,
    _file_value,
    _operation_card_incompatibility,
    _operation_payload,
    _parse_operation_classification,
    _planning_payload,
    _planning_value,
    _successful_orchestrator,
    _validate_operation_payload,
    normalize_operation,
)


class OperationAdapter:
    provider = "nan"

    def __init__(self, *, error: Exception | None = None):
        self.error = error

    @staticmethod
    def supports_operation(*, operation, model):
        return operation in {"reranker", "transcription", "tts", "image_generation"}

    def execute_operation(self, *, operation, model, request, payload):
        if self.error:
            raise self.error
        return OperationResponse(operation=operation, output={"ok": True}, metadata={"model": model})


class ChatClassifier:
    provider = "openai"

    def __init__(self, text: str, *, error: Exception | None = None):
        self.text = text
        self.error = error

    def generate(self, *, model, prompt, request):
        if self.error:
            raise self.error
        return AdapterResponse(text=self.text, metadata={"model": model})


class EmbeddingOperationAdapter(OperationAdapter):
    def embed(self, *, model, input, dimensions=None):
        return EmbeddingResponse(embeddings=[[1.0]], metadata={"model": model})


def make_config(tmp_path, *, allow=None) -> CrupierConfig:
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "operations-edges"},
            "providers": {
                "nan": {"enabled": True, "env_key": "NAN_API_KEY"},
                "openai": {"enabled": True, "env_key": "OPENAI_API_KEY"},
            },
            "models": {"allow": allow or ["nan:rerank"]},
            "routing": {
                "require_operational_providers": False,
                "max_calls": 4,
                "max_latency_ms": 5000,
            },
        }
    )
    config.root = tmp_path
    return config


def make_client(tmp_path, *, adapter=None, allow=None, adapters=None) -> Crupier:
    configured = adapters if adapters is not None else {"nan": adapter or OperationAdapter()}
    return Crupier(make_config(tmp_path, allow=allow), adapters=configured)


def test_specialized_route_rejects_non_single_plan_and_non_list_planning_calls(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    router = client.operations

    def invalid_plan(request, cards, filters, *, dry_run):
        request.metadata["_crupier_orchestrator_calls"] = "invalid"
        return RoutePlan(
            strategy="fallback",
            steps=[RouteStep(role="fallback", models=["nan:rerank"])],
            estimated_cost=CostEstimate(),
        )

    monkeypatch.setattr(router, "_plan", invalid_plan)
    monkeypatch.setattr(client.policy, "validate_route", lambda *args: None)

    with pytest.raises(CrupierRouteValidationError, match="requires one selected model"):
        router.execute(
            "reranker",
            task="rank",
            payload={"query": "q", "documents": ["a"]},
            dry_run=True,
        )


def test_non_list_planning_calls_are_discarded_for_valid_dry_run(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    router = client.operations

    def valid_plan(request, cards, filters, *, dry_run):
        request.metadata["_crupier_orchestrator_calls"] = "invalid"
        return RoutePlan(strategy="single", steps=[RouteStep(role="primary", model="nan:rerank")])

    monkeypatch.setattr(router, "_plan", valid_plan)
    result = router.execute(
        "reranker",
        task="rank",
        payload={"query": "q", "documents": ["a"]},
        dry_run=True,
    )

    assert result.trace.provider_calls == []


@pytest.mark.parametrize(("operation", "allow", "payload", "message"), [
    ("embedding", ["nan:qwen3-embedding"], {"input": "x"}, "no embedding execution method"),
    ("reranker", ["nan:rerank"], {"query": "q", "documents": ["a"]}, "no 'reranker' execution method"),
])
def test_execute_rejects_missing_adapter_operation_method(tmp_path, monkeypatch, operation, allow, payload, message):
    client = make_client(tmp_path, allow=allow, adapters={"nan": object()})
    router = client.operations
    monkeypatch.setattr(
        router,
        "_filter_operation_candidates",
        lambda operation, cards, payload: (cards, []),
    )

    with pytest.raises(CrupierModelUnsupportedError, match=message):
        router.execute(operation, task="x", payload=payload)


def test_operation_execution_failure_is_traced_before_reraising(tmp_path, monkeypatch):
    client = make_client(tmp_path, adapter=OperationAdapter(error=RuntimeError("operation failed")))
    captured = {}

    def capture(result, *, request, dry_run, trace):
        captured["trace"] = result.trace
        return result

    monkeypatch.setattr(client.operations, "_finalize_result", capture)

    with pytest.raises(RuntimeError, match="operation failed"):
        client.rerank(query="q", documents=["a"])

    # Execution errors are attached to the in-flight trace even though the
    # provider exception remains the public failure.
    assert "trace" not in captured


def test_run_explicit_chat_and_specialized_alias_paths(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    seen = {}

    def fake_deal(task, input=None, **kwargs):
        seen["chat"] = kwargs
        return "chat-result"

    def fake_execute(operation, **kwargs):
        seen["operation"] = operation
        return "operation-result"

    monkeypatch.setattr(client, "deal", fake_deal)
    monkeypatch.setattr(client.operations, "execute", fake_execute)

    assert client.run("x", operation="chat", dry_run=True) == "chat-result"
    assert client.run("x", input="hello", operation="embeddings", dry_run=True) == "operation-result"
    assert seen["operation"] == "embedding"


def test_classifier_without_models_or_adapter_uses_deterministic_result(tmp_path):
    client = make_client(tmp_path, allow=["nan:rerank", "nan:flux-2-klein"])
    client.config.orchestrator.mode = "model"
    client.config.orchestrator.model = None
    request = RequestEnvelope(task="create an image")

    assert client.operations._classify_operation(request, input=None, files=[], dry_run=False) == "image_generation"

    client.config.orchestrator.model = "openai:planner"
    assert client.operations._classify_operation(request, input=None, files=[], dry_run=False) == "image_generation"


def test_classifier_failure_can_fall_back_to_deterministic_result(tmp_path):
    config = make_config(tmp_path, allow=["nan:rerank", "nan:flux-2-klein"])
    config.orchestrator.mode = "model"
    config.orchestrator.model = "openai:planner"
    client = Crupier(
        config,
        adapters={"nan": OperationAdapter(), "openai": ChatClassifier("not json")},
    )

    result = client.operations._classify_operation(
        RequestEnvelope(task="create an image", metadata={"_crupier_orchestrator_calls": []}),
        input=None,
        files=[],
        dry_run=False,
    )

    assert result == "image_generation"


def test_classifier_invalid_response_records_failure_and_error_mode_raises(tmp_path):
    config = make_config(tmp_path)
    config.orchestrator.mode = "model"
    config.orchestrator.model = "openai:planner"
    config.orchestrator.fallback = "error"
    client = Crupier(
        config,
        adapters={"nan": OperationAdapter(), "openai": ChatClassifier("not json")},
    )
    request = RequestEnvelope(task="rank documents", metadata={"_crupier_orchestrator_calls": []})

    with pytest.raises(CrupierRouteValidationError, match="did not return"):
        client.operations._classify_operation(request, input=None, files=[], dry_run=False)

    call = request.metadata["_crupier_orchestrator_calls"][0]
    assert call["error_type"] == "CrupierRouteValidationError"


def test_available_operations_handles_missing_embedding_and_empty_adapters(tmp_path):
    client = make_client(
        tmp_path,
        allow=["nan:qwen3-embedding", "nan:rerank"],
        adapters={"nan": OperationAdapter()},
    )
    assert client.operations._available_operations() == ["reranker"]

    embedding = make_client(
        tmp_path,
        allow=["nan:qwen3-embedding"],
        adapters={"nan": EmbeddingOperationAdapter()},
    )
    assert embedding.operations._available_operations() == ["embedding"]

    empty = make_client(tmp_path, adapters={})
    with pytest.raises(CrupierPolicyError, match="No executable operations"):
        empty.operations._available_operations()


def test_requested_model_resolution_supports_qualified_unknown_and_ambiguous_ids():
    cards = [
        CapabilityCard(ModelRef.parse("openai:same"), "test", model_kind="embedding"),
        CapabilityCard(ModelRef.parse("nan:same"), "test", model_kind="embedding"),
    ]

    assert OperationRouter._resolve_requested_model("openai:same", cards, "embedding") == "openai:same"
    assert OperationRouter._resolve_requested_model("auto", cards, "embedding") is None
    with pytest.raises(CrupierModelUnsupportedError, match="No allowed"):
        OperationRouter._resolve_requested_model("missing", cards, "embedding")
    with pytest.raises(CrupierModelUnsupportedError, match="ambiguous"):
        OperationRouter._resolve_requested_model("same", cards, "embedding")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("embeddings", "embedding"),
        ("rerank", "reranker"),
        ("stt", "transcription"),
        ("speech", "tts"),
        ("images", "image_generation"),
    ],
)
def test_operation_aliases(value, expected):
    assert normalize_operation(value) == expected


def test_unknown_operation_is_rejected():
    with pytest.raises(CrupierModelUnsupportedError, match="Unsupported operation"):
        normalize_operation("video")


@pytest.mark.parametrize(
    ("operation", "payload", "message"),
    [
        ("embedding", {}, "input is required"),
        ("embedding", {"input": "x", "dimensions": "bad"}, "must be an integer"),
        ("reranker", {"query": "", "documents": ["a"]}, "query must be"),
        ("reranker", {"query": "q", "documents": []}, "documents must be"),
        ("reranker", {"query": "q", "documents": ["a"], "top_n": "bad"}, "must be an integer"),
        ("reranker", {"query": "q", "documents": ["a"], "top_n": 2}, "between 1"),
        ("transcription", {}, "file is required"),
        ("tts", {"input": "", "voice": "v"}, "input must be"),
        ("tts", {"input": "x", "voice": ""}, "voice must be"),
        ("image_generation", {"prompt": ""}, "prompt must be"),
    ],
)
def test_invalid_operation_payloads(operation, payload, message):
    with pytest.raises(CrupierModelUnsupportedError, match=message):
        _validate_operation_payload(operation, payload)


def test_valid_operation_payloads_are_normalized():
    embedding = {"input": "x", "dimensions": "3"}
    rerank = {"query": "q", "documents": ["a", "b"], "top_n": "1"}

    _validate_operation_payload("embedding", embedding)
    _validate_operation_payload("reranker", rerank)

    assert embedding["dimensions"] == 3
    assert rerank["top_n"] == 1


def test_embedding_card_maximum_dimensions_are_enforced():
    model = CapabilityCard(
        ModelRef.parse("openai:embed"),
        "test",
        model_kind="embedding",
        embedding_dimensions=1536,
    )

    assert "at most 1536" in _operation_card_incompatibility("embedding", model, {"dimensions": 3072})
    assert _operation_card_incompatibility("embedding", model, {"dimensions": 1024}) is None


def test_planning_helpers_summarize_paths_collections_and_file_objects(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")
    upload = BytesIO(b"data")
    upload.name = "upload.bin"

    assert _planning_value(path)["exists"] is True
    assert _planning_value(("a", "b")) == ["a", "b"]
    assert _planning_value({"a": b"x"}) == {"a": {"type": "bytes", "size_bytes": 1}}
    assert _planning_value(upload) == {"type": "file_object", "name": "upload.bin"}
    assert _planning_payload({"file": b"abc", "none": None}) == {
        "file": {"type": "bytes", "size_bytes": 3}
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("no object", "did not return"),
        ("{bad}", "invalid JSON"),
        ('{"operation":"video"}', "outside available"),
        ('{"operation":"chat","confidence":2}', "between 0 and 1"),
        ('{"operation":"chat","confidence":"high"}', "between 0 and 1"),
    ],
)
def test_invalid_operation_classification(text, message):
    with pytest.raises(CrupierRouteValidationError, match=message):
        _parse_operation_classification(text, ["chat"])


def test_operation_classification_accepts_fenced_json():
    assert _parse_operation_classification('```json\n{"operation":"chat","confidence":0.8}\n```', ["chat"]) == "chat"


@pytest.mark.parametrize(
    ("task", "input_value", "available", "expected"),
    [
        ("rank documents", None, ["reranker", "chat"], "reranker"),
        ("semantic vector", None, ["embedding", "chat"], "embedding"),
        ("read this aloud", None, ["tts", "chat"], "tts"),
        ("unknown", None, ["embedding"], "embedding"),
    ],
)
def test_deterministic_operation_signals(task, input_value, available, expected):
    assert _deterministic_operation(task, input=input_value, files=[], available=available) == expected


def test_deterministic_operation_detects_structured_rerank_payload():
    assert (
        _deterministic_operation(
            "process",
            input={"query": "q", "documents": ["a"]},
            files=[],
            available=["reranker", "chat"],
        )
        == "reranker"
    )


def test_operation_payload_defaults_and_file_values():
    assert _operation_payload("embedding", task="embed", input=None, files=[], supplied={})["input"] == "embed"
    assert _operation_payload("reranker", task="q", input=None, files=[], supplied={}) == {
        "query": "q",
        "documents": [],
    }
    assert _operation_payload("tts", task="speak", input=None, files=[], supplied={})["input"] == "speak"
    image = _operation_payload(
        "image_generation",
        task="draw",
        input=None,
        files=[Path("image.png")],
        supplied={},
    )
    assert image == {"prompt": "draw", "images": [Path("image.png")]}

    class Asset:
        uri = "file:///tmp/a.wav"

    assert _file_value(Asset()) == "file:///tmp/a.wav"
    assert _file_value(b"raw") == b"raw"


def test_edit_image_forwards_edit_specific_payload(tmp_path):
    client = make_client(tmp_path, allow=["nan:flux-2-klein"])

    result = client.edit_image(
        prompt="remove background",
        images=["source.png"],
        mask="mask.png",
        size="1024x1024",
        dry_run=True,
    )

    assert result.operation == "image_generation"
    assert result.model == "nan:flux-2-klein"


def test_successful_orchestrator_ignores_failed_calls():
    calls = [
        {"model": "openai:one"},
        {"model": "openai:two", "error": "failed"},
    ]
    assert _successful_orchestrator(calls) == "openai:one"
    assert _successful_orchestrator([{"error": "failed"}]) is None
