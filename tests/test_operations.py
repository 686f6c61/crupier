from crupier import Crupier
from crupier.adapters import AdapterResponse, EmbeddingResponse, OperationResponse
from crupier.config import CrupierConfig
from crupier.errors import CrupierModelUnsupportedError, CrupierPolicyError, CrupierRouteValidationError


class FakeOperationAdapter:
    provider = "nan"

    def __init__(self):
        self.calls = []

    @staticmethod
    def supports_operation(*, operation, model):
        expected = {
            "reranker": "rerank",
            "transcription": "whisper",
            "tts": "kokoro",
            "image_generation": "flux-2-klein",
        }
        return expected.get(operation) == model

    def execute_operation(self, *, operation, model, request, payload):
        self.calls.append(
            {
                "operation": operation,
                "model": model,
                "timeout": request.constraints.get("timeout_seconds"),
                "payload": payload,
            }
        )
        outputs = {
            "reranker": [{"index": 1, "relevance_score": 0.95}],
            "transcription": {"text": "hola"},
            "tts": b"audio",
            "image_generation": [{"url": "https://example.test/image.png"}],
        }
        return OperationResponse(
            operation=operation,
            output=outputs[operation],
            usage={"input_tokens": 10} if operation == "reranker" else {},
            metadata={"provider": "nan", "model": model},
        )

    def embed(self, *, model, input, dimensions=None):
        self.calls.append({"operation": "embedding", "model": model, "input": input, "dimensions": dimensions})
        return EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            usage={"input_tokens": 2},
            metadata={"provider": "nan", "model": model},
        )


class NoOperationAdapter(FakeOperationAdapter):
    @staticmethod
    def supports_operation(*, operation, model):
        return False


class FakeClassifierAdapter:
    provider = "ollama"

    def __init__(self, operation):
        self.operation = operation
        self.prompts = []

    def generate(self, *, model, prompt, request):
        self.prompts.append(prompt)
        return AdapterResponse(
            text=f'{{"operation":"{self.operation}","confidence":0.99,"reason":"request intent"}}',
            usage={"prompt_eval_count": 20, "eval_count": 10},
            metadata={"provider": "ollama", "model": model},
        )


class FakeAdjustableEmbeddingAdapter:
    provider = "openai"

    def __init__(self):
        self.calls = []

    def embed(self, *, model, input, dimensions=None):
        self.calls.append({"model": model, "input": input, "dimensions": dimensions})
        return EmbeddingResponse(
            embeddings=[[0.1] * int(dimensions or 3)],
            metadata={"provider": "openai", "model": model},
        )


class FakeChatAndOperationAdapter(FakeOperationAdapter):
    def generate(self, *, model, prompt, request):
        self.calls.append({"operation": "chat", "model": model, "prompt": prompt})
        return AdapterResponse(
            text="chat answer",
            usage={"input_tokens": 4, "output_tokens": 2},
            metadata={"provider": "nan", "model": model},
        )


def make_client(tmp_path, *, adapter=None, allow=None):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "operations"},
            "providers": {"nan": {"enabled": True, "env_key": "NAN_API_KEY"}},
            "models": {
                "allow": allow
                or [
                    "nan:qwen3.6",
                    "nan:qwen3-embedding",
                    "nan:rerank",
                    "nan:kokoro",
                    "nan:whisper",
                    "nan:flux-2-klein",
                ]
            },
            "routing": {
                "require_operational_providers": False,
                "max_calls": 4,
                "max_latency_ms": 5000,
            },
        }
    )
    config.root = tmp_path
    return Crupier(config, adapters={"nan": adapter or FakeOperationAdapter()})


def make_model_classifier_client(tmp_path, *, classified_operation, allow, nan_adapter=None):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "operation-classifier"},
            "providers": {
                "ollama": {"enabled": True, "host": "https://ollama.com/api"},
                "nan": {"enabled": True, "env_key": "NAN_API_KEY"},
            },
            "models": {"allow": allow},
            "routing": {
                "require_operational_providers": False,
                "max_calls": 4,
                "max_latency_ms": 5000,
            },
            "orchestrator": {
                "mode": "model",
                "model": "ollama:glm-5.2",
                "fallback": "deterministic",
            },
        }
    )
    config.root = tmp_path
    classifier = FakeClassifierAdapter(classified_operation)
    client = Crupier(
        config,
        adapters={
            "ollama": classifier,
            "nan": nan_adapter or FakeOperationAdapter(),
        },
    )
    return client, classifier


def test_rerank_routes_only_to_reranker_and_records_trace(tmp_path):
    adapter = FakeOperationAdapter()
    client = make_client(tmp_path, adapter=adapter)

    result = client.rerank(
        query="capital of France",
        documents=["Berlin", "Paris"],
        top_n=1,
        constraints={"max_calls": 1},
        trace="debug",
    )

    assert result.operation == "reranker"
    assert result.model == "nan:rerank"
    assert result.data == [{"index": 1, "relevance_score": 0.95}]
    assert result.route.models == ["nan:rerank"]
    assert result.trace.candidate_models == ["nan:rerank"]
    assert result.trace.provider_calls[-1]["operation"] == "reranker"
    assert result.provider_metadata["budget"]["calls_started"] == 1
    assert adapter.calls[0]["payload"]["top_n"] == 1
    assert "pricing" in result.warnings[0].lower()


def test_operation_router_supports_embedding_and_binary_results(tmp_path):
    client = make_client(tmp_path)

    embedding = client.embed(input="hola", model="qwen3-embedding", dimensions=4096)
    speech = client.synthesize(input="Hola", voice="ef_dora")

    assert embedding.model == "nan:qwen3-embedding"
    assert embedding.data == [[0.1, 0.2, 0.3]]
    assert speech.model == "nan:kokoro"
    assert speech.data == b"audio"
    assert speech.to_dict()["data"] == {"bytes": 5}


def test_embedding_dimensions_filter_fixed_models_before_selection(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "embedding-dimensions"},
            "providers": {
                "nan": {"enabled": True, "env_key": "NAN_API_KEY"},
                "openai": {"enabled": True, "env_key": "OPENAI_API_KEY"},
            },
            "models": {"allow": ["nan:qwen3-embedding", "openai:text-embedding-3-small"]},
            "routing": {"require_operational_providers": False},
        }
    )
    config.root = tmp_path
    fixed = FakeOperationAdapter()
    adjustable = FakeAdjustableEmbeddingAdapter()
    client = Crupier(config, adapters={"nan": fixed, "openai": adjustable})

    result = client.embed(input="hola", dimensions=1536, trace="debug")

    assert result.model == "openai:text-embedding-3-small"
    assert len(result.data[0]) == 1536
    assert adjustable.calls[0]["dimensions"] == 1536
    assert not any(call.get("operation") == "embedding" for call in fixed.calls)
    assert any(
        exclusion["model"] == "nan:qwen3-embedding" and "fixed at 4096" in exclusion["reason"]
        for exclusion in result.trace.excluded_models
    )


def test_invalid_specialized_payload_fails_before_provider_call(tmp_path):
    adapter = FakeOperationAdapter()
    client = make_client(tmp_path, adapter=adapter)

    try:
        client.embed(input="hello", dimensions=0)
    except CrupierModelUnsupportedError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("invalid dimensions must fail before routing")

    assert adapter.calls == []


def test_operation_router_dry_run_does_not_call_adapter(tmp_path):
    adapter = FakeOperationAdapter()
    client = make_client(tmp_path, adapter=adapter)

    result = client.generate_image(prompt="A lighthouse", dry_run=True)

    assert result.model == "nan:flux-2-klein"
    assert result.data is None
    assert result.provider_metadata["dry_run"] is True
    assert adapter.calls == []


def test_operation_router_enforces_zero_call_route_before_provider(tmp_path):
    adapter = FakeOperationAdapter()
    client = make_client(tmp_path, adapter=adapter)

    try:
        client.transcribe(file=b"audio", constraints={"max_calls": 0})
    except CrupierRouteValidationError as exc:
        assert "max_calls=0" in str(exc)
    else:
        raise AssertionError("specialized operations must share the request call budget")
    assert adapter.calls == []


def test_operation_router_rejects_adapter_without_real_operation_support(tmp_path):
    client = make_client(
        tmp_path,
        adapter=NoOperationAdapter(),
        allow=["nan:qwen3.6", "nan:flux-2-klein"],
    )

    try:
        client.generate_image(prompt="A lighthouse")
    except CrupierPolicyError as exc:
        assert "operation support" in str(exc)
        assert "flux-2-klein" in str(exc)
    else:
        raise AssertionError("capability cards cannot override missing adapter transport")


def test_operation_trace_persistence_omits_binary_content(tmp_path):
    client = make_client(tmp_path)

    result = client.synthesize(
        input="Hola",
        voice="ef_dora",
        constraints={"store_trace": True, "store_response": True},
        trace=False,
    )

    stored = client.traces.read(result.provider_metadata["stored_trace_path"].split("/")[-1][:-5])
    assert result.trace is None
    assert stored["replayable"] is False
    assert stored["result"]["operation"] == "tts"
    assert stored["result"]["data"] == {"binary_content_stored": False, "bytes": 5}


def test_run_uses_model_to_classify_and_execute_specialized_operation(tmp_path):
    client, classifier = make_model_classifier_client(
        tmp_path,
        classified_operation="image_generation",
        allow=["nan:qwen3.6", "nan:flux-2-klein", "nan:rerank"],
    )

    result = client.run(
        "Crea una imagen de un faro al atardecer",
        operation_payload={"size": "1024x1024", "response_format": "url"},
        constraints={"max_calls": 2},
        trace="debug",
    )

    assert result.operation == "image_generation"
    assert result.model == "nan:flux-2-klein"
    assert result.provider_metadata["budget"]["calls_started"] == 2
    assert [call["role"] for call in result.trace.provider_calls] == [
        "operation_classifier",
        "primary",
    ]
    assert "available_operations" in classifier.prompts[0]
    assert "image_generation" in classifier.prompts[0]


def test_run_carries_classifier_budget_and_trace_into_chat_route(tmp_path):
    nan_adapter = FakeChatAndOperationAdapter()
    client, _ = make_model_classifier_client(
        tmp_path,
        classified_operation="chat",
        allow=["nan:qwen3.6", "nan:flux-2-klein"],
        nan_adapter=nan_adapter,
    )

    result = client.run(
        "Explica por que 17 es primo",
        constraints={"max_calls": 2},
        trace="debug",
    )

    assert result.output_text == "chat answer"
    assert [call["role"] for call in result.trace.provider_calls] == [
        "operation_classifier",
        "primary",
    ]
    assert result.trace.final_quality_signals["execution_budget"]["calls_started"] == 2


def test_operation_classifier_redacts_secrets_and_respects_summary_only(tmp_path):
    client, classifier = make_model_classifier_client(
        tmp_path,
        classified_operation="image_generation",
        allow=["nan:qwen3.6", "nan:flux-2-klein"],
    )

    client.run(
        "Crea una imagen con OPENAI_API_KEY=sk-secretvalue123456",
        constraints={"max_calls": 2},
    )

    assert "sk-secretvalue" not in classifier.prompts[0]
    assert "OPENAI_API_KEY=[redacted]" in classifier.prompts[0]

    classifier.prompts.clear()
    client.config.orchestrator.allow_prompt_summary_only = True
    result = client.run(
        "Crea una imagen de un faro",
        constraints={"max_calls": 1},
    )

    assert result.operation == "image_generation"
    assert result.provider_metadata["budget"]["calls_started"] == 1
    assert classifier.prompts == []


def test_operation_classifier_summarizes_upload_tuples_without_binary_content(tmp_path):
    client, classifier = make_model_classifier_client(
        tmp_path,
        classified_operation="transcription",
        allow=["nan:whisper"],
    )

    result = client.run(
        "Transcribe este audio",
        files=[("private.wav", b"secret-audio-content", "audio/wav")],
        constraints={"max_calls": 2},
    )

    assert result.operation == "transcription"
    assert '"name": "private.wav"' in classifier.prompts[0]
    assert '"size_bytes": 20' in classifier.prompts[0]
    assert "secret-audio-content" not in classifier.prompts[0]
