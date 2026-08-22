import json
from datetime import UTC, datetime, timedelta

import pytest

from crupier.capabilities import capability_evidence, capability_reason
from crupier.errors import CrupierError
from crupier.feedback import (
    HumanFeedbackStore,
    _load_review_report,
    _optional_float,
    _optional_int,
    _review_source_type,
    _validate_rating,
    _validate_verdict,
    import_human_decisions,
)
from crupier.model_profiles import (
    _apply_capability_status,
    _apply_family_inference,
    _apply_provider_metadata,
    _model_decision,
    _profile_confidence,
    _tradeoffs,
    classify_task_signals,
)
from crupier.models import CapabilityCard, ModelRef, RequestEnvelope
from crupier.probes import (
    CapabilityProbeRunner,
    ProbeReport,
    _applicable_probes,
    _extract_json_object,
    _operation_probe_ok,
    _operation_probe_payload,
)
from crupier.retention import is_expired, prune_jsonl
from crupier.trace_store import TraceStore, _jsonable, _safe_metadata_value


def _card(model: str, **kwargs):
    return CapabilityCard(ModelRef.parse(model), "test", **kwargs)


def test_retention_handles_timestamp_fallbacks_and_mixed_jsonl(tmp_path):
    now = datetime(2026, 8, 22, tzinfo=UTC)
    fallback = tmp_path / "fallback.json"
    fallback.write_text("{}", encoding="utf-8")
    old_timestamp = (now - timedelta(days=10)).timestamp()
    fallback.touch()
    fallback.chmod(0o600)
    import os

    os.utime(fallback, (old_timestamp, old_timestamp))

    assert is_expired(None, None, fallback_path=fallback, now=now) is False
    assert is_expired(None, 7, fallback_path=fallback, now=now) is True
    assert is_expired(None, 7, fallback_path=tmp_path / "missing", now=now) is False
    assert is_expired("not-a-date", 7, now=now) is False
    assert is_expired("2026-08-01T00:00:00", 7, now=now) is True

    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n"
        "not-json\n"
        + json.dumps({"created_at": "2026-08-01T00:00:00+00:00", "id": "old"})
        + "\n"
        + json.dumps({"created_at": "2999-01-01T00:00:00+00:00", "id": "future"})
        + "\n",
        encoding="utf-8",
    )

    assert prune_jsonl(path, 7) == 1
    assert path.read_text(encoding="utf-8") == "\nnot-json\n" + json.dumps(
        {"created_at": "2999-01-01T00:00:00+00:00", "id": "future"}
    ) + "\n"
    assert prune_jsonl(path, None) == 0
    assert prune_jsonl(tmp_path / "absent.jsonl", 7) == 0
    assert prune_jsonl(tmp_path, 7) == 0


@pytest.mark.parametrize(
    ("model", "kind", "inputs", "outputs"),
    [
        ("openai:text-embedding-new", "embedding", ["text"], ["embedding"]),
        ("openai:gpt-image-next", "image", ["text", "image"], ["image"]),
        ("openai:gpt-audio-next", "audio", ["text", "audio"], ["text", "audio"]),
        ("openai:gpt-realtime-next", "realtime", ["text", "audio"], ["text", "audio"]),
        ("openai:whisper-next", "transcription", ["audio"], ["text"]),
        ("openai:tts-next", "tts", ["text"], ["audio"]),
        ("openai:sora-next", "video", ["text", "image"], ["video"]),
        ("google:imagen-next", "image", ["text", "image"], ["text", "image"]),
        ("google:veo-next", "video", ["text", "image"], ["video"]),
        ("google:lyria-next", "music", ["text"], ["audio"]),
    ],
)
def test_family_inference_classifies_specialized_models(model, kind, inputs, outputs):
    card = _card(model)

    _apply_family_inference(card)

    assert card.model_kind == kind
    assert card.modalities_input == inputs
    assert card.modalities_output == outputs
    assert card.strengths


def test_provider_metadata_and_failed_probe_evidence_override_declarations():
    anthropic = _card("anthropic:claude-future")
    _apply_provider_metadata(
        anthropic,
        {
            "max_input_tokens": 200000,
            "max_tokens": 8192,
            "capabilities": {
                "image_input": {"supported": True},
                "pdf_input": {"supported": True},
                "structured_outputs": {"supported": True},
                "code_execution": {"supported": True},
                "thinking": {"supported": True},
            },
            "input_per_million_usd": 3,
        },
    )
    assert anthropic.context_window == 200000
    assert anthropic.max_output_tokens == 8192
    assert anthropic.supports_file_input is True
    assert anthropic.supports_structured_output is True
    assert anthropic.supports_code_execution is True
    assert anthropic.pricing["source"] == "provider_metadata"

    card = _card(
        "openai:text-embedding-new",
        supports_tools=True,
        supports_structured_output=True,
        supports_embeddings=True,
        strengths=["tool_use", "structured_output", "embeddings", "rag", "semantic_search"],
        capability_status={
            "tool_call": {"status": "failed"},
            "structured_output": {"status": "failed"},
            "streaming": {"status": "failed"},
            "embeddings": {"status": "failed"},
        },
    )
    _apply_capability_status(card)
    assert not card.supports_tools
    assert not card.supports_structured_output
    assert not card.supports_streaming
    assert not card.supports_embeddings
    assert card.strengths == []


def test_model_decisions_cover_explicit_opt_in_and_specialized_families():
    expected = {
        "openai:gpt-codex-future": "opt_in",
        "openai:deep-research-future": "specialized",
        "openai:search-api-future": "specialized",
        "openai:gpt-4-legacy": "legacy",
        "google:gemma-future": "specialized",
        "custom:model-2025": "opt_in",
        "custom:model": "unknown",
    }
    assert {model: _model_decision(_card(model))["routing_status"] for model in expected} == expected

    preview = CapabilityCard(ModelRef("custom", "model", stability="preview"), "test")
    latest = CapabilityCard(ModelRef("custom", "model", stability="latest"), "test")
    assert _model_decision(preview)["lifecycle"] == "preview"
    assert _model_decision(latest)["lifecycle"] == "latest_alias"


def test_signal_wrapper_tradeoffs_and_confidence_cover_public_profile_edges():
    request = RequestEnvelope(task="", mode="agentic", tools=[{"name": "search"}])
    assert classify_task_signals(request) >= {"agentic", "tool_use"}

    card = _card(
        "ollama:qwen3-coder",
        cost_tier="low",
        latency_tier="fast",
        quality_tier="strong",
        probe_results={"text_basic": {"status": "verified"}},
    )
    assert _tradeoffs(card) == [
        "cost tier is low",
        "latency tier is fast",
        "quality tier is strong",
        "Ollama Cloud availability and quotas depend on the configured account",
    ]
    assert _profile_confidence(card) == "high"


def test_capability_evidence_distinguishes_verified_failed_inferred_and_unknown():
    card = _card(
        "openai:test",
        capability_status={
            "verified": {"status": "verified", "source": "probe"},
            "failed": {"status": "failed"},
            "inferred": {"status": "inferred", "declared": True},
            "odd": {"status": "unrecognized"},
        },
    )
    assert capability_evidence(card, "verified").supported is True
    assert capability_evidence(card, "failed").supported is False
    assert capability_evidence(card, "inferred").supported is True
    assert capability_evidence(card, "odd").supported is False
    declared = capability_evidence(card, "missing", declared=True)
    unknown = capability_evidence(card, "missing")
    assert declared.status == "inferred"
    assert unknown.status == "unknown"
    assert capability_reason(declared) == "missing support is inferred via capability_card"


class _Registry:
    def __init__(self, card):
        self.card = card

    def get(self, model):
        return self.card


class _FailingAdapter:
    def generate(self, **kwargs):
        raise RuntimeError("generation failed")

    def embed(self, **kwargs):
        raise RuntimeError("embedding failed")

    def supports_operation(self, **kwargs):
        return True

    def execute_operation(self, **kwargs):
        raise RuntimeError("operation failed")

    def probe_capability(self, **kwargs):
        raise NotImplementedError


def test_probe_failures_are_reported_without_aborting_other_capabilities():
    card = _card("openai:test", supports_embeddings=True)
    runner = CapabilityProbeRunner(_Registry(card), {"openai": _FailingAdapter()})

    assert runner._run_probe(card, "text_basic").error_type == "RuntimeError"
    assert runner._run_probe(card, "embeddings").error_type == "RuntimeError"
    operation = _card("openai:test", model_kind="tts")
    assert runner._run_probe(operation, "tts").error_type == "RuntimeError"
    assert runner._run_probe(card, "tool_call").status == "unknown"
    with pytest.raises(ValueError, match="Unknown capability probe"):
        runner._run_probe(card, "not-real")


def test_probe_payload_and_result_validators_reject_malformed_provider_outputs():
    assert _extract_json_object("prefix {not-json} suffix") == {}
    assert _extract_json_object("no object") == {}
    assert _operation_probe_payload("unknown") == {}
    assert _operation_probe_ok("reranker", []) is False
    assert _operation_probe_ok("transcription", {"other": "value"}) is False
    assert _operation_probe_ok("tts", b"") is False
    assert _operation_probe_ok("image_generation", ["wrong"]) is False
    assert _operation_probe_ok("unknown", {}) is False


def test_probe_runner_reports_missing_adapters_and_unsupported_model_kinds():
    chat = _card("openai:test", supports_embeddings=True)
    runner = CapabilityProbeRunner(_Registry(chat), {})

    assert runner._run_probe(chat, "text_basic").status == "unknown"
    assert runner._run_probe(chat, "structured_output").status == "unknown"
    assert runner._run_probe(chat, "embeddings").status == "unknown"
    operation = _card("openai:test", model_kind="tts")
    assert runner._run_probe(operation, "tts").status == "unknown"
    assert ProbeReport(dry_run=True, applied=False).to_dict()["summary"] == {}
    with pytest.raises(ValueError, match="Unknown capability probes"):
        runner.probe(["openai:test"], probes=["not-real"])

    image = _card("openai:image", model_kind="image")
    unsupported = CapabilityProbeRunner(_Registry(image), {}).readiness(["openai:image"])
    assert unsupported.items[0].status == "unsupported_executor"
    assert _applicable_probes(chat)[-1] == "embeddings"
    assert _applicable_probes(image) == ()


def test_trace_store_reports_every_corrupt_artifact_shape(tmp_path):
    root = tmp_path / "traces"
    root.mkdir()
    (root / "io.json").mkdir()
    (root / "encoding.json").write_bytes(b"\xff")
    (root / "scalar.json").write_text("[]", encoding="utf-8")
    (root / "shape.json").write_text(json.dumps({"result": [], "request": {}}), encoding="utf-8")
    (root / "route.json").write_text(
        json.dumps({"result": {"route": "wrong"}, "request": {}}), encoding="utf-8"
    )

    result = TraceStore(root).list()

    assert result.complete is False
    assert {item.error_type for item in result.diagnostics} == {
        "io_error",
        "invalid_encoding",
        "invalid_schema",
    }
    assert len(result.diagnostics) == 5


def test_trace_replay_rejects_non_replayable_and_tool_bearing_records(tmp_path):
    root = tmp_path / "traces"
    root.mkdir()
    store = TraceStore(root)
    (root / "private.json").write_text(json.dumps({"replayable": False}), encoding="utf-8")
    (root / "tools.json").write_text(
        json.dumps({"replayable": True, "request": {"has_tools": True}}), encoding="utf-8"
    )

    with pytest.raises(CrupierError, match="not replayable"):
        store.replay("private", object())
    with pytest.raises(CrupierError, match="tool-bearing"):
        store.replay("tools", object())
    with pytest.raises(CrupierError, match="was not found"):
        store.read("missing")


def test_trace_serialization_falls_back_safely_for_unusual_values():
    class Broken:
        def to_dict(self):
            raise RuntimeError("cannot serialize")

        def __repr__(self):
            return "<broken>"

    assert _jsonable(Broken()) == "<broken>"
    assert _jsonable({1, 2}) in {"{1, 2}", "{2, 1}"}
    assert _safe_metadata_value(("secret prose", 3), store_response=False) == ["[content omitted]", 3]


def test_feedback_list_reports_io_encoding_and_schema_failures(tmp_path):
    io_store = HumanFeedbackStore(tmp_path / "io")
    io_store.root.mkdir()
    io_store.path.mkdir()
    assert io_store.list().diagnostics[0].error_type == "io_error"

    encoding_store = HumanFeedbackStore(tmp_path / "encoding")
    encoding_store.root.mkdir()
    encoding_store.path.write_bytes(b"\xff")
    assert encoding_store.list().diagnostics[0].error_type == "invalid_encoding"

    schema_store = HumanFeedbackStore(tmp_path / "schema")
    schema_store.root.mkdir()
    schema_store.path.write_text("\n[]\n{}\nnot-json\n", encoding="utf-8")
    result = schema_store.list()
    assert result.complete is False
    assert [item.error_type for item in result.diagnostics] == [
        "invalid_schema",
        "invalid_schema",
        "invalid_json",
    ]


def test_feedback_record_and_registry_application_reject_missing_inputs(tmp_path):
    store = HumanFeedbackStore(tmp_path)
    with pytest.raises(CrupierError, match="at least one"):
        store.record(project="test", rating=3)
    assert store._derive_from_trace(None, object()) == {}

    store.record(project="test", rating=4, models=["openai:a"], mode="agentic")
    assert store.summary(model="openai:b")["count"] == 0
    assert store.summary(mode="other")["count"] == 0

    below_minimum = store.apply_to_registry(object(), min_count=2)
    assert below_minimum["skipped"][0]["reason"] == "count below min_count=2"

    class MissingRegistry:
        def get(self, model):
            raise KeyError(model)

    missing = store.apply_to_registry(MissingRegistry())
    assert "openai:a" in missing["skipped"][0]["reason"]


def test_human_decision_import_validates_files_items_models_and_ratings(tmp_path):
    store = HumanFeedbackStore(tmp_path / "feedback")
    path = tmp_path / "decisions.json"

    with pytest.raises(CrupierError, match="not found"):
        import_human_decisions(store, project="test", decision_path=path)

    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(CrupierError, match="not valid JSON"):
        import_human_decisions(store, project="test", decision_path=path)

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(CrupierError, match="decisions list"):
        import_human_decisions(store, project="test", decision_path=path)

    path.write_text(json.dumps({"decisions": ["bad", {"id": "skip", "record": False}]}), encoding="utf-8")
    skipped = import_human_decisions(store, project="test", decision_path=path)
    assert skipped.imported == 0
    assert len(skipped.skipped) == 2

    path.write_text(json.dumps({"decisions": [{"id": "missing", "record": True, "rating": 3}]}), encoding="utf-8")
    with pytest.raises(CrupierError, match="has no models"):
        import_human_decisions(store, project="test", decision_path=path)

    path.write_text(
        json.dumps({"decisions": [{"id": "rating", "record": True, "models": ["openai:a"]}]}),
        encoding="utf-8",
    )
    with pytest.raises(CrupierError, match="rating must be an integer"):
        import_human_decisions(store, project="test", decision_path=path)

    path.write_text(
        json.dumps(
            {
                "source_dry_run": True,
                "decisions": [
                    {
                        "id": "accepted",
                        "record": True,
                        "models": ["openai:a", "openai:a"],
                        "rating": 5,
                        "verdict": "accept",
                        "tags": ["human review"],
                        "note": "safe note",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    preview = import_human_decisions(store, project="test", decision_path=path, dry_run=True)
    assert preview.imported == 1
    assert preview.records[0].feedback_id == "dry_run:accepted"
    assert preview.records[0].models == ["openai:a"]
    assert set(preview.records[0].tags) == {"human_review", "decision_import", "dry_run_source"}


def test_review_and_numeric_validators_reject_ambiguous_artifacts(tmp_path):
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("not-json", encoding="utf-8")
    with pytest.raises(CrupierError, match="not valid JSON"):
        _load_review_report(invalid_json)
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(CrupierError, match="JSON object"):
        _load_review_report(scalar)
    with pytest.raises(CrupierError, match="eval compare"):
        _review_source_type({})
    assert _optional_float(object()) is None
    assert _optional_int(object()) is None
    with pytest.raises(CrupierError, match="integer from 1 to 5"):
        _validate_rating("bad")
    with pytest.raises(CrupierError, match="integer from 1 to 5"):
        _validate_rating(6)
    with pytest.raises(CrupierError, match="must be one of"):
        _validate_verdict("maybe")
