"""Validate specialized operations and compatibility surfaces with real providers.

Offline preview:

    python examples/live_operations_validation.py

Real validation from an operation-capable project:

    python examples/live_operations_validation.py --real --project . --write-report
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import struct
import tempfile
import threading
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from _example_support import offline_client, print_route
from crupier import Crupier, ModelRef, OperationResult
from crupier.compat.openai import OpenAI
from crupier.server import build_openai_compatible_server


CASE_NAMES = (
    "classifier",
    "embeddings",
    "rerank",
    "audio",
    "image",
    "compat",
    "http",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="Execute configured provider calls")
    parser.add_argument("--project", default=".", help="Directory containing crupier.toml")
    parser.add_argument("--case", action="append", choices=CASE_NAMES, help="Run only selected cases")
    parser.add_argument(
        "--write-report",
        nargs="?",
        const=".crupier/evals/live-operations-validation.json",
        help="Write the sanitized JSON report",
    )
    args = parser.parse_args()
    if not args.real:
        _offline_preview()
        return

    client = Crupier.from_project(args.project)
    selected = args.case or list(CASE_NAMES)
    cases = [_run_case(name, client) for name in selected]
    report = {
        "schema_version": 1,
        "project": client.config.project.name,
        "real_provider_calls": True,
        "summary": {
            "passed": sum(case["status"] == "pass" for case in cases),
            "failed": sum(case["status"] == "fail" for case in cases),
            "total": len(cases),
        },
        "cases": cases,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.write_report:
        path = Path(args.project) / args.write_report
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"report={path}")
    if report["summary"]["failed"]:
        raise SystemExit(1)


def _offline_preview() -> None:
    client = offline_client(
        project="live-operations-validation",
        profile="agentic",
        allow=[
            "openai:gpt-5.5",
            "anthropic:claude-opus-4-8",
            "google:gemini-3.5-flash",
        ],
    )
    result = client.deal(
        task="Choose a route for validating an AI operations boundary.",
        mode="agentic",
        dry_run=True,
    )
    print_route("live_operations_validation", result, extra={"validation": "offline-preview"})


def _run_case(name: str, client: Crupier) -> dict[str, Any]:
    runners: dict[str, Callable[[Crupier], dict[str, Any]]] = {
        "classifier": _classifier_case,
        "embeddings": _embeddings_case,
        "rerank": _rerank_case,
        "audio": _audio_case,
        "image": _image_case,
        "compat": _compat_case,
        "http": _http_case,
    }
    try:
        return runners[name](client)
    except Exception as exc:  # noqa: BLE001 - the validation report records every case
        return {
            "id": name,
            "status": "fail",
            "error_type": exc.__class__.__name__,
            "error": str(exc)[:1000],
        }


def _classifier_case(client: Crupier) -> dict[str, Any]:
    expected_model = _require_model(client, "reranker")
    result = client.run(
        "Rank these documents by relevance to the query.",
        input={
            "query": "the exact token ZEBRA-991",
            "documents": ["no match", "contains ZEBRA-991 exactly", "unrelated"],
            "top_n": 3,
        },
        constraints={"max_calls": 4, "max_cost_usd": 0.25, "max_latency_ms": 120000},
        trace="debug",
    )
    if not isinstance(result, OperationResult):
        raise TypeError("Operation classifier returned a chat result for a reranking request.")
    roles = _roles(result)
    classified = [
        call.get("metadata", {}).get("classified_operation")
        for call in _calls(result)
        if call.get("role") == "operation_classifier"
    ]
    checks = _operation_checks(result) | {
        "classified_as_reranker": result.operation == "reranker" and classified == ["reranker"],
        "expected_model_selected": result.model == expected_model,
        "classifier_and_executor_called": roles == ["operation_classifier", "primary"],
        "top_document_correct": _top_document_is_expected(result.data),
    }
    return _operation_observation("classifier", result, checks, {"classified": classified})


def _embeddings_case(client: Crupier) -> dict[str, Any]:
    models = _models_for_kind(client, "embedding")
    if not models:
        raise RuntimeError("No executable embedding model is present in the project allowlist.")
    observations = []
    checks: dict[str, bool] = {}
    for model in models:
        provider = ModelRef.parse(model).provider
        requested_dimensions = 128 if provider in {"openai", "google"} else None
        result = client.embed(
            input=["Crupier live embedding", "capability-aware model router"],
            model=model,
            dimensions=requested_dimensions,
            constraints={"max_calls": 2, "max_cost_usd": 0.25, "max_latency_ms": 120000},
            trace="debug",
        )
        vectors = result.data if isinstance(result.data, list) else []
        dimensions = [len(vector) for vector in vectors if isinstance(vector, list)]
        valid = (
            len(vectors) == 2
            and len(dimensions) == 2
            and dimensions[0] > 0
            and len(set(dimensions)) == 1
            and (requested_dimensions is None or dimensions == [requested_dimensions, requested_dimensions])
            and all(any(abs(float(value)) > 0 for value in vector) for vector in vectors)
            and all(_operation_checks(result).values())
        )
        checks[f"{model}_valid"] = valid
        observations.append(
            {
                "model": model,
                "vectors": len(vectors),
                "dimensions": dimensions,
                "requested_dimensions": requested_dimensions,
                "trace": _trace_summary(result),
            }
        )
    return {
        "id": "embeddings",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "models": observations,
    }


def _rerank_case(client: Crupier) -> dict[str, Any]:
    model = _require_model(client, "reranker")
    result = client.rerank(
        query="the exact token ZEBRA-991",
        documents=["no matching token", "contains the exact token ZEBRA-991", "unrelated"],
        top_n=3,
        model=model,
        constraints={"max_calls": 2, "max_cost_usd": 0.25, "max_latency_ms": 120000},
        trace="debug",
    )
    scores = [float(item.get("relevance_score", 0)) for item in result.data or []]
    checks = _operation_checks(result) | {
        "expected_model_selected": result.model == model,
        "three_results_returned": len(result.data or []) == 3,
        "scores_descending": all(scores[index] >= scores[index + 1] for index in range(len(scores) - 1)),
        "top_document_correct": _top_document_is_expected(result.data),
    }
    return _operation_observation("rerank", result, checks, {"scores": scores})


def _audio_case(client: Crupier) -> dict[str, Any]:
    tts_model = _require_model(client, "tts")
    transcription_model = _require_model(client, "transcription")
    phrase = "The secret phrase is blue ocean seven."
    speech = client.synthesize(
        input=phrase,
        voice="ef_dora",
        model=tts_model,
        response_format="wav",
        constraints={"max_calls": 2, "max_cost_usd": 0.50, "max_latency_ms": 120000},
        trace="debug",
    )
    if not isinstance(speech.data, bytes):
        raise TypeError("TTS operation did not return bytes.")
    transcription = client.transcribe(
        file=speech.data,
        filename="crupier-validation.wav",
        language="en",
        model=transcription_model,
        constraints={"max_calls": 2, "max_cost_usd": 0.50, "max_latency_ms": 120000},
        trace="debug",
    )
    text = (
        str(transcription.data.get("text", ""))
        if isinstance(transcription.data, dict)
        else str(transcription.data or "")
    )
    checks = {
        **{f"tts_{key}": value for key, value in _operation_checks(speech).items()},
        **{f"transcription_{key}": value for key, value in _operation_checks(transcription).items()},
        "wav_returned": len(speech.data) > 1000 and speech.data[:4] == b"RIFF",
        "phrase_preserved": "blue ocean" in text.lower() and ("seven" in text.lower() or "7" in text),
    }
    return {
        "id": "audio",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "tts": {
            "model": speech.model,
            "bytes": len(speech.data),
            "format_signature": speech.data[:4].hex(),
            "trace": _trace_summary(speech),
        },
        "transcription": {
            "model": transcription.model,
            "text": text[:200],
            "trace": _trace_summary(transcription),
        },
    }


def _image_case(client: Crupier) -> dict[str, Any]:
    model = _require_model(client, "image_generation")
    generated = client.generate_image(
        prompt="A centered green circle on a white background, no text.",
        model=model,
        n=1,
        size="256x256",
        response_format="b64_json",
        constraints={"max_calls": 2, "max_cost_usd": 1.0, "max_latency_ms": 180000},
        trace="debug",
    )
    generated_bytes, generated_reference = _image_result_evidence(generated.data)
    with tempfile.NamedTemporaryFile(suffix=".png") as fixture:
        fixture.write(_solid_png((255, 0, 0), width=256, height=256))
        fixture.flush()
        edited = client.edit_image(
            prompt="Change the red square to blue and keep the background white. No text.",
            images=fixture.name,
            model=model,
            n=1,
            size="256x256",
            response_format="b64_json",
            constraints={"max_calls": 2, "max_cost_usd": 1.0, "max_latency_ms": 180000},
            trace="debug",
        )
    edited_bytes, edited_reference = _image_result_evidence(edited.data)
    checks = {
        **{f"generation_{key}": value for key, value in _operation_checks(generated).items()},
        **{f"edit_{key}": value for key, value in _operation_checks(edited).items()},
        "generation_returned_image": generated_reference and generated_bytes != 0,
        "edit_returned_image": edited_reference and edited_bytes != 0,
        "edit_endpoint_recorded": str(edited.provider_metadata.get("api", "")).endswith("/images/edits"),
    }
    return {
        "id": "image",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "generation": {
            "model": generated.model,
            "count": len(generated.data or []),
            "decoded_bytes": generated_bytes,
            "reference_type": generated_reference,
            "trace": _trace_summary(generated),
        },
        "edit": {
            "model": edited.model,
            "count": len(edited.data or []),
            "decoded_bytes": edited_bytes,
            "reference_type": edited_reference,
            "trace": _trace_summary(edited),
        },
    }


def _compat_case(client: Crupier) -> dict[str, Any]:
    chat_model = _require_model(client, "chat")
    embedding_model = _require_model(client, "embedding")
    compat = OpenAI(crupier=client, dry_run=False, compat_mode="balanced")
    chat = compat.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Reply with exactly COMPAT-OK and nothing else."}],
        max_tokens=40,
        trace="debug",
    )
    chat_text = _exact_text(chat.choices[0].message.content)
    stream = compat.responses.create(
        model=chat_model,
        instructions="Reply with exactly STREAM-OK and nothing else.",
        input="Run the compatibility streaming validation.",
        max_output_tokens=40,
        stream=True,
        trace="debug",
    )
    events = list(stream)
    deltas = "".join(
        str(event.delta)
        for event in events
        if event.type == "response.output_text.delta"
    )
    native_events = list(
        client.stream(
            task="Reply with exactly NATIVE-STREAM-OK and nothing else.",
            mode="fast",
            constraints={"max_output_tokens": 40, "max_cost_usd": 0.25, "max_latency_ms": 120000},
            trace="debug",
            dry_run=False,
        )
    )
    native_text = _exact_text(native_events[-1].result.output_text)
    provider = ModelRef.parse(embedding_model).provider
    dimensions = 64 if provider in {"openai", "google"} else None
    embedding = compat.embeddings.create(
        model=embedding_model,
        input=["compatibility embedding"],
        dimensions=dimensions,
        trace="debug",
    )
    event_types = [event.type for event in events]
    native_types = [event.type for event in native_events]
    checks = {
        "chat_completion_shape": chat.object == "chat.completion" and chat_text == "COMPAT-OK",
        "responses_event_contract": event_types[0] == "response.created"
        and event_types[-1] == "response.completed"
        and _exact_text(deltas) == "STREAM-OK",
        "native_stream_contract": native_types == ["route_started", "route_selected", "final"]
        and native_text == "NATIVE-STREAM-OK",
        "embedding_shape": embedding.object == "list"
        and len(embedding.data) == 1
        and len(embedding.data[0].embedding) > 0,
    }
    return {
        "id": "compat",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "chat": {"model": chat.model, "text": chat_text, "strategy": chat.crupier.route["strategy"]},
        "responses_stream": {"event_types": event_types, "text": _exact_text(deltas)},
        "native_stream": {"event_types": native_types, "text": native_text},
        "embedding": {
            "model": embedding.model,
            "vectors": len(embedding.data),
            "dimensions": len(embedding.data[0].embedding),
        },
    }


def _http_case(client: Crupier) -> dict[str, Any]:
    models = {
        "chat": _require_model(client, "chat"),
        "embedding": _require_model(client, "embedding"),
        "reranker": _require_model(client, "reranker"),
        "tts": _require_model(client, "tts"),
        "transcription": _require_model(client, "transcription"),
        "image_generation": _require_model(client, "image_generation"),
    }
    server = build_openai_compatible_server(
        crupier=client,
        host="127.0.0.1",
        port=0,
        dry_run=False,
        max_request_bytes=2_000_000,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    observations: dict[str, Any] = {}
    try:
        status, headers, body = _http_request(base_url, "/health")
        health = json.loads(body)
        observations["health"] = {
            "status": status,
            "ok": health.get("ok"),
            "request_id": bool(headers.get("x-request-id")),
        }

        status, _, body = _http_request(base_url, "/v1/models")
        listed = json.loads(body)
        observations["models"] = {"status": status, "count": len(listed.get("data", []))}

        status, headers, body = _http_request(
            base_url,
            "/v1/responses",
            payload={
                "model": models["chat"],
                "instructions": "Reply with exactly HTTP-RESPONSES-OK and nothing else.",
                "input": "Validate the endpoint.",
                "max_output_tokens": 40,
                "trace": "debug",
            },
        )
        response = json.loads(body)
        observations["responses"] = {
            "status": status,
            "text": _exact_text(response.get("output_text", "")),
            "request_id": bool(headers.get("x-request-id")),
        }

        status, headers, body = _http_request(
            base_url,
            "/v1/chat/completions",
            payload={
                "model": models["chat"],
                "messages": [
                    {"role": "user", "content": "Reply with exactly HTTP-STREAM-OK and nothing else."}
                ],
                "max_tokens": 40,
                "stream": True,
                "stream_options": {"include_usage": True},
                "trace": "debug",
            },
        )
        stream_text, done = _chat_sse_text(body)
        observations["chat_stream"] = {
            "status": status,
            "text": _exact_text(stream_text),
            "done": done,
            "content_type": headers.get("content-type"),
        }

        embedding_provider = ModelRef.parse(models["embedding"]).provider
        requested_dimensions = 32 if embedding_provider in {"openai", "google"} else None
        embedding_payload: dict[str, Any] = {
            "model": models["embedding"],
            "input": ["HTTP embedding"],
            "trace": "debug",
        }
        if requested_dimensions is not None:
            embedding_payload["dimensions"] = requested_dimensions
        status, _, body = _http_request(base_url, "/v1/embeddings", payload=embedding_payload)
        embedding = json.loads(body)
        vectors = embedding.get("data", [])
        observations["embeddings"] = {
            "status": status,
            "vectors": len(vectors),
            "dimensions": len(vectors[0]["embedding"]) if vectors else 0,
            "requested_dimensions": requested_dimensions,
        }

        status, _, body = _http_request(
            base_url,
            "/v1/rerank",
            payload={
                "model": models["reranker"],
                "query": "exact token ZEBRA-991",
                "documents": ["no match", "contains ZEBRA-991 exactly", "unrelated"],
                "top_n": 3,
                "trace": "debug",
            },
        )
        rerank = json.loads(body)
        observations["rerank"] = {
            "status": status,
            "top_index": rerank.get("results", [{}])[0].get("index"),
            "top_document": rerank.get("results", [{}])[0].get("document"),
        }

        status, headers, speech = _http_request(
            base_url,
            "/v1/audio/speech",
            payload={
                "model": models["tts"],
                "input": "The secret phrase is blue ocean seven.",
                "voice": "ef_dora",
                "response_format": "wav",
                "trace": "debug",
            },
        )
        observations["speech"] = {
            "status": status,
            "bytes": len(speech),
            "signature": speech[:4].hex(),
            "content_type": headers.get("content-type"),
        }

        multipart, content_type = _multipart_body(
            {
                "model": models["transcription"],
                "language": "en",
                "response_format": "json",
                "trace": "debug",
            },
            [("file", "validation.wav", "audio/wav", speech)],
        )
        status, _, body = _http_request(
            base_url,
            "/v1/audio/transcriptions",
            data=multipart,
            content_type=content_type,
        )
        transcription = json.loads(body)
        observations["transcription"] = {
            "status": status,
            "text": str(transcription.get("text", ""))[:200],
        }

        status, _, body = _http_request(
            base_url,
            "/v1/images/generations",
            payload={
                "model": models["image_generation"],
                "prompt": "A centered green circle on a white background, no text.",
                "n": 1,
                "size": "256x256",
                "response_format": "b64_json",
                "trace": "debug",
            },
        )
        generated = json.loads(body)
        generated_bytes, generated_reference = _image_result_evidence(generated.get("data"))
        observations["image_generation"] = {
            "status": status,
            "bytes": generated_bytes,
            "reference_type": generated_reference,
        }

        multipart, content_type = _multipart_body(
            {
                "model": models["image_generation"],
                "prompt": "Change the red square to blue and keep the background white. No text.",
                "n": "1",
                "size": "256x256",
                "response_format": "b64_json",
                "trace": "debug",
            },
            [("image", "red.png", "image/png", _solid_png((255, 0, 0), width=256, height=256))],
        )
        status, _, body = _http_request(
            base_url,
            "/v1/images/edits",
            data=multipart,
            content_type=content_type,
        )
        edited = json.loads(body)
        edited_bytes, edited_reference = _image_result_evidence(edited.get("data"))
        observations["image_edit"] = {
            "status": status,
            "bytes": edited_bytes,
            "reference_type": edited_reference,
        }

        status, headers, body = _http_request(
            base_url,
            "/v1/chat/completions",
            payload={"model": models["chat"]},
        )
        error = json.loads(body).get("error", {})
        observations["typed_error"] = {
            "status": status,
            "type": error.get("type"),
            "code": error.get("code"),
            "request_id": bool(headers.get("x-request-id")),
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    checks = {
        "health": observations["health"] == {"status": 200, "ok": True, "request_id": True},
        "models": observations["models"]["status"] == 200
        and observations["models"]["count"] == len(client.config.models.allow),
        "responses": observations["responses"]["status"] == 200
        and observations["responses"]["text"] == "HTTP-RESPONSES-OK"
        and observations["responses"]["request_id"],
        "chat_stream": observations["chat_stream"]["status"] == 200
        and observations["chat_stream"]["text"] == "HTTP-STREAM-OK"
        and observations["chat_stream"]["done"],
        "embeddings": observations["embeddings"]["status"] == 200
        and observations["embeddings"]["vectors"] == 1
        and observations["embeddings"]["dimensions"] > 0
        and (
            observations["embeddings"]["requested_dimensions"] is None
            or observations["embeddings"]["dimensions"] == observations["embeddings"]["requested_dimensions"]
        ),
        "rerank": observations["rerank"]["status"] == 200
        and observations["rerank"]["top_index"] == 1,
        "speech": observations["speech"]["status"] == 200
        and observations["speech"]["bytes"] > 1000
        and observations["speech"]["signature"] == b"RIFF".hex(),
        "transcription": observations["transcription"]["status"] == 200
        and "blue ocean" in observations["transcription"]["text"].lower(),
        "image_generation": observations["image_generation"]["status"] == 200
        and observations["image_generation"]["bytes"] != 0,
        "image_edit": observations["image_edit"]["status"] == 200
        and observations["image_edit"]["bytes"] != 0,
        "typed_error": observations["typed_error"] == {
            "status": 400,
            "type": "invalid_request_error",
            "code": "invalid_request",
            "request_id": True,
        },
    }
    return {
        "id": "http",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "models": models,
        "endpoints": observations,
    }


def _models_for_kind(client: Crupier, kind: str) -> list[str]:
    cards = {card.model_ref.key: card for card in client.registry.allowed_cards()}
    models: list[str] = []
    for raw in client.config.models.allow:
        key = ModelRef.parse(raw).key
        card = cards.get(key)
        if card is None or card.model_kind != kind:
            continue
        adapter = client.adapters.get(card.model_ref.provider)
        if adapter is None:
            continue
        if kind == "chat" and callable(getattr(adapter, "generate", None)):
            models.append(key)
        elif kind == "embedding" and callable(getattr(adapter, "embed", None)):
            models.append(key)
        else:
            supports = getattr(adapter, "supports_operation", None)
            if callable(supports) and supports(operation=kind, model=card.model_ref.model):
                models.append(key)
    return models


def _require_model(client: Crupier, kind: str) -> str:
    models = _models_for_kind(client, kind)
    if not models:
        raise RuntimeError(f"No executable {kind!r} model is present in the project allowlist.")
    return models[0]


def _operation_checks(result: OperationResult) -> dict[str, bool]:
    trace = result.trace
    return {
        "single_model_route": bool(result.route and result.route.strategy == "single" and len(result.route.models) == 1),
        "primary_called": _roles(result)[-1:] == ["primary"],
        "no_trace_errors": bool(trace is not None and not trace.errors),
        "within_budget": bool(result.provider_metadata.get("budget", {}).get("calls_started", 0) >= 1),
    }


def _operation_observation(
    case_id: str,
    result: OperationResult,
    checks: dict[str, bool],
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": case_id,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "operation": result.operation,
        "model": result.model,
        "details": details,
        "trace": _trace_summary(result),
    }


def _trace_summary(result: OperationResult) -> dict[str, Any]:
    trace = result.trace
    return {
        "calls": [_sanitize_call(call) for call in _calls(result)],
        "errors": trace.errors if trace else [],
        "latency_ms": result.latency_ms,
        "cost": result.cost.to_dict(),
        "warnings": result.warnings,
    }


def _calls(result: OperationResult) -> list[dict[str, Any]]:
    return list(result.trace.provider_calls) if result.trace else []


def _roles(result: OperationResult) -> list[str]:
    return [str(call.get("role")) for call in _calls(result)]


def _sanitize_call(call: dict[str, Any]) -> dict[str, Any]:
    metadata = call.get("metadata") if isinstance(call.get("metadata"), dict) else {}
    return {
        key: value
        for key, value in {
            "role": call.get("role"),
            "provider": call.get("provider"),
            "model": call.get("model"),
            "operation": call.get("operation"),
            "latency_ms": call.get("latency_ms"),
            "classified_operation": metadata.get("classified_operation"),
            "api": metadata.get("api"),
        }.items()
        if value is not None
    }


def _top_document_is_expected(data: Any) -> bool:
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return False
    top = data[0]
    if int(top.get("index", -1)) != 1:
        return False
    document = top.get("document")
    if document is None:
        return True
    if isinstance(document, dict):
        document = document.get("text", "")
    return "ZEBRA-991" in str(document)


def _image_result_evidence(data: Any) -> tuple[int, str]:
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return 0, "missing"
    item = data[0]
    encoded = item.get("b64_json")
    if isinstance(encoded, str) and encoded:
        try:
            return len(base64.b64decode(encoded, validate=True)), "b64_json"
        except (ValueError, binascii.Error):
            return 0, "invalid_b64_json"
    url = item.get("url")
    if isinstance(url, str) and url.startswith(("https://", "http://")):
        return -1, "url"
    return 0, "missing"


def _exact_text(value: Any) -> str:
    return str(value or "").strip().strip(".\n ")


def _http_request(
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    data: bytes | None = None,
    content_type: str = "application/json",
) -> tuple[int, dict[str, str], bytes]:
    body = data
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url + path,
        data=body,
        method="POST" if body is not None else "GET",
    )
    if body is not None:
        request.add_header("content-type", content_type)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()


def _multipart_body(
    fields: dict[str, Any],
    files: list[tuple[str, str, str, bytes]],
) -> tuple[bytes, str]:
    boundary = "----crupier-" + uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode(),
                b"\r\n",
            ]
        )
    for name, filename, mime_type, content in files:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode(),
                f"Content-Type: {mime_type}\r\n\r\n".encode(),
                content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _chat_sse_text(body: bytes) -> tuple[str, bool]:
    chunks: list[str] = []
    done = False
    for line in body.decode("utf-8").splitlines():
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        if raw == "[DONE]":
            done = True
            continue
        event = json.loads(raw)
        choices = event.get("choices", [])
        if choices:
            chunks.append(str(choices[0].get("delta", {}).get("content", "") or ""))
    return "".join(chunks), done


def _solid_png(
    rgb: tuple[int, int, int],
    *,
    width: int,
    height: int,
) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = binascii.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


if __name__ == "__main__":
    main()
