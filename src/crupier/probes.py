"""Capability probing for real provider adapters."""

from __future__ import annotations

import io
import json
import wave
from dataclasses import asdict, dataclass, field
from datetime import date
from time import perf_counter
from typing import Any, Iterable

from .adapters import ProviderAdapter
from .capabilities import capability_evidence
from .models import CapabilityCard, ModelRef, RequestEnvelope
from .registry import ModelRegistry

DEFAULT_PROBES = (
    "text_basic",
    "json_instruction",
    "max_output_param",
    "structured_output",
    "tool_call",
    "streaming",
)
OPERATION_PROBES = ("reranker", "transcription", "tts", "image_generation")
AVAILABLE_PROBES = DEFAULT_PROBES + ("embeddings",) + OPERATION_PROBES

PROBE_CAPABILITIES = {
    "text_basic": "text_generation",
    "json_instruction": "json_instruction",
    "max_output_param": "max_output_control",
    "structured_output": "structured_output",
    "tool_call": "tool_call",
    "streaming": "streaming",
    "embeddings": "embeddings",
    "reranker": "reranker",
    "transcription": "transcription",
    "tts": "tts",
    "image_generation": "image_generation",
}
CORE_PROBES = ("text_basic", "json_instruction", "max_output_param")


@dataclass(slots=True)
class ProbeResult:
    model: str
    provider: str
    probe: str
    status: str
    ok: bool | None = None
    latency_ms: int | None = None
    error_type: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(slots=True)
class ProbeReport:
    dry_run: bool
    applied: bool
    results: list[ProbeResult] = field(default_factory=list)
    written_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "applied": self.applied,
            "results": [result.to_dict() for result in self.results],
            "written_files": self.written_files,
            "warnings": self.warnings,
            "summary": self.summary(),
        }

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return counts


@dataclass(slots=True)
class ReadinessItem:
    model: str
    provider: str
    status: str
    required_probes: list[dict[str, Any]]
    missing_probes: list[str] = field(default_factory=list)
    failed_probes: list[str] = field(default_factory=list)
    inferred_probes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReadinessReport:
    strict: bool
    items: list[ReadinessItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strict": self.strict,
            "items": [item.to_dict() for item in self.items],
            "summary": self.summary(),
        }

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts


class CapabilityProbeRunner:
    def __init__(self, registry: ModelRegistry, adapters: dict[str, ProviderAdapter]):
        self.registry = registry
        self.adapters = adapters

    def probe(
        self,
        models: Iterable[str],
        *,
        probes: Iterable[str] | None = None,
        apply: bool = False,
        dry_run: bool = False,
    ) -> ProbeReport:
        selected_probes = tuple(probes) if probes is not None else None
        unknown = sorted(set(selected_probes or ()) - set(AVAILABLE_PROBES))
        if unknown:
            raise ValueError(f"Unknown capability probes: {', '.join(unknown)}")

        report = ProbeReport(dry_run=dry_run, applied=apply and not dry_run)
        for model_key in [ModelRef.parse(model).key for model in models]:
            card = self.registry.get(model_key)
            card_results: list[ProbeResult] = []
            card_probes = selected_probes or _applicable_probes(card)
            for probe_name in card_probes:
                if not _probe_is_applicable(card, probe_name):
                    result = ProbeResult(
                        model=model_key,
                        provider=card.model_ref.provider,
                        probe=probe_name,
                        status="skipped",
                        ok=None,
                        error=f"Probe {probe_name!r} does not apply to model kind {card.model_kind!r}.",
                        metadata={"capability": PROBE_CAPABILITIES[probe_name]},
                    )
                    report.results.append(result)
                    card_results.append(result)
                    continue
                if dry_run:
                    result = ProbeResult(
                        model=model_key,
                        provider=card.model_ref.provider,
                        probe=probe_name,
                        status="planned",
                    )
                else:
                    result = self._run_probe(card, probe_name)
                report.results.append(result)
                card_results.append(result)

            if apply and not dry_run:
                updated = self._apply_results(card, [result for result in card_results if result.status != "skipped"])
                written = self.registry.save_card(updated)
                if written:
                    report.written_files.append(written)
        if dry_run:
            report.warnings.append("Dry run only: no provider calls were made and no cards were updated.")
        elif not apply:
            report.warnings.append("Probe results were not saved. Re-run with --apply to update capability cards.")
        return report

    def readiness(self, models: Iterable[str], *, strict: bool = False) -> ReadinessReport:
        report = ReadinessReport(strict=strict)
        for model_key in [ModelRef.parse(model).key for model in models]:
            card = self.registry.get(model_key)
            report.items.append(self._readiness_item(card, strict=strict))
        return report

    def _readiness_item(self, card: CapabilityCard, *, strict: bool) -> ReadinessItem:
        if card.model_kind not in {"chat", "embedding", *OPERATION_PROBES}:
            return ReadinessItem(
                model=card.model_ref.key,
                provider=card.model_ref.provider,
                status="unsupported_executor",
                required_probes=[],
                notes=[f"Crupier has no execution facade for model kind {card.model_kind!r}."],
            )
        embedding_only = _embedding_only(card)
        if embedding_only:
            required = ["embeddings"]
        elif card.model_kind in OPERATION_PROBES:
            required = [card.model_kind]
        else:
            required = list(DEFAULT_PROBES if strict else CORE_PROBES)
        if not strict and not embedding_only:
            if card.supports_structured_output:
                required.append("structured_output")
            if card.supports_tools:
                required.append("tool_call")
            if card.supports_streaming:
                required.append("streaming")
        if card.supports_embeddings and "embeddings" not in required:
            required.append("embeddings")

        required_probes: list[dict[str, Any]] = []
        missing: list[str] = []
        failed: list[str] = []
        inferred: list[str] = []
        notes: list[str] = []

        for probe_name in required:
            capability = PROBE_CAPABILITIES[probe_name]
            declared = _declared_capability(card, capability)
            evidence = capability_evidence(card, capability, declared=declared)
            required_probes.append(
                {
                    "probe": probe_name,
                    "capability": capability,
                    "status": evidence.status,
                    "source": evidence.source,
                    "supported": evidence.supported,
                }
            )
            if evidence.status == "failed":
                failed.append(probe_name)
            elif evidence.status == "verified":
                continue
            elif evidence.status == "inferred":
                inferred.append(probe_name)
            else:
                missing.append(probe_name)

        if failed:
            status = "failed"
        elif missing or inferred:
            status = "needs_probes"
        else:
            status = "ready"

        if strict:
            notes.append("strict mode requires every applicable probe for this model kind to be verified.")
        return ReadinessItem(
            model=card.model_ref.key,
            provider=card.model_ref.provider,
            status=status,
            required_probes=required_probes,
            missing_probes=missing,
            failed_probes=failed,
            inferred_probes=inferred,
            notes=notes,
        )

    def _run_probe(self, card: CapabilityCard, probe_name: str) -> ProbeResult:
        if probe_name == "text_basic":
            return self._probe_text_basic(card)
        if probe_name == "json_instruction":
            return self._probe_json_instruction(card)
        if probe_name == "max_output_param":
            return self._probe_max_output_param(card)
        if probe_name == "structured_output":
            return self._probe_native(
                card,
                probe_name,
                capability="structured_output",
                declared=card.supports_structured_output,
            )
        if probe_name == "tool_call":
            return self._probe_native(card, probe_name, capability="tool_call", declared=card.supports_tools)
        if probe_name == "streaming":
            return self._probe_native(card, probe_name, capability="streaming", declared=card.supports_streaming)
        if probe_name == "embeddings":
            return self._probe_embeddings(card)
        if probe_name in OPERATION_PROBES:
            return self._probe_operation(card, probe_name)
        raise ValueError(f"Unknown capability probe: {probe_name}")

    def _probe_text_basic(self, card: CapabilityCard) -> ProbeResult:
        prompt = 'Capability probe. Reply with exactly: "crupier-probe-ok"'
        request = RequestEnvelope(
            task="Capability probe: text generation",
            constraints={"max_output_tokens": 128, "timeout_seconds": 60, "disable_thinking": True},
        )
        return self._call_and_check(
            card,
            probe="text_basic",
            prompt=prompt,
            request=request,
            check=lambda text: _normalize_probe_text(text).find("crupierprobeok") >= 0,
            metadata={"capability": "text_generation"},
        )

    def _probe_json_instruction(self, card: CapabilityCard) -> ProbeResult:
        prompt = 'Return exactly this JSON object and no prose: {"ok": true, "probe": "crupier"}'
        request = RequestEnvelope(
            task="Capability probe: JSON instruction adherence",
            constraints={"max_output_tokens": 512, "timeout_seconds": 60, "disable_thinking": True},
        )

        def check(text: str) -> bool:
            data = _extract_json_object(text)
            return data.get("ok") is True and data.get("probe") == "crupier"

        return self._call_and_check(
            card,
            probe="json_instruction",
            prompt=prompt,
            request=request,
            check=check,
            metadata={"capability": "json_instruction"},
        )

    def _probe_max_output_param(self, card: CapabilityCard) -> ProbeResult:
        prompt = 'Reply with exactly one word: "ok"'
        request = RequestEnvelope(
            task="Capability probe: max output parameter",
            constraints={"max_output_tokens": 128, "timeout_seconds": 60, "disable_thinking": True},
        )
        return self._call_and_check(
            card,
            probe="max_output_param",
            prompt=prompt,
            request=request,
            check=lambda text: bool(text.strip()),
            metadata={"capability": "max_output_control", "requested_max_output_tokens": 128},
        )

    def _probe_native(
        self,
        card: CapabilityCard,
        probe_name: str,
        *,
        capability: str,
        declared: bool,
    ) -> ProbeResult:
        adapter = self.adapters.get(card.model_ref.provider)
        if adapter is None:
            return ProbeResult(
                model=card.model_ref.key,
                provider=card.model_ref.provider,
                probe=probe_name,
                status="unknown",
                ok=None,
                error="No adapter configured for provider.",
                metadata={"capability": capability},
            )

        native_probe = getattr(adapter, "probe_capability", None)
        if not callable(native_probe):
            return self._probe_declared(card, probe_name, capability=capability, declared=declared)

        request = RequestEnvelope(
            task=f"Capability probe: {capability}",
            constraints={"max_output_tokens": 512, "timeout_seconds": 60},
        )
        started = perf_counter()
        try:
            response = native_probe(model=card.model_ref.model, probe=probe_name, request=request)
            latency_ms = int((perf_counter() - started) * 1000)
            metadata = {"capability": capability} | response.metadata
            status = str(metadata.get("probe_status", "verified"))
            ok = bool(metadata.get("ok", status == "verified"))
            return ProbeResult(
                model=card.model_ref.key,
                provider=card.model_ref.provider,
                probe=probe_name,
                status=status,
                ok=ok,
                latency_ms=latency_ms,
                metadata=metadata | {"usage": response.usage},
            )
        except NotImplementedError:
            return self._probe_declared(card, probe_name, capability=capability, declared=declared)
        except Exception as exc:  # noqa: BLE001 - probes must continue across capabilities
            latency_ms = int((perf_counter() - started) * 1000)
            return ProbeResult(
                model=card.model_ref.key,
                provider=card.model_ref.provider,
                probe=probe_name,
                status="failed",
                ok=False,
                latency_ms=latency_ms,
                error_type=exc.__class__.__name__,
                error=str(exc),
                metadata={"capability": capability, "native_probe": True},
            )

    def _call_and_check(
        self,
        card: CapabilityCard,
        *,
        probe: str,
        prompt: str,
        request: RequestEnvelope,
        check: Any,
        metadata: dict[str, Any],
    ) -> ProbeResult:
        adapter = self.adapters.get(card.model_ref.provider)
        if adapter is None:
            return ProbeResult(
                model=card.model_ref.key,
                provider=card.model_ref.provider,
                probe=probe,
                status="unknown",
                ok=None,
                error="No adapter configured for provider.",
                metadata=metadata,
            )

        started = perf_counter()
        try:
            response = adapter.generate(model=card.model_ref.model, prompt=prompt, request=request)
            latency_ms = int((perf_counter() - started) * 1000)
            ok = bool(check(response.text))
            result_metadata = metadata | {
                "usage": response.usage,
                "provider_metadata": response.metadata,
            }
            return ProbeResult(
                model=card.model_ref.key,
                provider=card.model_ref.provider,
                probe=probe,
                status="verified" if ok else "failed",
                ok=ok,
                latency_ms=latency_ms,
                metadata=result_metadata,
            )
        except Exception as exc:  # noqa: BLE001 - probes must report provider-specific failures
            latency_ms = int((perf_counter() - started) * 1000)
            return ProbeResult(
                model=card.model_ref.key,
                provider=card.model_ref.provider,
                probe=probe,
                status="failed",
                ok=False,
                latency_ms=latency_ms,
                error_type=exc.__class__.__name__,
                error=str(exc),
                metadata=metadata,
            )

    def _probe_embeddings(self, card: CapabilityCard) -> ProbeResult:
        adapter = self.adapters.get(card.model_ref.provider)
        declared = card.supports_embeddings
        if adapter is None:
            return ProbeResult(
                model=card.model_ref.key,
                provider=card.model_ref.provider,
                probe="embeddings",
                status="unknown",
                ok=None,
                error="No adapter configured for provider.",
                metadata={"capability": "embeddings"},
            )
        embed = getattr(adapter, "embed", None)
        if not callable(embed):
            return self._probe_declared(card, "embeddings", capability="embeddings", declared=declared)

        started = perf_counter()
        try:
            response = embed(model=card.model_ref.model, input=["crupier embedding probe"])
            latency_ms = int((perf_counter() - started) * 1000)
            ok = bool(response.embeddings and response.embeddings[0])
            dimensions = len(response.embeddings[0]) if ok else None
            return ProbeResult(
                model=card.model_ref.key,
                provider=card.model_ref.provider,
                probe="embeddings",
                status="verified" if ok else "failed",
                ok=ok,
                latency_ms=latency_ms,
                metadata={
                    "capability": "embeddings",
                    "usage": response.usage,
                    "provider_metadata": response.metadata,
                    "embedding_dimensions": dimensions,
                },
            )
        except Exception as exc:  # noqa: BLE001 - probes must report provider-specific failures
            latency_ms = int((perf_counter() - started) * 1000)
            return ProbeResult(
                model=card.model_ref.key,
                provider=card.model_ref.provider,
                probe="embeddings",
                status="failed",
                ok=False,
                latency_ms=latency_ms,
                error_type=exc.__class__.__name__,
                error=str(exc),
                metadata={"capability": "embeddings"},
            )

    def _probe_operation(self, card: CapabilityCard, operation: str) -> ProbeResult:
        adapter = self.adapters.get(card.model_ref.provider)
        if adapter is None:
            return ProbeResult(
                model=card.model_ref.key,
                provider=card.model_ref.provider,
                probe=operation,
                status="unknown",
                ok=None,
                error="No adapter configured for provider.",
                metadata={"capability": operation},
            )
        execute = getattr(adapter, "execute_operation", None)
        supports = getattr(adapter, "supports_operation", None)
        declared = card.model_kind == operation
        if not callable(execute) or not callable(supports) or not supports(
            operation=operation,
            model=card.model_ref.model,
        ):
            return self._probe_declared(card, operation, capability=operation, declared=declared)
        request = RequestEnvelope(
            task=f"Capability probe: {operation}",
            constraints={"timeout_seconds": 60, "max_output_tokens": 128},
        )
        started = perf_counter()
        try:
            response = execute(
                operation=operation,
                model=card.model_ref.model,
                request=request,
                payload=_operation_probe_payload(operation),
            )
            latency_ms = int((perf_counter() - started) * 1000)
            ok = _operation_probe_ok(operation, response.output)
            return ProbeResult(
                model=card.model_ref.key,
                provider=card.model_ref.provider,
                probe=operation,
                status="verified" if ok else "failed",
                ok=ok,
                latency_ms=latency_ms,
                metadata={
                    "capability": operation,
                    "usage": response.usage,
                    "provider_metadata": response.metadata,
                },
            )
        except Exception as exc:  # noqa: BLE001 - probes report provider failures without aborting the batch
            latency_ms = int((perf_counter() - started) * 1000)
            return ProbeResult(
                model=card.model_ref.key,
                provider=card.model_ref.provider,
                probe=operation,
                status="failed",
                ok=False,
                latency_ms=latency_ms,
                error_type=exc.__class__.__name__,
                error=str(exc),
                metadata={"capability": operation},
            )

    @staticmethod
    def _probe_declared(card: CapabilityCard, probe_name: str, *, capability: str, declared: bool) -> ProbeResult:
        return ProbeResult(
            model=card.model_ref.key,
            provider=card.model_ref.provider,
            probe=probe_name,
            status="inferred" if declared else "unknown",
            ok=declared,
            metadata={
                "capability": capability,
                "declared": declared,
                "note": "No native provider probe is registered for this capability; readiness is inferred from the capability card.",
            },
        )

    def _apply_results(self, card: CapabilityCard, results: list[ProbeResult]) -> CapabilityCard:
        today = date.today().isoformat()
        card.last_updated = today
        for result in results:
            capability = str(result.metadata.get("capability", result.probe))
            stored = result.to_dict()
            stored.pop("model", None)
            stored.pop("provider", None)
            card.probe_results[result.probe] = stored
            card.capability_status[capability] = {
                "status": result.status,
                "source": f"probe:{result.probe}",
                "checked_at": today,
            }
            if result.latency_ms is not None:
                card.capability_status[capability]["latency_ms"] = result.latency_ms

            if result.probe == "text_basic":
                card.local_eval_scores["probe_text_basic"] = 1.0 if result.status == "verified" else 0.0
            if result.probe == "json_instruction":
                card.local_eval_scores["probe_json_instruction"] = 1.0 if result.status == "verified" else 0.0
            if result.probe == "max_output_param":
                card.local_eval_scores["probe_max_output_param"] = 1.0 if result.status == "verified" else 0.0
            if result.probe == "structured_output" and result.status == "verified":
                card.supports_structured_output = True
            if result.probe == "tool_call" and result.status == "verified":
                card.supports_tools = True
            if result.probe == "streaming" and result.status == "verified":
                card.supports_streaming = True
            if result.probe == "embeddings" and result.status == "verified":
                card.supports_embeddings = True
                card.embedding_input_modalities = ["text"]
                card.modalities_output = ["embedding"]
                card.model_kind = "embedding"
                dimensions = result.metadata.get("embedding_dimensions")
                if isinstance(dimensions, int):
                    card.embedding_dimensions = dimensions
        return card


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}


def _normalize_probe_text(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum())


def _declared_capability(card: CapabilityCard, capability: str) -> bool:
    if capability == "structured_output":
        return card.supports_structured_output
    if capability == "tool_call":
        return card.supports_tools
    if capability == "streaming":
        return card.supports_streaming
    if capability == "embeddings":
        return card.supports_embeddings
    if capability in OPERATION_PROBES:
        return card.model_kind == capability
    return False


def _embedding_only(card: CapabilityCard) -> bool:
    return card.model_kind == "embedding" or (
        card.supports_embeddings and card.modalities_output == ["embedding"] and not card.supports_tools
    )


def _applicable_probes(card: CapabilityCard) -> tuple[str, ...]:
    if _embedding_only(card):
        return ("embeddings",)
    if card.model_kind == "chat":
        probes = list(DEFAULT_PROBES)
        if card.supports_embeddings:
            probes.append("embeddings")
        return tuple(probes)
    if card.model_kind in OPERATION_PROBES:
        return (card.model_kind,)
    return ()


def _probe_is_applicable(card: CapabilityCard, probe_name: str) -> bool:
    if probe_name == "embeddings":
        return card.supports_embeddings or _embedding_only(card)
    if probe_name in OPERATION_PROBES:
        return card.model_kind == probe_name
    return card.model_kind == "chat"


def _operation_probe_payload(operation: str) -> dict[str, Any]:
    if operation == "reranker":
        return {
            "query": "capital of France",
            "documents": ["Berlin is in Germany.", "Paris is the capital of France."],
            "top_n": 1,
        }
    if operation == "transcription":
        return {
            "file": _silent_wav(),
            "filename": "crupier-probe.wav",
            "response_format": "json",
        }
    if operation == "tts":
        return {
            "input": "Crupier probe.",
            "voice": "ef_dora",
            "response_format": "mp3",
        }
    if operation == "image_generation":
        return {
            "prompt": "A centered red circle on a white background",
            "size": "256x256",
            "n": 1,
            "response_format": "b64_json",
            "seed": 1,
        }
    return {}


def _operation_probe_ok(operation: str, output: Any) -> bool:
    if operation == "reranker":
        return isinstance(output, list) and bool(output) and isinstance(output[0], dict)
    if operation == "transcription":
        return isinstance(output, dict) and "text" in output
    if operation == "tts":
        return isinstance(output, bytes | bytearray) and bool(output)
    if operation == "image_generation":
        return isinstance(output, list) and bool(output) and isinstance(output[0], dict)
    return False


def _silent_wav() -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 3_200)
    return target.getvalue()
