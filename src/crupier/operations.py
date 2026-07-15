"""Capability-aware routing for non-chat model operations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .budgets import ExecutionBudget, request_with_timeout
from .adapters import OperationResponse
from .costs import actual_cost_from_calls, usage_estimated_cost_from_calls
from .errors import (
    CrupierBudgetExceededError,
    CrupierExecutionLimitError,
    CrupierModelUnsupportedError,
    CrupierPolicyError,
    CrupierRouteValidationError,
)
from .models import CostEstimate, CrupierResult, DecisionTrace, ModelRef, OperationResult, RequestEnvelope
from .orchestrator import DeterministicOrchestrator, ModelOrchestrator
from .policy import Exclusion
from .prompts import build_operation_classification_prompt
from .registry import ModelRegistry
from .runtime_policy import apply_runtime_policy

if TYPE_CHECKING:
    from .client import Crupier
    from .models import CapabilityCard, RoutePlan


SPECIALIZED_OPERATIONS = {"embedding", "reranker", "transcription", "tts", "image_generation"}


class OperationRouter:
    """Route and execute typed operations through policy-approved adapters."""

    def __init__(self, client: Crupier):
        self.client = client

    def execute(
        self,
        operation: str,
        *,
        task: str,
        payload: dict[str, Any],
        model: str | None = None,
        constraints: dict[str, Any] | None = None,
        dry_run: bool = False,
        trace: bool | str = "summary",
        _budget: ExecutionBudget | None = None,
        _started_at: float | None = None,
        _planning_calls: list[dict[str, Any]] | None = None,
    ) -> OperationResult:
        operation = normalize_operation(operation)
        _validate_operation_payload(operation, payload)
        started = _started_at if _started_at is not None else perf_counter()
        constraints = dict(constraints or {})
        constraints["model_kind"] = operation
        cards = self.client.registry.allowed_cards()
        forced_model = self._resolve_requested_model(model, cards, operation)
        if forced_model:
            constraints["force_model"] = forced_model
        request = RequestEnvelope(
            task=task,
            input=_planning_payload(payload),
            mode=operation,
            strategy="single",
            constraints=constraints,
            metadata={
                "operation": operation,
                "_crupier_orchestrator_calls": list(_planning_calls or []),
            },
        )
        if dry_run:
            request.metadata["_crupier_offline_planning"] = True
        budget = _budget
        if not dry_run:
            if not isinstance(budget, ExecutionBudget):
                budget = ExecutionBudget(
                    self.client.config,
                    request,
                    self.client.registry.list(allowed_only=False),
                    started_at=started,
                )
            request.metadata["_crupier_execution_budget"] = budget

        policy_result = self.client.policy.filter_candidates(request, cards)
        cards = policy_result.allowed
        exclusions: list[Exclusion] = []
        filters: list[str] = []
        cards, adapter_exclusions, adapter_filters = self.client._filter_adapter_candidates(cards)
        exclusions.extend(adapter_exclusions)
        filters.extend(item for item in adapter_filters if item not in filters)
        cards, operation_exclusions = self._filter_operation_candidates(operation, cards, payload=payload)
        exclusions.extend(operation_exclusions)
        if operation_exclusions:
            filters.append("adapter_operation_support")
        cards, provider_exclusions, provider_filters = self.client._filter_operational_candidates(
            request,
            cards,
            dry_run=dry_run,
        )
        exclusions.extend(provider_exclusions)
        filters.extend(item for item in provider_filters if item not in filters)
        policy_result.allowed = cards
        policy_result.excluded.extend(exclusions)
        for name in filters:
            if name not in policy_result.filters_applied:
                policy_result.filters_applied.append(name)

        plan = self._plan(request, policy_result.allowed, policy_result.filters_applied, dry_run=dry_run)
        plan.estimated_cost = CostEstimate(estimated_usd=0.0)
        self.client.policy.validate_route(plan, policy_result, request)
        if plan.strategy != "single" or len(plan.models) != 1:
            raise CrupierRouteValidationError(
                f"Specialized operation {operation!r} requires one selected model, got {plan.strategy!r}."
            )

        planning_calls = request.metadata.pop("_crupier_orchestrator_calls", [])
        if not isinstance(planning_calls, list):
            planning_calls = []
        trace_obj = DecisionTrace(
            trace_id=f"trc_{uuid4().hex[:16]}",
            request_summary=self.client._summarize_task(task),
            candidate_models=[card.model_ref.key for card in policy_result.allowed],
            excluded_models=policy_result.excluded_dicts(),
            policy_filters=policy_result.filters_applied,
            orchestrator_model=_successful_orchestrator(planning_calls),
            route_plan=plan,
            provider_calls=list(planning_calls),
            errors=[
                {
                    "phase": "operation_classifier",
                    "provider": item.get("provider"),
                    "model": item.get("model"),
                    "error_type": item.get("error_type"),
                    "error": item.get("error"),
                }
                for item in planning_calls
                if item.get("error")
            ],
            storage_decision=self.client._storage_decision(constraints),
        )
        selected = plan.models[0]
        warnings = _operation_warnings(operation)
        if dry_run:
            latency_ms = int((perf_counter() - started) * 1000)
            trace_obj.latency_ms = latency_ms
            result = OperationResult(
                operation=operation,
                model=selected,
                route=plan,
                trace=trace_obj,
                latency_ms=latency_ms,
                provider_metadata={"dry_run": True, "calls": planning_calls},
                warnings=warnings,
            )
            return self._finalize_result(result, request=request, dry_run=True, trace=trace)

        model_ref = ModelRef.parse(selected)
        adapter = self.client.adapters[model_ref.provider]
        assert budget is not None
        reservation = budget.reserve_call()
        effective_request = request_with_timeout(request, reservation.timeout_seconds)
        call_started = perf_counter()
        try:
            if operation == "embedding":
                embed = getattr(adapter, "embed", None)
                if not callable(embed):
                    raise CrupierModelUnsupportedError(
                        f"Provider {model_ref.provider!r} has no embedding execution method."
                    )
                embedding_response = embed(
                    model=model_ref.model,
                    input=payload.get("input"),
                    dimensions=payload.get("dimensions"),
                )
                response = OperationResponse(
                    operation="embedding",
                    output=embedding_response.embeddings,
                    raw=embedding_response.raw,
                    usage=embedding_response.usage,
                    metadata=embedding_response.metadata,
                )
            else:
                execute_operation = getattr(adapter, "execute_operation", None)
                if not callable(execute_operation):
                    raise CrupierModelUnsupportedError(
                        f"Provider {model_ref.provider!r} has no {operation!r} execution method."
                    )
                response = execute_operation(
                    operation=operation,
                    model=model_ref.model,
                    request=effective_request,
                    payload=payload,
                )
        except Exception as exc:
            trace_obj.provider_calls.append(
                {
                    "role": "primary",
                    "provider": model_ref.provider,
                    "model": selected,
                    "operation": operation,
                    "latency_ms": int((perf_counter() - call_started) * 1000),
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
            )
            trace_obj.errors.append(
                {
                    "phase": "operation",
                    "provider": model_ref.provider,
                    "model": selected,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
            )
            raise
        budget.ensure_deadline()
        call = {
            "role": "primary",
            "provider": model_ref.provider,
            "model": selected,
            "operation": operation,
            "latency_ms": int((perf_counter() - call_started) * 1000),
            "usage": response.usage,
            "metadata": response.metadata,
        }
        trace_obj.provider_calls.append(call)
        all_cards = self.client.registry.list(allowed_only=False)
        estimated = usage_estimated_cost_from_calls(trace_obj.provider_calls, all_cards)
        if estimated is None:
            reserved = budget.snapshot()["estimated_cost_reserved_usd"]
            estimated = float(reserved) if isinstance(reserved, int | float) else 0.0
        cost = CostEstimate(
            estimated_usd=estimated,
            actual_usd=actual_cost_from_calls(trace_obj.provider_calls, all_cards),
        )
        latency_ms = int((perf_counter() - started) * 1000)
        trace_obj.cost = cost
        trace_obj.latency_ms = latency_ms
        trace_obj.final_quality_signals.update(
            {
                "operation": operation,
                "budget": budget.snapshot(),
                "orchestrator_calls": len(planning_calls),
            }
        )
        result = OperationResult(
            operation=operation,
            model=selected,
            data=response.output,
            raw=response.raw,
            route=plan,
            trace=trace_obj,
            cost=cost,
            latency_ms=latency_ms,
            usage=response.usage,
            provider_metadata={
                **response.metadata,
                "calls": trace_obj.provider_calls,
                "budget": budget.snapshot(),
            },
            warnings=warnings,
        )
        return self._finalize_result(result, request=request, dry_run=False, trace=trace)

    def run(
        self,
        task: str,
        input: Any = None,
        *,
        operation: str = "auto",
        model: str | None = None,
        files: list[Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
        response_schema: Any = None,
        mode: str | None = None,
        strategy: str | None = None,
        constraints: dict[str, Any] | None = None,
        operation_payload: dict[str, Any] | None = None,
        dry_run: bool = False,
        trace: bool | str = "summary",
    ) -> CrupierResult | OperationResult:
        started = perf_counter()
        constraints = dict(constraints or {})
        calls: list[dict[str, Any]] = []
        classifier_request = RequestEnvelope(
            task=task,
            input=_planning_value(input),
            mode="operation_classifier",
            constraints=constraints,
            metadata={"_crupier_orchestrator_calls": calls},
        )
        budget = None
        if not dry_run:
            budget = ExecutionBudget(
                self.client.config,
                classifier_request,
                self.client.registry.list(allowed_only=False),
                started_at=started,
            )
            classifier_request.metadata["_crupier_execution_budget"] = budget
        selected_operation = operation
        if operation == "auto":
            selected_operation = self._classify_operation(
                classifier_request,
                input=input,
                files=files or [],
                dry_run=dry_run,
            )
        elif operation == "chat":
            selected_operation = "chat"
        else:
            selected_operation = normalize_operation(operation)

        if selected_operation == "chat":
            metadata: dict[str, Any] = {
                "_crupier_started_at": started,
                "_crupier_orchestrator_calls": calls,
            }
            if budget is not None:
                metadata["_crupier_execution_budget"] = budget
            return self.client.deal(
                task,
                input,
                mode=mode,
                strategy=strategy,
                constraints=constraints,
                files=files,
                messages=messages,
                tools=tools,
                response_schema=response_schema,
                metadata=metadata,
                trace=trace,
                dry_run=dry_run,
            )

        payload = _operation_payload(
            selected_operation,
            task=task,
            input=input,
            files=files or [],
            supplied=operation_payload or {},
        )
        return self.execute(
            selected_operation,
            task=task,
            payload=payload,
            model=model,
            constraints=constraints,
            dry_run=dry_run,
            trace=trace,
            _budget=budget,
            _started_at=started,
            _planning_calls=calls,
        )

    def _classify_operation(
        self,
        request: RequestEnvelope,
        *,
        input: Any,
        files: list[Any],
        dry_run: bool,
    ) -> str:
        available = self._available_operations()
        deterministic = _deterministic_operation(request.task, input=input, files=files, available=available)
        if (
            dry_run
            or self.client.config.orchestrator.allow_prompt_summary_only
            or self.client.config.orchestrator.mode not in {"model", "hybrid"}
        ):
            return deterministic
        orchestrator_models = [
            value
            for value in (
                self.client.config.orchestrator.model,
                self.client.config.orchestrator.fallback_model,
            )
            if value
        ]
        if not orchestrator_models:
            return deterministic
        prompt = build_operation_classification_prompt(
            {
                "task": _planning_value(request.task),
                "input": _planning_value(input),
                "files": [_planning_value(item) for item in files],
                "available_operations": available,
            }
        )
        last_error = ""
        for model_key in dict.fromkeys(orchestrator_models):
            model_ref = ModelRef.parse(str(model_key))
            adapter = self.client.adapters.get(model_ref.provider)
            if adapter is None:
                last_error = f"no adapter for {model_ref.provider}"
                continue
            classifier = RequestEnvelope(
                task="Classify the required Crupier operation.",
                input={"task": request.task},
                mode="structured",
                constraints={
                    "temperature": 0,
                    "max_output_tokens": 160,
                    "timeout_seconds": 30,
                },
                metadata={"purpose": "crupier_operation_classifier"},
            )
            card = ModelRegistry(self.client.config).load().get(model_ref.key)
            effective_request, runtime_policy = apply_runtime_policy(model_ref.key, classifier, card)
            budget = request.metadata.get("_crupier_execution_budget")
            reservation = None
            if isinstance(budget, ExecutionBudget):
                reservation = budget.reserve(model=model_ref.key, prompt=prompt, request=effective_request)
                effective_request = request_with_timeout(effective_request, reservation.timeout_seconds)
            call_started = perf_counter()
            try:
                response = adapter.generate(model=model_ref.model, prompt=prompt, request=effective_request)
                if isinstance(budget, ExecutionBudget):
                    budget.ensure_deadline()
                classified = _parse_operation_classification(response.text, available)
                self._record_classifier_call(
                    request,
                    {
                        "role": "operation_classifier",
                        "provider": model_ref.provider,
                        "model": model_ref.key,
                        "latency_ms": int((perf_counter() - call_started) * 1000),
                        "usage": response.usage,
                        "estimated_usd_reserved": reservation.estimated_usd if reservation else None,
                        "metadata": response.metadata
                        | {"classified_operation": classified, "runtime_policy": runtime_policy},
                    },
                )
                return classified
            except (CrupierBudgetExceededError, CrupierExecutionLimitError):
                raise
            except Exception as exc:  # noqa: BLE001 - fallback is explicit and traceable
                last_error = str(exc)
                self._record_classifier_call(
                    request,
                    {
                        "role": "operation_classifier",
                        "provider": model_ref.provider,
                        "model": model_ref.key,
                        "latency_ms": int((perf_counter() - call_started) * 1000),
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )
        if self.client.config.orchestrator.fallback == "error":
            raise CrupierRouteValidationError(last_error or "operation classification failed")
        return deterministic

    def _available_operations(self) -> list[str]:
        available: set[str] = set()
        for card in self.client.registry.allowed_cards():
            adapter = self.client.adapters.get(card.model_ref.provider)
            if adapter is None:
                continue
            if card.model_kind == "chat" and callable(getattr(adapter, "generate", None)):
                available.add("chat")
            elif card.model_kind == "embedding" and callable(getattr(adapter, "embed", None)):
                available.add("embedding")
            else:
                supports = getattr(adapter, "supports_operation", None)
                if callable(supports) and supports(operation=card.model_kind, model=card.model_ref.model):
                    available.add(card.model_kind)
        if not available:
            raise CrupierPolicyError("No executable operations remain for the configured allowlist and adapters.")
        return sorted(available)

    def _finalize_result(
        self,
        result: OperationResult,
        *,
        request: RequestEnvelope,
        dry_run: bool,
        trace: bool | str,
    ) -> OperationResult:
        stored_trace_path = self.client.traces.write(
            project=self.client.config.project.name,
            request=request,
            result=result,
            dry_run=dry_run,
            trace_level=trace,
        )
        if stored_trace_path:
            result.provider_metadata["stored_trace_path"] = str(stored_trace_path)
        if not trace:
            result.trace = None
        return result

    @staticmethod
    def _record_classifier_call(request: RequestEnvelope, call: dict[str, Any]) -> None:
        calls = request.metadata.setdefault("_crupier_orchestrator_calls", [])
        if isinstance(calls, list):
            calls.append(call)

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> OperationResult:
        return self.execute(
            "reranker",
            task=f"Rerank {len(documents)} documents for the supplied query.",
            payload={"query": query, "documents": documents, "top_n": top_n},
            model=model,
            **kwargs,
        )

    def embed(
        self,
        *,
        input: Any,
        dimensions: int | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> OperationResult:
        return self.execute(
            "embedding",
            task="Create embeddings for the supplied input.",
            payload={"input": input, "dimensions": dimensions},
            model=model,
            **kwargs,
        )

    def transcribe(self, *, file: Any, model: str | None = None, **kwargs: Any) -> OperationResult:
        operation_keys = {"language", "response_format", "timestamp_granularities", "temperature", "filename"}
        payload = {"file": file, **{key: kwargs.pop(key) for key in list(kwargs) if key in operation_keys}}
        return self.execute(
            "transcription",
            task="Transcribe the supplied audio accurately.",
            payload=payload,
            model=model,
            **kwargs,
        )

    def synthesize(
        self,
        *,
        input: str,
        voice: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> OperationResult:
        operation_keys = {"response_format", "speed"}
        payload = {
            "input": input,
            "voice": voice,
            **{key: kwargs.pop(key) for key in list(kwargs) if key in operation_keys},
        }
        return self.execute(
            "tts",
            task="Synthesize the supplied text as speech.",
            payload=payload,
            model=model,
            **kwargs,
        )

    def generate_image(self, *, prompt: str, model: str | None = None, **kwargs: Any) -> OperationResult:
        operation_keys = {"n", "size", "response_format", "seed", "guidance"}
        payload = {
            "prompt": prompt,
            **{key: kwargs.pop(key) for key in list(kwargs) if key in operation_keys},
        }
        return self.execute(
            "image_generation",
            task="Generate an image from the supplied prompt.",
            payload=payload,
            model=model,
            **kwargs,
        )

    def edit_image(
        self,
        *,
        prompt: str,
        images: Any,
        model: str | None = None,
        **kwargs: Any,
    ) -> OperationResult:
        operation_keys = {"n", "size", "response_format", "seed", "guidance", "mask"}
        payload = {
            "prompt": prompt,
            "images": images,
            **{key: kwargs.pop(key) for key in list(kwargs) if key in operation_keys},
        }
        return self.execute(
            "image_generation",
            task="Edit the supplied image references according to the prompt.",
            payload=payload,
            model=model,
            **kwargs,
        )

    def _plan(
        self,
        request: RequestEnvelope,
        cards: list[CapabilityCard],
        filters: list[str],
        *,
        dry_run: bool,
    ) -> RoutePlan:
        force_model = bool(request.constraints.get("force_model"))
        single_candidate = len(cards) == 1
        if isinstance(self.client.planner.orchestrator, ModelOrchestrator) and (
            dry_run or force_model or single_candidate
        ):
            context = self.client.planner.build_context(request, cards, filters)
            return DeterministicOrchestrator(
                self.client.config,
                selector=self.client.planner.selector,
            ).plan(context)
        return self.client.planner.plan(request, cards, filters)

    def _filter_operation_candidates(
        self,
        operation: str,
        cards: list[CapabilityCard],
        *,
        payload: dict[str, Any],
    ) -> tuple[list[CapabilityCard], list[Exclusion]]:
        allowed: list[CapabilityCard] = []
        excluded: list[Exclusion] = []
        for card in cards:
            adapter = self.client.adapters.get(card.model_ref.provider)
            supports_operation = getattr(adapter, "supports_operation", None)
            if operation == "embedding":
                supported = callable(getattr(adapter, "embed", None))
            else:
                supported = callable(supports_operation) and supports_operation(
                    operation=operation,
                    model=card.model_ref.model,
                )
            incompatibility = _operation_card_incompatibility(operation, card, payload) if supported else None
            if supported and incompatibility is None:
                allowed.append(card)
            else:
                excluded.append(
                    Exclusion(
                        card.model_ref.key,
                        incompatibility or f"configured adapter cannot execute operation {operation!r}",
                    )
                )
        if not allowed:
            reasons = "; ".join(f"{item.model}: {item.reason}" for item in excluded)
            raise CrupierPolicyError(f"No models remain after operation support checks. {reasons}")
        return allowed, excluded

    @staticmethod
    def _resolve_requested_model(
        model: str | None,
        cards: list[CapabilityCard],
        operation: str,
    ) -> str | None:
        if not model or model == "auto":
            return None
        if ":" in model:
            return ModelRef.parse(model).key
        matches = [
            card.model_ref.key
            for card in cards
            if card.model_ref.model == model and card.model_kind == operation
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise CrupierModelUnsupportedError(
                f"No allowed {operation!r} model has provider model ID {model!r}."
            )
        raise CrupierModelUnsupportedError(
            f"Model ID {model!r} is ambiguous; use provider:model. Matches: {', '.join(matches)}."
        )


def normalize_operation(operation: str) -> str:
    aliases = {
        "embeddings": "embedding",
        "rerank": "reranker",
        "stt": "transcription",
        "speech_to_text": "transcription",
        "speech": "tts",
        "text_to_speech": "tts",
        "image": "image_generation",
        "images": "image_generation",
    }
    normalized = aliases.get(operation.strip().lower(), operation.strip().lower())
    if normalized not in SPECIALIZED_OPERATIONS:
        raise CrupierModelUnsupportedError(
            f"Unsupported operation {operation!r}; expected one of {sorted(SPECIALIZED_OPERATIONS)}."
        )
    return normalized


def _validate_operation_payload(operation: str, payload: dict[str, Any]) -> None:
    if operation == "embedding":
        if payload.get("input") is None:
            raise CrupierModelUnsupportedError("Embedding input is required.")
        dimensions = payload.get("dimensions")
        if dimensions is not None:
            try:
                dimensions = int(dimensions)
            except (TypeError, ValueError) as exc:
                raise CrupierModelUnsupportedError("Embedding dimensions must be an integer.") from exc
            if dimensions <= 0:
                raise CrupierModelUnsupportedError("Embedding dimensions must be positive.")
            payload["dimensions"] = dimensions
    elif operation == "reranker":
        query = payload.get("query")
        documents = payload.get("documents")
        if not isinstance(query, str) or not query.strip():
            raise CrupierModelUnsupportedError("Rerank query must be a non-empty string.")
        if not isinstance(documents, list) or not documents or not all(isinstance(item, str) for item in documents):
            raise CrupierModelUnsupportedError("Rerank documents must be a non-empty list of strings.")
        top_n = payload.get("top_n")
        if top_n is not None:
            try:
                top_n = int(top_n)
            except (TypeError, ValueError) as exc:
                raise CrupierModelUnsupportedError("Rerank top_n must be an integer.") from exc
            if top_n <= 0 or top_n > len(documents):
                raise CrupierModelUnsupportedError("Rerank top_n must be between 1 and the document count.")
            payload["top_n"] = top_n
    elif operation == "transcription" and payload.get("file") is None:
        raise CrupierModelUnsupportedError("Transcription file is required.")
    elif operation == "tts":
        if not isinstance(payload.get("input"), str) or not str(payload["input"]).strip():
            raise CrupierModelUnsupportedError("Text-to-speech input must be a non-empty string.")
        if not isinstance(payload.get("voice"), str) or not str(payload["voice"]).strip():
            raise CrupierModelUnsupportedError("Text-to-speech voice must be a non-empty string.")
    elif operation == "image_generation":
        if not isinstance(payload.get("prompt"), str) or not str(payload["prompt"]).strip():
            raise CrupierModelUnsupportedError("Image prompt must be a non-empty string.")


def _operation_card_incompatibility(
    operation: str,
    card: CapabilityCard,
    payload: dict[str, Any],
) -> str | None:
    if operation != "embedding" or payload.get("dimensions") is None or card.embedding_dimensions is None:
        return None
    requested = int(payload["dimensions"])
    available = int(card.embedding_dimensions)
    dimension_mode = str(card.routing_hints.get("embedding_dimensions_mode") or "maximum")
    if dimension_mode == "fixed" and requested != available:
        return f"embedding output is fixed at {available} dimensions, not requested {requested}"
    if dimension_mode != "fixed" and requested > available:
        return f"embedding output supports at most {available} dimensions, below requested {requested}"
    return None


def _planning_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _planning_value(value) for key, value in payload.items() if value is not None}


def _planning_value(value: Any) -> Any:
    if isinstance(value, bytes | bytearray):
        return {"type": "bytes", "size_bytes": len(value)}
    if isinstance(value, tuple) and len(value) in {2, 3} and isinstance(value[1], bytes | bytearray):
        summary = {
            "type": "upload",
            "name": str(value[0] or "upload"),
            "size_bytes": len(value[1]),
        }
        if len(value) == 3 and value[2]:
            summary["mime_type"] = str(value[2])
        return summary
    if isinstance(value, Path):
        return {"type": "file", "name": value.name, "suffix": value.suffix, "exists": value.is_file()}
    if isinstance(value, str):
        redacted = _redact_planning_text(value)
        return redacted if len(redacted) <= 1200 else redacted[:1197] + "..."
    if isinstance(value, list):
        return [_planning_value(item) for item in value[:12]]
    if isinstance(value, tuple):
        return [_planning_value(item) for item in value[:12]]
    if isinstance(value, dict):
        return {str(key): _planning_value(item) for key, item in list(value.items())[:20]}
    if hasattr(value, "read"):
        return {"type": "file_object", "name": str(getattr(value, "name", "upload"))}
    return value


def _successful_orchestrator(calls: list[dict[str, Any]]) -> str | None:
    for call in reversed(calls):
        if call.get("model") and not call.get("error"):
            return str(call["model"])
    return None


def _operation_warnings(operation: str) -> list[str]:
    if operation in {"tts", "transcription", "image_generation", "reranker"}:
        return [
            "Provider pricing for this operation is not token-comparable; estimated_usd excludes unreported "
            "subscription or quota consumption."
        ]
    return []


def _parse_operation_classification(text: str, available: list[str]) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise CrupierRouteValidationError("Operation classifier did not return a JSON object.")
    try:
        data = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise CrupierRouteValidationError(f"Operation classifier returned invalid JSON: {exc}") from exc
    operation = str(data.get("operation") or "").strip().lower()
    if operation not in available:
        raise CrupierRouteValidationError(
            f"Operation classifier selected {operation!r}, outside available_operations={available!r}."
        )
    confidence = data.get("confidence")
    if confidence is not None and (not isinstance(confidence, int | float) or not 0 <= float(confidence) <= 1):
        raise CrupierRouteValidationError("Operation classifier confidence must be between 0 and 1.")
    return operation


def _deterministic_operation(
    task: str,
    *,
    input: Any,
    files: list[Any],
    available: list[str],
) -> str:
    text = f"{task} {_planning_value(input)}".lower()
    structured = input if isinstance(input, dict) else {}
    candidates = []
    if isinstance(structured, dict) and structured.get("documents") and structured.get("query"):
        candidates.append("reranker")
    if any(marker in text for marker in ("rerank", "reordena", "ordena por relevancia", "rank documents")):
        candidates.append("reranker")
    if any(
        marker in text
        for marker in (
            "embedding",
            "embeddings",
            "vector semantico",
            "vector semántico",
            "semantic vector",
        )
    ):
        candidates.append("embedding")
    if any(
        marker in text
        for marker in (
            "transcribe",
            "transcribir",
            "transcripcion",
            "transcripción",
            "speech to text",
            "audio a texto",
        )
    ):
        candidates.append("transcription")
    if any(
        marker in text
        for marker in (
            "text to speech",
            "texto a voz",
            "sintetiza la voz",
            "sintetizar voz",
            "genera audio hablado",
            "read this aloud",
        )
    ):
        candidates.append("tts")
    generation = any(
        marker in text
        for marker in ("genera", "generar", "crea", "crear", "dibuja", "generate", "create", "edit", "edita")
    )
    visual = any(marker in text for marker in ("imagen", "image", "foto", "ilustracion", "ilustración", "picture"))
    if generation and visual:
        candidates.append("image_generation")
    if files and candidates and "transcription" in candidates:
        candidates.insert(0, "transcription")
    for operation in candidates:
        if operation in available:
            return operation
    if "chat" in available:
        return "chat"
    return available[0]


def _operation_payload(
    operation: str,
    *,
    task: str,
    input: Any,
    files: list[Any],
    supplied: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(input) if isinstance(input, dict) else {}
    payload.update(supplied)
    if operation == "embedding":
        payload.setdefault("input", input if input is not None else task)
    elif operation == "reranker":
        payload.setdefault("query", task)
        payload.setdefault("documents", [])
    elif operation == "transcription":
        if "file" not in payload and files:
            payload["file"] = _file_value(files[0])
    elif operation == "tts":
        payload.setdefault("input", input if isinstance(input, str) and input else task)
    elif operation == "image_generation":
        payload.setdefault("prompt", input if isinstance(input, str) and input else task)
        if files and "images" not in payload:
            payload["images"] = [_file_value(item) for item in files]
    return payload


def _file_value(value: Any) -> Any:
    uri = getattr(value, "uri", None)
    return uri if uri is not None else value


def _redact_planning_text(text: str) -> str:
    redacted = text
    for pattern, replacement in _PLANNING_SECRET_REPLACERS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


_PLANNING_SECRET_REPLACERS = (
    (re.compile(("s" + "k-") + r"[A-Za-z0-9_\-]{10,}"), "[redacted]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{12,}", re.IGNORECASE), r"\1[redacted]"),
    (re.compile(r"([A-Z][A-Z0-9_]*_API_KEY=)[^\s]+"), r"\1[redacted]"),
)
