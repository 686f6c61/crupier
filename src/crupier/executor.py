"""Route execution."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter, sleep

from .adapters import AdapterResponse, ProviderAdapter
from .adapters.common import build_prompt
from .config import CrupierConfig
from .costs import actual_cost_from_calls
from .errors import (
    CrupierProviderAuthError,
    CrupierProviderRateLimitError,
    CrupierModelUnsupportedError,
    CrupierProviderUnavailableError,
    CrupierStructuredOutputError,
)
from .models import CostEstimate, CrupierResult, DecisionTrace, RequestEnvelope, RoutePlan
from .multimodal import can_execute_native_images
from .registry import ModelRegistry
from .structured import build_repair_prompt, build_structured_prompt, parse_and_validate_json, schema_from_request
from .tools import (
    ToolExecution,
    build_tool_final_prompt,
    build_tool_planning_prompt,
    execute_tool_plan,
    normalize_tools,
    parse_tool_plan,
)


class RouteExecutor:
    def __init__(self, config: CrupierConfig, adapters: dict[str, ProviderAdapter] | None = None):
        self.config = config
        self.adapters = adapters or {}

    def execute(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        *,
        dry_run: bool = True,
    ) -> CrupierResult:
        started = perf_counter()
        if not dry_run:
            return self._execute_real(request, plan, trace, started)

        trace.provider_calls.extend(
            {"role": step.role, "model": step.model, "models": step.models, "dry_run": True}
            for step in plan.steps
        )
        latency_ms = int((perf_counter() - started) * 1000)
        trace.latency_ms = latency_ms
        trace.cost = CostEstimate(estimated_usd=plan.estimated_cost.estimated_usd, actual_usd=0.0)
        trace.final_quality_signals = {
            "dry_run": True,
            "note": "No provider calls were made.",
        }

        output_text = self._dry_run_text(request, plan)
        output_json = None
        if request.response_schema is not None or request.constraints.get("response_schema"):
            output_json = {
                "dry_run": True,
                "route": plan.to_dict(),
                "message": "Structured provider output is not generated in dry-run mode.",
            }

        return CrupierResult(
            output_text=output_text,
            output_json=output_json,
            raw_outputs=[],
            route=plan,
            trace=trace,
            cost=trace.cost,
            latency_ms=latency_ms,
            warnings=[
                "Dry run only: no provider calls were made.",
                "Pricing is unknown in the seed registry; estimated cost is 0.0 until provider pricing refresh exists.",
            ],
            provider_metadata={"dry_run": True},
        )

    @staticmethod
    def _dry_run_text(request: RequestEnvelope, plan: RoutePlan) -> str:
        mode = request.mode or "default"
        models = plan.model_summary or "no models"
        text = (
            f"Crupier dry-run planned a {plan.strategy!r} route for mode {mode!r}. "
            f"Models: {models}. Reason: {plan.reason}"
        )
        if request.file_plan is not None:
            representations = ", ".join(
                f"{item.kind}->{item.representation}" for item in request.file_plan.representations
            )
            text += f" File plan: {representations}."
        return text

    def _execute_real(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        started: float,
    ) -> CrupierResult:
        if request.files and not can_execute_native_images(request.file_plan):
            raise CrupierModelUnsupportedError(
                "Real provider execution is currently implemented only for native image inputs. "
                "Use dry_run=True to inspect routing for PDFs/audio/video/documents, or extract them before calling."
            )
        raw_outputs: list[AdapterResponse] = []
        warnings: list[str] = []
        output_json = None
        tool_ledger: list[ToolExecution] = []
        structured_schema = schema_from_request(request)

        if request.tools:
            output_text, output_json, tool_ledger = self._execute_tools(
                request,
                plan,
                trace,
                raw_outputs,
                structured_schema,
            )
        elif structured_schema is not None:
            output_json, output_text = self._execute_structured(request, plan, trace, raw_outputs, structured_schema)
        elif plan.strategy in {"single", "local_first"}:
            response = self._execute_first_model(request, plan, trace, raw_outputs)
            output_text = response.text
        elif plan.strategy == "fallback":
            response = self._execute_fallback(request, plan, trace, raw_outputs)
            output_text = response.text
        elif plan.strategy == "cascade":
            response = self._execute_first_model(request, plan, trace, raw_outputs)
            warnings.append("Cascade escalation ran the primary step only because no escalation validator is configured.")
            output_text = response.text
        elif plan.strategy == "panel":
            output_text = self._execute_panel(request, plan, trace, raw_outputs)
        elif plan.strategy == "fusion":
            output_text = self._execute_fusion(request, plan, trace, raw_outputs)
        elif plan.strategy == "critique_repair":
            output_text = self._execute_critique_repair(request, plan, trace, raw_outputs)
        else:
            response = self._execute_first_model(request, plan, trace, raw_outputs)
            output_text = response.text
            warnings.append(f"Strategy {plan.strategy!r} executed as first-model route.")

        latency_ms = int((perf_counter() - started) * 1000)
        trace.latency_ms = latency_ms
        actual_usd = self._actual_cost(trace.provider_calls)
        trace.cost = CostEstimate(estimated_usd=plan.estimated_cost.estimated_usd, actual_usd=actual_usd)
        final_quality_signals = {"real_provider_calls": True, "tool_calls": len(tool_ledger)}
        provider_retry_errors = [
            item for item in trace.errors if item.get("phase") == "provider_call" and item.get("retryable")
        ]
        if provider_retry_errors:
            final_quality_signals["provider_retry_errors"] = len(provider_retry_errors)
        file_context = request.metadata.get("extracted_file_context")
        if isinstance(file_context, dict):
            final_quality_signals["file_context"] = {
                "files": file_context.get("files", []),
                "warnings": file_context.get("warnings", []),
                "max_chars": file_context.get("max_chars"),
            }
        trace.final_quality_signals = final_quality_signals

        include_raw = bool(request.constraints.get("include_raw_outputs", False))
        return CrupierResult(
            output_text=output_text,
            output_json=output_json,
            raw_outputs=[item.raw for item in raw_outputs] if include_raw else [],
            route=plan,
            trace=trace,
            cost=trace.cost,
            latency_ms=latency_ms,
            warnings=warnings,
            provider_metadata={
                "dry_run": False,
                "calls": [item.metadata | {"usage": item.usage} for item in raw_outputs],
                "tool_calls": [item.to_dict() for item in tool_ledger],
            },
        )

    def _execute_tools(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        schema: dict[str, object] | None,
    ) -> tuple[str, object | None, list[ToolExecution]]:
        tools = normalize_tools(request.tools)
        model = self._models_in_execution_order(plan)[0]
        planning_prompt = build_tool_planning_prompt(request, tools, response_schema=schema)
        planning = self._call_model(
            model,
            planning_prompt,
            self._request_without_response_schema(request),
            trace,
            raw_outputs,
            role="tool_planner",
        )
        calls, final = parse_tool_plan(planning.text)
        if not calls:
            output_text = final or planning.text
            output_json = self._parse_or_repair_structured(
                output_text,
                request,
                trace,
                raw_outputs,
                schema,
                model=model,
            )
            return output_text, output_json, []

        executions = execute_tool_plan(calls, tools, request)
        final_prompt = build_tool_final_prompt(request, executions, response_schema=schema)
        final_response = self._call_model(
            model,
            final_prompt,
            self._request_with_response_schema(request, schema),
            trace,
            raw_outputs,
            role="tool_final",
        )
        output_json = self._parse_or_repair_structured(
            final_response.text,
            request,
            trace,
            raw_outputs,
            schema,
            model=model,
        )
        return final_response.text, output_json, executions

    def _execute_structured(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        schema: dict[str, object],
    ) -> tuple[object, str]:
        last_error: Exception | None = None
        prompt = build_structured_prompt(request, schema)
        structured_request = self._request_with_response_schema(request, schema)
        for model in self._models_in_execution_order(plan):
            response = self._call_model(model, prompt, structured_request, trace, raw_outputs, role="structured")
            try:
                data = parse_and_validate_json(response.text, schema)
                return data, response.text
            except CrupierStructuredOutputError as exc:
                last_error = exc
                repair_prompt = build_repair_prompt(request, schema, bad_output=response.text, error=str(exc))
                repair = self._call_model(
                    model,
                    repair_prompt,
                    structured_request,
                    trace,
                    raw_outputs,
                    role="structured_repair",
                )
                try:
                    data = parse_and_validate_json(repair.text, schema)
                    return data, repair.text
                except CrupierStructuredOutputError as repair_exc:
                    last_error = repair_exc
                    trace.errors.append({"model": model, "error": str(repair_exc), "phase": "structured_output"})
        raise CrupierStructuredOutputError(f"Structured output failed for all route models. Last error: {last_error}")

    def _parse_or_repair_structured(
        self,
        text: str,
        request: RequestEnvelope,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        schema: dict[str, object] | None,
        *,
        model: str,
    ) -> object | None:
        if schema is None:
            return None
        try:
            return parse_and_validate_json(text, schema)
        except CrupierStructuredOutputError as exc:
            repair_prompt = build_repair_prompt(request, schema, bad_output=text, error=str(exc))
            repair = self._call_model(
                model,
                repair_prompt,
                self._request_with_response_schema(request, schema),
                trace,
                raw_outputs,
                role="structured_repair",
            )
            return parse_and_validate_json(repair.text, schema)

    @staticmethod
    def _request_with_response_schema(request: RequestEnvelope, schema: dict[str, object] | None) -> RequestEnvelope:
        if schema is None:
            return RouteExecutor._request_without_response_schema(request)
        constraints = dict(request.constraints)
        constraints.pop("response_schema", None)
        return replace(request, response_schema=schema, constraints=constraints)

    @staticmethod
    def _request_without_response_schema(request: RequestEnvelope) -> RequestEnvelope:
        constraints = dict(request.constraints)
        constraints.pop("response_schema", None)
        return replace(request, response_schema=None, constraints=constraints)

    def _execute_first_model(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
    ) -> AdapterResponse:
        for step in plan.steps:
            model = step.model or (step.models[0] if step.models else None)
            if model:
                return self._call_model(model, build_prompt(request), request, trace, raw_outputs, role=step.role)
        raise CrupierProviderUnavailableError("Route has no executable model step.")

    def _execute_fallback(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
    ) -> AdapterResponse:
        models: list[str] = []
        for step in plan.steps:
            models.extend(step.models)
            if step.model:
                models.append(step.model)
        last_error: Exception | None = None
        for model in models:
            try:
                return self._call_model(model, build_prompt(request), request, trace, raw_outputs, role="fallback")
            except Exception as exc:  # noqa: BLE001 - fallback records provider-specific failures
                last_error = exc
                trace.fallbacks.append({"model": model, "error": str(exc)})
        raise CrupierProviderUnavailableError(f"All fallback models failed. Last error: {last_error}") from last_error

    def _execute_panel(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
    ) -> str:
        panel_models = next((step.models for step in plan.steps if step.role == "panel"), [])
        outputs: list[str] = []
        for model in panel_models:
            try:
                response = self._call_model(model, build_prompt(request), request, trace, raw_outputs, role="panel")
                outputs.append(f"## {model}\n{response.text}")
            except Exception as exc:  # noqa: BLE001
                trace.errors.append({"model": model, "error": str(exc)})
        if not outputs:
            raise CrupierProviderUnavailableError("All panel models failed.")
        return "\n\n".join(outputs)

    def _execute_fusion(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
    ) -> str:
        panel_models = next((step.models for step in plan.steps if step.role == "panel"), [])
        panel_outputs: list[str] = []
        for model in panel_models:
            try:
                response = self._call_model(model, build_prompt(request), request, trace, raw_outputs, role="panel")
                panel_outputs.append(f"Model {model}:\n{response.text}")
            except Exception as exc:  # noqa: BLE001
                trace.errors.append({"model": model, "error": str(exc)})
        if not panel_outputs:
            raise CrupierProviderUnavailableError("Fusion panel failed; no model outputs available.")

        judge_model = self._model_for_role(plan, "judge") or panel_models[0]
        judge_prompt = build_prompt(
            request,
            extra=(
                "Panel outputs:\n"
                + "\n\n---\n\n".join(panel_outputs)
                + "\n\nReturn a concise structured synthesis with consensus, contradictions, gaps, and risks. "
                "Do not include hidden chain-of-thought."
            ),
        )
        judge = self._call_model(judge_model, judge_prompt, request, trace, raw_outputs, role="judge")

        writer_model = self._model_for_role(plan, "final_writer") or judge_model
        final_prompt = build_prompt(
            request,
            extra=(
                "Judge synthesis:\n"
                + judge.text
                + "\n\nWrite the final answer for the user. Be direct, cite uncertainty, and do not include hidden reasoning."
            ),
        )
        final = self._call_model(writer_model, final_prompt, request, trace, raw_outputs, role="final_writer")
        return final.text

    def _execute_critique_repair(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
    ) -> str:
        generator_model = self._model_for_role(plan, "generator") or plan.models[0]
        critic_model = self._model_for_role(plan, "critic") or generator_model
        repair_model = self._model_for_role(plan, "repair") or generator_model

        draft = self._call_model(generator_model, build_prompt(request), request, trace, raw_outputs, role="generator")
        critique_prompt = build_prompt(
            request,
            extra=(
                "Draft answer:\n"
                + draft.text
                + "\n\nCritique this draft for correctness, missing constraints, cost/latency tradeoffs, and tool-risk. "
                "Do not include hidden chain-of-thought."
            ),
        )
        critique = self._call_model(critic_model, critique_prompt, request, trace, raw_outputs, role="critic")
        repair_prompt = build_prompt(
            request,
            extra=(
                "Draft answer:\n"
                + draft.text
                + "\n\nCritique:\n"
                + critique.text
                + "\n\nProduce the repaired final answer. Do not include hidden chain-of-thought."
            ),
        )
        final = self._call_model(repair_model, repair_prompt, request, trace, raw_outputs, role="repair")
        return final.text

    def _call_model(
        self,
        model_key: str,
        prompt: str,
        request: RequestEnvelope,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        *,
        role: str,
    ) -> AdapterResponse:
        provider, model = model_key.split(":", 1)
        adapter = self.adapters.get(provider)
        if adapter is None:
            raise CrupierProviderUnavailableError(
                f"No adapter configured for provider {provider!r}. Enable [providers.{provider}] and install the matching extra."
            )
        max_retries = self._provider_retry_budget(request)
        backoff_seconds = self._provider_retry_backoff_seconds(request)
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 2):
            call_started = perf_counter()
            try:
                response = adapter.generate(model=model, prompt=prompt, request=request)
            except (CrupierProviderAuthError, CrupierProviderRateLimitError, CrupierProviderUnavailableError) as exc:
                duration_ms = int((perf_counter() - call_started) * 1000)
                retryable = attempt <= max_retries and self._provider_error_retryable(exc)
                last_error = exc
                trace.errors.append(
                    {
                        "phase": "provider_call",
                        "role": role,
                        "provider": provider,
                        "model": model_key,
                        "attempt": attempt,
                        "max_provider_retries": max_retries,
                        "latency_ms": duration_ms,
                        "retryable": retryable,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    }
                )
                if not retryable:
                    raise
                if backoff_seconds > 0:
                    sleep(backoff_seconds * (2 ** (attempt - 1)))
                continue
            duration_ms = int((perf_counter() - call_started) * 1000)
            raw_outputs.append(response)
            trace.provider_calls.append(
                {
                    "role": role,
                    "provider": provider,
                    "model": model_key,
                    "attempt": attempt,
                    "latency_ms": duration_ms,
                    "usage": response.usage,
                    "metadata": response.metadata,
                }
            )
            return response
        raise CrupierProviderUnavailableError(f"Provider call failed after retries. Last error: {last_error}") from last_error

    def _provider_retry_budget(self, request: RequestEnvelope) -> int:
        value = request.constraints.get("max_provider_retries", self.config.routing.max_provider_retries)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return max(0, int(self.config.routing.max_provider_retries))

    def _provider_retry_backoff_seconds(self, request: RequestEnvelope) -> float:
        value = request.constraints.get("retry_backoff_seconds", self.config.routing.retry_backoff_seconds)
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return max(0.0, float(self.config.routing.retry_backoff_seconds))

    @staticmethod
    def _provider_error_retryable(exc: Exception) -> bool:
        if isinstance(exc, CrupierProviderAuthError):
            return False
        if isinstance(exc, CrupierProviderRateLimitError):
            return True
        return bool(getattr(exc, "retryable", True))

    def _actual_cost(self, calls: list[dict[str, object]]) -> float | None:
        try:
            cards = ModelRegistry(self.config).list(allowed_only=False)
        except Exception:  # noqa: BLE001 - cost accounting should not hide a successful provider call
            return None
        return actual_cost_from_calls(calls, cards)

    @staticmethod
    def _models_in_execution_order(plan: RoutePlan) -> list[str]:
        models: list[str] = []
        for step in plan.steps:
            if step.model and step.model not in models:
                models.append(step.model)
            for model in step.models:
                if model not in models:
                    models.append(model)
        return models

    @staticmethod
    def _model_for_role(plan: RoutePlan, role: str) -> str | None:
        for step in plan.steps:
            if step.role == role:
                return step.model or (step.models[0] if step.models else None)
        return None
