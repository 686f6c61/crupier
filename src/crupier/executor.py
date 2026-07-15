"""Route execution."""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from threading import RLock
from time import perf_counter, sleep

from .adapters import AdapterResponse, ProviderAdapter
from .adapters.common import build_prompt
from .budgets import ExecutionBudget, request_with_timeout
from .config import CrupierConfig
from .costs import actual_cost_from_calls, usage_estimated_cost_from_calls
from .errors import (
    CrupierBudgetExceededError,
    CrupierExecutionLimitError,
    CrupierProviderAuthError,
    CrupierProviderRateLimitError,
    CrupierProviderUnavailableError,
    CrupierRouteValidationError,
    CrupierStructuredOutputError,
)
from .models import CapabilityCard, CostEstimate, CrupierResult, DecisionTrace, RequestEnvelope, RoutePlan
from .prompts import (
    build_critique_instruction,
    build_repair_instruction,
    build_tool_critique_instruction,
    build_tool_repair_instruction,
)
from .registry import ModelRegistry
from .runtime_policy import apply_runtime_policy
from .structured import build_repair_prompt, build_structured_prompt, parse_and_validate_json, schema_from_request
from .tools import (
    ToolExecution,
    build_tool_final_prompt,
    build_tool_planning_prompt,
    build_tool_replanning_prompt,
    execute_tool_plan,
    extract_final_answer,
    normalize_tools,
    parse_tool_plan,
)


def _non_negative_int(value: object, *, default: int) -> int:
    try:
        return max(0, int(value))  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return max(0, int(default))


def _positive_int(value: object, *, default: int) -> int:
    return max(1, _non_negative_int(value, default=default))


def _optional_non_negative_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class RouteExecutor:
    def __init__(self, config: CrupierConfig, adapters: dict[str, ProviderAdapter] | None = None):
        self.config = config
        self.adapters = adapters or {}
        self._provider_failure_counts: dict[str, int] = {}
        self._provider_circuit_open_until: dict[str, float] = {}
        self._state_lock = RLock()
        try:
            self._cards = ModelRegistry(config).load()
        except Exception:  # noqa: BLE001 - execution can still proceed without optional runtime hints
            self._cards = {}

    def execute(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        *,
        dry_run: bool = True,
        budget: ExecutionBudget | None = None,
    ) -> CrupierResult:
        started = perf_counter()
        if not dry_run:
            return self._execute_real(request, plan, trace, started, budget)

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
        budget: ExecutionBudget | None = None,
    ) -> CrupierResult:
        budget = budget or ExecutionBudget(self.config, request, self._budget_cards())
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
                budget,
            )
            if plan.strategy == "critique_repair":
                output_text, output_json = self._execute_tool_critique_repair(
                    request,
                    plan,
                    output_text,
                    tool_ledger,
                    trace,
                    raw_outputs,
                    structured_schema,
                    budget,
                )
        elif structured_schema is not None:
            output_json, output_text = self._execute_structured(
                request, plan, trace, raw_outputs, structured_schema, budget
            )
        elif plan.strategy in {"single", "local_first"}:
            response = self._execute_first_model(request, plan, trace, raw_outputs, budget)
            output_text = response.text
        elif plan.strategy == "fallback":
            response = self._execute_fallback(request, plan, trace, raw_outputs, budget)
            output_text = response.text
        elif plan.strategy == "cascade":
            response = self._execute_cascade(request, plan, trace, raw_outputs, budget)
            output_text = response.text
        elif plan.strategy == "panel":
            output_text = self._execute_panel(request, plan, trace, raw_outputs, budget)
        elif plan.strategy == "fusion":
            output_text = self._execute_fusion(request, plan, trace, raw_outputs, budget)
        elif plan.strategy == "critique_repair":
            output_text = self._execute_critique_repair(request, plan, trace, raw_outputs, budget)
        elif plan.strategy == "delegate":
            output_text = self._execute_delegate(request, plan, trace, budget)
        else:
            response = self._execute_first_model(request, plan, trace, raw_outputs, budget)
            output_text = response.text
            warnings.append(f"Strategy {plan.strategy!r} executed as first-model route.")

        latency_ms = int((perf_counter() - started) * 1000)
        trace.latency_ms = latency_ms
        actual_usd = self._actual_cost(trace.provider_calls)
        usage_estimate = self._usage_estimated_cost(trace.provider_calls)
        reserved_estimate = _optional_non_negative_float(
            budget.snapshot()["estimated_cost_reserved_usd"]
        ) or 0.0
        trace.cost = CostEstimate(
            estimated_usd=usage_estimate if usage_estimate is not None else reserved_estimate,
            actual_usd=actual_usd,
        )
        final_quality_signals: dict[str, object] = dict(trace.final_quality_signals)
        final_quality_signals.update(
            {
                "real_provider_calls": True,
                "tool_calls": len(tool_ledger),
                "execution_budget": budget.snapshot(),
            }
        )
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
        budget: ExecutionBudget,
    ) -> tuple[str, object | None, list[ToolExecution]]:
        tools = normalize_tools(request.tools)
        model = self._models_in_execution_order(plan)[0]
        max_rounds = self._max_tool_rounds(request)
        executions: list[ToolExecution] = []
        final: str | None = None
        for round_index in range(max_rounds):
            planning_prompt = (
                build_tool_planning_prompt(request, tools, response_schema=schema)
                if round_index == 0
                else build_tool_replanning_prompt(request, tools, executions, response_schema=schema)
            )
            model, planning = self._call_plan_role(
                plan,
                "generator",
                planning_prompt,
                self._request_without_response_schema(request),
                trace,
                raw_outputs,
                trace_role=f"tool_planner_round_{round_index + 1}",
                budget=budget,
            )
            calls, final = parse_tool_plan(planning.text)
            max_calls_per_round = self._max_tool_calls_per_round(request)
            if len(calls) > max_calls_per_round:
                raise CrupierRouteValidationError(
                    f"Tool planner requested {len(calls)} calls in one round, above "
                    f"max_tool_calls_per_round={max_calls_per_round}."
                )
            if not calls:
                if executions and schema is not None:
                    break
                output_text = final or planning.text
                output_json = self._parse_or_repair_structured(
                    output_text,
                    request,
                    trace,
                    raw_outputs,
                    schema,
                    model=model,
                    budget=budget,
                )
                return output_text, output_json, executions
            executions.extend(
                execute_tool_plan(
                    calls,
                    tools,
                    request,
                    previous_executions=executions,
                    max_result_chars=self._max_tool_result_chars(request),
                )
            )

        final_prompt = build_tool_final_prompt(request, executions, response_schema=schema)
        model, final_response = self._call_plan_role(
            plan,
            "generator",
            final_prompt,
            self._request_with_response_schema(request, schema),
            trace,
            raw_outputs,
            trace_role="tool_final",
            budget=budget,
        )
        output_json = self._parse_or_repair_structured(
            final_response.text,
            request,
            trace,
            raw_outputs,
            schema,
            model=model,
            budget=budget,
        )
        return final_response.text, output_json, executions

    def _execute_tool_critique_repair(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        draft: str,
        executions: list[ToolExecution],
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        schema: dict[str, object] | None,
        budget: ExecutionBudget,
    ) -> tuple[str, object | None]:
        tool_results = json.dumps(
            [execution.to_dict() for execution in executions],
            ensure_ascii=False,
            sort_keys=True,
        )
        critique_prompt = build_prompt(
            request,
            extra=build_tool_critique_instruction(
                tool_results=tool_results,
                draft=draft,
            ),
        )
        _, critique = self._call_plan_role(
            plan,
            "critic",
            critique_prompt,
            self._request_without_response_schema(request),
            trace,
            raw_outputs,
            trace_role="tool_critic",
            budget=budget,
        )
        repair_prompt = build_prompt(
            request,
            extra=build_tool_repair_instruction(
                tool_results=tool_results,
                draft=draft,
                critique=critique.text,
                structured_output=schema is not None,
            ),
        )
        repair_model, repair = self._call_plan_role(
            plan,
            "repair",
            repair_prompt,
            self._request_with_response_schema(request, schema),
            trace,
            raw_outputs,
            trace_role="tool_repair",
            budget=budget,
        )
        repaired_text = extract_final_answer(repair.text) if schema is None else repair.text
        output_json = self._parse_or_repair_structured(
            repaired_text,
            request,
            trace,
            raw_outputs,
            schema,
            model=repair_model,
            budget=budget,
        )
        return repaired_text, output_json

    def _execute_structured(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        schema: dict[str, object],
        budget: ExecutionBudget,
    ) -> tuple[object, str]:
        last_error: Exception | None = None
        prompt = build_structured_prompt(request, schema)
        structured_request = self._request_with_response_schema(request, schema)
        for model in self._models_in_execution_order(plan):
            response = self._call_model(
                model, prompt, structured_request, trace, raw_outputs, role="structured", budget=budget
            )
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
                    budget=budget,
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
        budget: ExecutionBudget,
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
                budget=budget,
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
        budget: ExecutionBudget,
    ) -> AdapterResponse:
        for step in plan.steps:
            model = step.model or (step.models[0] if step.models else None)
            if model:
                return self._call_model(
                    model,
                    build_prompt(request),
                    request,
                    trace,
                    raw_outputs,
                    role=step.role,
                    budget=budget,
                )
        raise CrupierProviderUnavailableError("Route has no executable model step.")

    def _execute_fallback(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        budget: ExecutionBudget,
    ) -> AdapterResponse:
        models: list[str] = []
        for step in plan.steps:
            models.extend(step.models)
            if step.model:
                models.append(step.model)
        last_error: Exception | None = None
        for model in models:
            try:
                return self._call_model(
                    model,
                    build_prompt(request),
                    request,
                    trace,
                    raw_outputs,
                    role="fallback",
                    budget=budget,
                )
            except (CrupierBudgetExceededError, CrupierExecutionLimitError):
                raise
            except Exception as exc:  # noqa: BLE001 - fallback records provider-specific failures
                last_error = exc
                trace.fallbacks.append({"model": model, "error": str(exc)})
        raise CrupierProviderUnavailableError(f"All fallback models failed. Last error: {last_error}") from last_error

    def _execute_cascade(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        budget: ExecutionBudget,
    ) -> AdapterResponse:
        models = self._cascade_models(plan)
        if not models:
            raise CrupierProviderUnavailableError("Cascade route has no executable model step.")

        last_response: AdapterResponse | None = None
        last_error: Exception | None = None
        last_validation_reason: str | None = None
        for index, model in enumerate(models):
            try:
                response = self._call_model(
                    model,
                    build_prompt(request),
                    request,
                    trace,
                    raw_outputs,
                    role="cascade",
                    budget=budget,
                )
            except (CrupierBudgetExceededError, CrupierExecutionLimitError):
                raise
            except Exception as exc:  # noqa: BLE001 - cascade escalates past provider failures
                last_error = exc
                next_model = models[index + 1] if index + 1 < len(models) else None
                trace.fallbacks.append(
                    {"model": model, "error": str(exc), "phase": "cascade_provider_call", "next_model": next_model}
                )
                continue
            last_response = response
            ok, reason = self._cascade_response_sufficient(
                request, plan, response, trace, raw_outputs, budget
            )
            if ok:
                if index > 0:
                    trace.fallbacks.append(
                        {"model": model, "phase": "cascade_escalation", "reason": "escalated response accepted"}
                    )
                return response
            last_validation_reason = reason
            next_model = models[index + 1] if index + 1 < len(models) else None
            trace.fallbacks.append(
                {
                    "model": model,
                    "phase": "cascade_validation",
                    "reason": reason,
                    "next_model": next_model,
                }
            )

        if last_response is not None:
            raise CrupierProviderUnavailableError(
                "Cascade exhausted every model without a sufficient response. "
                f"Last validation reason: {last_validation_reason or 'unknown'}",
                retryable=False,
            )
        raise CrupierProviderUnavailableError(f"All cascade models failed. Last error: {last_error}") from last_error

    def _cascade_response_sufficient(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        response: AdapterResponse,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        budget: ExecutionBudget,
    ) -> tuple[bool, str]:
        validator_model = self._model_for_role(plan, "validator") or request.constraints.get("cascade_validator_model")
        if validator_model:
            return self._model_validate_cascade_response(
                str(validator_model),
                request,
                response,
                trace,
                raw_outputs,
                budget,
            )
        return self._heuristic_validate_cascade_response(request, response)

    def _model_validate_cascade_response(
        self,
        model: str,
        request: RequestEnvelope,
        response: AdapterResponse,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        budget: ExecutionBudget,
    ) -> tuple[bool, str]:
        prompt = build_prompt(
            request,
            extra=(
                "Candidate answer:\n"
                + response.text
                + "\n\nReturn only JSON: {\"sufficient\": true|false, \"reason\": \"short reason\"}. "
                "Mark sufficient=false if the answer is empty, refuses without cause, ignores required format, "
                "or says it lacks enough information."
            ),
        )
        verdict = self._call_model(
            model,
            prompt,
            request,
            trace,
            raw_outputs,
            role="cascade_validator",
            budget=budget,
        )
        try:
            data = json.loads(verdict.text.strip())
        except json.JSONDecodeError:
            return self._heuristic_validate_cascade_response(request, response)
        if not isinstance(data, dict):
            return self._heuristic_validate_cascade_response(request, response)
        sufficient = bool(data.get("sufficient"))
        default_reason = "validator marked response sufficient" if sufficient else "validator rejected response"
        reason = str(data.get("reason") or default_reason)
        return sufficient, reason

    def _heuristic_validate_cascade_response(
        self,
        request: RequestEnvelope,
        response: AdapterResponse,
    ) -> tuple[bool, str]:
        text = response.text.strip()
        if not text:
            return False, "empty response"
        min_chars = request.constraints.get("cascade_min_output_chars")
        if min_chars is not None:
            try:
                if len(text) < int(min_chars):
                    return False, f"response shorter than cascade_min_output_chars={min_chars}"
            except (TypeError, ValueError):
                pass
        lowered = text.lower()
        uncertainty_markers = [
            "i don't know",
            "i do not know",
            "cannot answer",
            "can't answer",
            "not enough information",
            "insufficient information",
            "no puedo responder",
            "no tengo suficiente informacion",
            "no tengo suficiente información",
        ]
        if any(marker in lowered for marker in uncertainty_markers):
            return False, "response contains uncertainty/refusal marker"
        return True, "heuristic validation passed"

    def _execute_panel(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        budget: ExecutionBudget,
    ) -> str:
        panel_models = next((step.models for step in plan.steps if step.role == "panel"), [])
        outputs = [
            f"## {model}\n{response.text}"
            for model, response in self._run_panel_models(
                request, panel_models, trace, raw_outputs, budget
            )
        ]
        if not outputs:
            raise CrupierProviderUnavailableError("All panel models failed.")
        return "\n\n".join(outputs)

    def _execute_fusion(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        budget: ExecutionBudget,
    ) -> str:
        panel_models = next((step.models for step in plan.steps if step.role == "panel"), [])
        panel_results = self._run_panel_models(
            request, panel_models, trace, raw_outputs, budget
        )
        required_panel_outputs = min(2, len(panel_models))
        trace.final_quality_signals.update(
            {
                "fusion_panel_planned": len(panel_models),
                "fusion_panel_successful": len(panel_results),
                "fusion_panel_quorum_required": required_panel_outputs,
                "fusion_panel_quorum": len(panel_results) >= required_panel_outputs,
            }
        )
        if len(panel_results) < required_panel_outputs:
            raise CrupierProviderUnavailableError(
                "Fusion requires at least 2 non-empty panel outputs; "
                f"received {len(panel_results)}."
            )
        panel_outputs = [
            f"Model {model}:\n{response.text}"
            for model, response in panel_results
        ]
        if not panel_outputs:
            raise CrupierProviderUnavailableError("Fusion panel failed; no model outputs available.")

        judge_prompt = build_prompt(
            request,
            extra=(
                "Panel outputs:\n"
                + "\n\n---\n\n".join(panel_outputs)
                + "\n\nReturn a concise structured synthesis with consensus, contradictions, gaps, and risks. "
                "Do not include hidden chain-of-thought."
            ),
        )
        _, judge = self._call_plan_role(
            plan,
            "judge",
            judge_prompt,
            request,
            trace,
            raw_outputs,
            trace_role="judge",
            budget=budget,
        )

        final_prompt = build_prompt(
            request,
            extra=(
                "Judge synthesis:\n"
                + judge.text
                + "\n\nWrite the final answer for the user. Be direct, cite uncertainty, and do not include hidden reasoning."
            ),
        )
        _, final = self._call_plan_role(
            plan,
            "final_writer",
            final_prompt,
            request,
            trace,
            raw_outputs,
            trace_role="final_writer",
            budget=budget,
        )
        return final.text

    def _run_panel_models(
        self,
        request: RequestEnvelope,
        panel_models: list[str],
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        budget: ExecutionBudget,
    ) -> list[tuple[str, AdapterResponse]]:
        if not panel_models:
            return []
        if not self.config.routing.allow_parallel or len(panel_models) == 1:
            return self._run_panel_models_sequential(request, panel_models, trace, raw_outputs, budget)

        max_workers = self._max_parallel_models(request, len(panel_models))
        results: dict[str, AdapterResponse] = {}
        limit_error: CrupierBudgetExceededError | CrupierExecutionLimitError | None = None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._call_model,
                    model,
                    build_prompt(request),
                    request,
                    trace,
                    raw_outputs,
                    role="panel",
                    budget=budget,
                ): model
                for model in panel_models
            }
            for future in as_completed(futures):
                model = futures[future]
                try:
                    results[model] = future.result()
                except (CrupierBudgetExceededError, CrupierExecutionLimitError) as exc:
                    limit_error = limit_error or exc
                except Exception as exc:  # noqa: BLE001
                    with self._state_lock:
                        trace.errors.append({"model": model, "error": str(exc), "phase": "panel"})
        if limit_error is not None:
            raise limit_error
        return [(model, results[model]) for model in panel_models if model in results]

    def _run_panel_models_sequential(
        self,
        request: RequestEnvelope,
        panel_models: list[str],
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        budget: ExecutionBudget,
    ) -> list[tuple[str, AdapterResponse]]:
        outputs: list[tuple[str, AdapterResponse]] = []
        for model in panel_models:
            try:
                response = self._call_model(
                    model,
                    build_prompt(request),
                    request,
                    trace,
                    raw_outputs,
                    role="panel",
                    budget=budget,
                )
                outputs.append((model, response))
            except (CrupierBudgetExceededError, CrupierExecutionLimitError):
                raise
            except Exception as exc:  # noqa: BLE001
                trace.errors.append({"model": model, "error": str(exc), "phase": "panel"})
        return outputs

    def _execute_critique_repair(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        budget: ExecutionBudget,
    ) -> str:
        _, draft = self._call_plan_role(
            plan,
            "generator",
            build_prompt(request),
            request,
            trace,
            raw_outputs,
            trace_role="generator",
            budget=budget,
        )
        critique_prompt = build_prompt(
            request,
            extra=build_critique_instruction(draft=draft.text),
        )
        _, critique = self._call_plan_role(
            plan,
            "critic",
            critique_prompt,
            request,
            trace,
            raw_outputs,
            trace_role="critic",
            budget=budget,
        )
        repair_prompt = build_prompt(
            request,
            extra=build_repair_instruction(draft=draft.text, critique=critique.text),
        )
        _, final = self._call_plan_role(
            plan,
            "repair",
            repair_prompt,
            request,
            trace,
            raw_outputs,
            trace_role="repair",
            budget=budget,
        )
        return extract_final_answer(final.text)

    def _call_plan_role(
        self,
        plan: RoutePlan,
        plan_role: str,
        prompt: str,
        request: RequestEnvelope,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        *,
        trace_role: str,
        budget: ExecutionBudget,
    ) -> tuple[str, AdapterResponse]:
        models = self._role_fallback_models(plan, plan_role)
        if not models:
            raise CrupierProviderUnavailableError(
                f"Route has no executable model for role {plan_role!r}."
            )
        last_error: Exception | None = None
        for index, model in enumerate(models):
            try:
                return model, self._call_model(
                    model,
                    prompt,
                    request,
                    trace,
                    raw_outputs,
                    role=trace_role,
                    budget=budget,
                )
            except (CrupierBudgetExceededError, CrupierExecutionLimitError):
                raise
            except Exception as exc:  # noqa: BLE001 - validated plan models provide bounded role fallback
                last_error = exc
                trace.fallbacks.append(
                    {
                        "phase": "role_fallback",
                        "role": trace_role,
                        "plan_role": plan_role,
                        "model": model,
                        "error": str(exc),
                        "next_model": models[index + 1] if index + 1 < len(models) else None,
                    }
                )
        raise CrupierProviderUnavailableError(
            f"All models for role {plan_role!r} failed. Last error: {last_error}"
        ) from last_error

    def _execute_delegate(
        self,
        request: RequestEnvelope,
        plan: RoutePlan,
        trace: DecisionTrace,
        budget: ExecutionBudget,
    ) -> str:
        step = next((item for item in plan.steps if item.role == "delegate"), None)
        if step is None:
            raise CrupierProviderUnavailableError("Delegate route has no delegate step.")
        remaining_depth = self._remaining_delegate_depth(request)
        if remaining_depth <= 0:
            raise CrupierProviderUnavailableError("Delegate route exceeded max_depth before execution.", retryable=False)

        params = dict(step.params)
        nested_constraints = dict(request.constraints)
        nested_constraints["max_depth"] = remaining_depth - 1
        nested_constraints["max_calls"] = budget.remaining_calls()
        remaining_cost = budget.remaining_cost_usd()
        if remaining_cost is not None:
            nested_constraints["max_cost_usd"] = remaining_cost
        remaining_latency_ms = budget.remaining_latency_ms()
        if remaining_latency_ms is not None:
            nested_constraints["max_latency_ms"] = remaining_latency_ms
        if "constraints" in params and isinstance(params["constraints"], dict):
            nested_constraints.update(params["constraints"])
            nested_constraints["max_depth"] = min(
                self._coerce_non_negative_int(nested_constraints.get("max_depth"), remaining_depth - 1),
                remaining_depth - 1,
            )
        nested_constraints.pop("dry_run", None)

        from .client import Crupier

        nested_client = Crupier(self.config, adapters=self.adapters)
        nested_result = nested_client.deal(
            task=str(params.get("task") or request.task),
            input=params.get("input", request.input),
            mode=params.get("mode", request.mode),
            strategy=str(params.get("strategy") or "orchestrated"),
            constraints=nested_constraints,
            files=params.get("files", request.files),
            messages=params.get("messages", request.messages),
            tools=request.tools if params.get("inherit_tools") is True else params.get("tools"),
            response_schema=params.get("response_schema", request.response_schema),
            metadata={**request.metadata, **dict(params.get("metadata", {}))},
            trace="debug",
            dry_run=False,
        )
        trace.provider_calls.append(
            {
                "role": "delegate",
                "provider": (step.model or "").split(":", 1)[0] if step.model else None,
                "model": step.model,
                "nested_strategy": nested_result.route.strategy if nested_result.route else None,
                "nested_models": nested_result.route.models if nested_result.route else [],
                "max_depth_remaining": remaining_depth - 1,
                "metadata": {"delegated": True},
            }
        )
        if nested_result.trace is not None:
            trace.provider_calls.extend(nested_result.trace.provider_calls)
            trace.fallbacks.extend(nested_result.trace.fallbacks)
            trace.errors.extend(nested_result.trace.errors)
            nested_budget = nested_result.trace.final_quality_signals.get("execution_budget")
            if isinstance(nested_budget, dict):
                budget.absorb(nested_budget)
        return nested_result.output_text

    def _call_model(
        self,
        model_key: str,
        prompt: str,
        request: RequestEnvelope,
        trace: DecisionTrace,
        raw_outputs: list[AdapterResponse],
        *,
        role: str,
        budget: ExecutionBudget,
    ) -> AdapterResponse:
        provider, model = model_key.split(":", 1)
        adapter = self.adapters.get(provider)
        if adapter is None:
            raise CrupierProviderUnavailableError(
                f"No adapter configured for provider {provider!r}. Enable [providers.{provider}] and install the matching extra."
            )
        circuit_error = self._provider_circuit_error(provider)
        if circuit_error is not None:
            trace.errors.append(
                {
                    "phase": "provider_call",
                    "role": role,
                    "provider": provider,
                    "model": model_key,
                    "attempt": 0,
                    "retryable": False,
                    "error_type": "CrupierProviderUnavailableError",
                    "error": str(circuit_error),
                    "circuit_open": True,
                }
            )
            raise circuit_error
        max_retries = self._provider_retry_budget(request)
        backoff_seconds = self._provider_retry_backoff_seconds(request)
        jitter_seconds = self._provider_retry_jitter_seconds(request)
        last_error: Exception | None = None
        effective_request, runtime_policy = apply_runtime_policy(model_key, request, self._cards.get(model_key))
        for attempt in range(1, max_retries + 2):
            try:
                reservation = budget.reserve(model=model_key, prompt=prompt, request=effective_request)
            except (CrupierBudgetExceededError, CrupierExecutionLimitError) as exc:
                with self._state_lock:
                    trace.errors.append(
                        {
                            "phase": "execution_budget",
                            "role": role,
                            "provider": provider,
                            "model": model_key,
                            "attempt": attempt,
                            "retryable": False,
                            "error_type": exc.__class__.__name__,
                            "error": str(exc),
                        }
                    )
                raise
            call_request = request_with_timeout(effective_request, reservation.timeout_seconds)
            call_started = perf_counter()
            response: AdapterResponse | None = None
            try:
                response = adapter.generate(model=model, prompt=prompt, request=call_request)
                if runtime_policy:
                    response.metadata = {**response.metadata, "runtime_policy": runtime_policy}
                if not response.text.strip():
                    raise CrupierProviderUnavailableError(
                        f"Provider {provider!r} model {model_key!r} returned an empty text response."
                    )
            except (CrupierProviderAuthError, CrupierProviderRateLimitError, CrupierProviderUnavailableError) as exc:
                duration_ms = int((perf_counter() - call_started) * 1000)
                retryable = attempt <= max_retries and self._provider_error_retryable(exc)
                last_error = exc
                with self._state_lock:
                    if response is not None:
                        raw_outputs.append(response)
                        trace.provider_calls.append(
                            {
                                "role": role,
                                "provider": provider,
                                "model": model_key,
                                "attempt": attempt,
                                "status": "failed",
                                "latency_ms": duration_ms,
                                "estimated_usd_reserved": reservation.estimated_usd,
                                "usage": response.usage,
                                "metadata": {**response.metadata, "empty_response": True},
                            }
                        )
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
                self._record_provider_failure(provider)
                if not retryable:
                    raise
                if backoff_seconds > 0:
                    sleep(backoff_seconds * (2 ** (attempt - 1)) + random.uniform(0.0, jitter_seconds))
                continue
            assert response is not None
            duration_ms = int((perf_counter() - call_started) * 1000)
            self._record_provider_success(provider)
            call_record = {
                "role": role,
                "provider": provider,
                "model": model_key,
                "attempt": attempt,
                "status": "success",
                "latency_ms": duration_ms,
                "estimated_usd_reserved": reservation.estimated_usd,
                "usage": response.usage,
                "metadata": response.metadata,
            }
            with self._state_lock:
                raw_outputs.append(response)
                trace.provider_calls.append(call_record)
            try:
                budget.ensure_deadline()
            except CrupierExecutionLimitError as exc:
                call_record["discarded_after_deadline"] = True
                with self._state_lock:
                    trace.errors.append(
                        {
                            "phase": "execution_budget",
                            "role": role,
                            "provider": provider,
                            "model": model_key,
                            "attempt": attempt,
                            "retryable": False,
                            "error_type": exc.__class__.__name__,
                            "error": str(exc),
                        }
                    )
                raise
            return response
        raise CrupierProviderUnavailableError(f"Provider call failed after retries. Last error: {last_error}") from last_error

    def _budget_cards(self) -> list[CapabilityCard]:
        try:
            return ModelRegistry(self.config).list(allowed_only=False)
        except Exception:  # noqa: BLE001 - fallback tier pricing still provides a conservative budget
            return []

    def _provider_retry_budget(self, request: RequestEnvelope) -> int:
        value = request.constraints.get("max_provider_retries", self.config.routing.max_provider_retries)
        try:
            return max(0, int(value))  # type: ignore[call-overload]
        except (TypeError, ValueError):
            return max(0, int(self.config.routing.max_provider_retries))

    def _provider_retry_backoff_seconds(self, request: RequestEnvelope) -> float:
        value = request.constraints.get("retry_backoff_seconds", self.config.routing.retry_backoff_seconds)
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return max(0.0, float(self.config.routing.retry_backoff_seconds))

    def _provider_retry_jitter_seconds(self, request: RequestEnvelope) -> float:
        value = request.constraints.get("retry_jitter_seconds", self.config.routing.retry_jitter_seconds)
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return max(0.0, float(self.config.routing.retry_jitter_seconds))

    def _provider_circuit_error(self, provider: str) -> CrupierProviderUnavailableError | None:
        with self._state_lock:
            open_until = self._provider_circuit_open_until.get(provider)
            if open_until is None:
                return None
            remaining = open_until - perf_counter()
            if remaining <= 0:
                self._provider_circuit_open_until.pop(provider, None)
                self._provider_failure_counts.pop(provider, None)
                return None
            return CrupierProviderUnavailableError(
                f"Provider {provider!r} circuit breaker is open for {remaining:.1f}s after repeated failures.",
                retryable=False,
            )

    def provider_circuit_open_reason(self, provider: str) -> str | None:
        error = self._provider_circuit_error(provider)
        return str(error) if error is not None else None

    def _record_provider_failure(self, provider: str) -> None:
        threshold = max(0, int(self.config.routing.circuit_breaker_failure_threshold))
        if threshold <= 0:
            return
        with self._state_lock:
            failures = self._provider_failure_counts.get(provider, 0) + 1
            self._provider_failure_counts[provider] = failures
            if failures >= threshold:
                cooldown = max(0.0, float(self.config.routing.circuit_breaker_cooldown_seconds))
                if cooldown > 0:
                    self._provider_circuit_open_until[provider] = perf_counter() + cooldown

    def _record_provider_success(self, provider: str) -> None:
        with self._state_lock:
            self._provider_failure_counts.pop(provider, None)
            self._provider_circuit_open_until.pop(provider, None)

    def _max_tool_rounds(self, request: RequestEnvelope) -> int:
        value = request.constraints.get("max_tool_rounds", self.config.routing.max_tool_rounds)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return max(1, int(self.config.routing.max_tool_rounds))

    def _max_tool_calls_per_round(self, request: RequestEnvelope) -> int:
        value = request.constraints.get(
            "max_tool_calls_per_round",
            self.config.routing.max_tool_calls_per_round,
        )
        return _positive_int(value, default=self.config.routing.max_tool_calls_per_round)

    def _max_tool_result_chars(self, request: RequestEnvelope) -> int:
        value = request.constraints.get(
            "max_tool_result_chars",
            self.config.routing.max_tool_result_chars,
        )
        return max(256, _positive_int(value, default=self.config.routing.max_tool_result_chars))

    def _remaining_delegate_depth(self, request: RequestEnvelope) -> int:
        return self._coerce_non_negative_int(request.constraints.get("max_depth"), self.config.routing.max_depth)

    @staticmethod
    def _coerce_non_negative_int(value: object, default: int) -> int:
        try:
            return max(0, int(value))  # type: ignore[call-overload]
        except (TypeError, ValueError):
            return max(0, int(default))

    @staticmethod
    def _max_parallel_models(request: RequestEnvelope, model_count: int) -> int:
        value = request.constraints.get("max_parallel_models", model_count)
        try:
            requested = int(value)
        except (TypeError, ValueError):
            requested = model_count
        return max(1, min(model_count, requested))

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

    def _usage_estimated_cost(self, calls: list[dict[str, object]]) -> float | None:
        try:
            cards = ModelRegistry(self.config).list(allowed_only=False)
        except Exception:  # noqa: BLE001 - reserved estimates remain available without registry cards
            return None
        return usage_estimated_cost_from_calls(calls, cards)

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
    def _cascade_models(plan: RoutePlan) -> list[str]:
        models: list[str] = []
        for role in ("primary", "escalation", "fallback"):
            for step in plan.steps:
                if step.role != role:
                    continue
                if step.model and step.model not in models:
                    models.append(step.model)
                for model in step.models:
                    if model not in models:
                        models.append(model)
        if not models:
            return RouteExecutor._models_in_execution_order(plan)
        return models

    @staticmethod
    def _model_for_role(plan: RoutePlan, role: str) -> str | None:
        for step in plan.steps:
            if step.role == role:
                return step.model or (step.models[0] if step.models else None)
        return None

    @staticmethod
    def _role_fallback_models(plan: RoutePlan, role: str) -> list[str]:
        preferred: list[str] = []
        step = next((item for item in plan.steps if item.role == role), None)
        if step is not None:
            for model in [step.model, *step.models]:
                if model and model not in preferred:
                    preferred.append(model)
        alternatives = [model for model in plan.models if model not in preferred]
        if preferred:
            provider = preferred[0].split(":", 1)[0]
            alternatives.sort(key=lambda model: model.split(":", 1)[0] == provider)
        return preferred + alternatives
