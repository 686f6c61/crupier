"""Route orchestration contracts and deterministic baseline."""

from __future__ import annotations

import json
from typing import Any, Protocol

from .adapters import ProviderAdapter
from .config import CrupierConfig, ProfileSettings
from .costs import estimate_route_cost
from .errors import CrupierRouteValidationError
from .models import (
    CapabilityCard,
    CostEstimate,
    ModelRef,
    PlanningContext,
    RequestEnvelope,
    RoutePlan,
    RouteStep,
)
from .policy import PolicyEngine, PolicyResult
from .prompts import (
    ORCHESTRATOR_ROUTE_PLAN_PROMPT_VERSION,
    build_orchestrator_planning_prompt,
    build_orchestrator_repair_prompt,
)
from .route_schema import ALLOWED_STRATEGIES, validate_route_plan_shape
from .selector import ModelSelector


class Orchestrator(Protocol):
    """RoutePlan producer.

    Implementations may be deterministic, model-powered, hybrid, or locked,
    but they must only plan with policy-filtered candidates.
    """

    def plan(self, context: PlanningContext) -> RoutePlan:
        """Return a candidate RoutePlan for a validated PlanningContext."""
        ...


class DeterministicOrchestrator:
    """Deterministic explainable baseline used before model-powered planning."""

    def __init__(self, config: CrupierConfig, *, selector: ModelSelector | None = None):
        self.config = config
        self.selector = selector or ModelSelector(config)

    def plan(self, context: PlanningContext) -> RoutePlan:
        request = context.request
        candidates = context.candidates
        if request.constraints.get("force_model"):
            plan = self._forced_model(request, candidates)
            return self._finalize_plan(plan, context)

        strategy = self._strategy_for(request)
        if strategy == "orchestrated":
            strategy = self._orchestrate_strategy(request, candidates)

        if strategy == "single":
            plan = self._single(request, candidates)
        elif strategy == "fallback":
            plan = self._fallback(request, candidates)
        elif strategy == "cascade":
            plan = self._cascade(request, candidates)
        elif strategy == "panel":
            plan = self._panel(request, candidates)
        elif strategy == "fusion":
            plan = self._fusion(request, candidates)
        elif strategy == "critique_repair":
            plan = self._critique_repair(request, candidates)
        elif strategy == "local_first":
            plan = self._local_first(request, candidates)
        elif strategy == "delegate":
            plan = self._delegate(request, candidates)
        else:
            plan = self._single(request, candidates)
            plan.reason += f" Requested strategy {strategy!r} is unavailable for deterministic routing; fell back to single."
        return self._finalize_plan(plan, context)

    def _finalize_plan(self, plan: RoutePlan, context: PlanningContext) -> RoutePlan:
        plan.policy_filters_applied = list(context.filters_applied)
        # The orchestrator may suggest a cost, but Crupier owns budget math.
        plan.estimated_cost = estimate_route_cost(plan, context.request, context.candidates)
        self._attach_input_plan(plan, context.request)
        self._attach_selection_scores(plan, context)
        return plan

    def _forced_model(self, request: RequestEnvelope, candidates: list[CapabilityCard]) -> RoutePlan:
        forced = ModelRef.parse(str(request.constraints["force_model"])).key
        for card in candidates:
            if card.model_ref.key == forced:
                return RoutePlan(
                    strategy="single",
                    steps=[
                        RouteStep(role="primary", model=card.model_ref.key, timeout_ms=self.config.routing.max_latency_ms)
                    ],
                    estimated_cost=CostEstimate(0.0),
                    estimated_latency_ms=self._latency_estimate([card]),
                    reason=f"Forced model {forced!r} requested by caller.",
                    risk_level=self._risk_level(request, "single"),
                    summary=f"Forced single model route using {card.model_ref.key}.",
                )
        raise CrupierRouteValidationError(
            f"Forced model {forced!r} is not allowed by the current project policy/model allowlist."
        )

    def _strategy_for(self, request: RequestEnvelope) -> str:
        if request.strategy:
            return request.strategy
        mode = request.mode or self.config.project.default_profile
        profile = self.config.profiles.get(mode)
        if profile:
            return profile.strategy
        return self.config.routing.default_strategy

    def _orchestrate_strategy(self, request: RequestEnvelope, candidates: list[CapabilityCard]) -> str:
        mode = request.mode or self.config.project.default_profile
        profile = self.config.profiles.get(mode)
        strategy = self._strategy_from_rules(request, candidates, profile)
        if strategy:
            return strategy
        risk = request.constraints.get("risk_level")
        if mode == "private":
            return "local_first"
        if mode == "research":
            return "fusion" if self.config.routing.allow_fusion and len(candidates) >= 2 else "single"
        if mode == "structured":
            return "cascade" if len(candidates) >= 2 else "single"
        if mode == "cheap":
            return "cascade" if len(candidates) >= 2 else "single"
        if mode == "fast":
            return "single"
        if mode == "quality":
            return "single"
        if mode == "agentic" and (request.tools or risk == "high"):
            return "critique_repair" if len(candidates) >= 2 else "single"
        return "single"

    def _strategy_from_rules(
        self,
        request: RequestEnvelope,
        candidates: list[CapabilityCard],
        profile: ProfileSettings | None,
    ) -> str | None:
        if profile is None:
            return None
        rules = profile.options.get("strategy_rules")
        if not isinstance(rules, list):
            return None
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            strategy = str(rule.get("strategy", ""))
            if strategy not in ALLOWED_STRATEGIES:
                continue
            when = rule.get("when", {})
            if isinstance(when, dict) and self._strategy_rule_matches(request, candidates, when):
                return strategy
        return None

    def _strategy_rule_matches(
        self,
        request: RequestEnvelope,
        candidates: list[CapabilityCard],
        when: dict[str, Any],
    ) -> bool:
        tool_count = len(request.tools)
        checks = {
            "tools": bool(request.tools),
            "structured": request.response_schema is not None or bool(request.constraints.get("response_schema")),
        }
        for key, expected in checks.items():
            if key in when and bool(when[key]) != expected:
                return False
        if "risk_level" in when and str(when["risk_level"]) != str(request.constraints.get("risk_level")):
            return False
        if "min_tools" in when and tool_count < int(when["min_tools"]):
            return False
        if "max_tools" in when and tool_count > int(when["max_tools"]):
            return False
        if "min_candidates" in when and len(candidates) < int(when["min_candidates"]):
            return False
        if "max_candidates" in when and len(candidates) > int(when["max_candidates"]):
            return False
        if "file_kind" in when:
            file_kinds = {asset.kind for asset in request.files}
            if str(when["file_kind"]) not in file_kinds:
                return False
        return True

    def _single(self, request: RequestEnvelope, candidates: list[CapabilityCard]) -> RoutePlan:
        card = self._rank(request, candidates)[0]
        return RoutePlan(
            strategy="single",
            steps=[RouteStep(role="primary", model=card.model_ref.key, timeout_ms=self.config.routing.max_latency_ms)],
            estimated_cost=CostEstimate(0.0),
            estimated_latency_ms=self._latency_estimate([card]),
            reason=f"Selected best single candidate for mode {request.mode or self.config.project.default_profile!r}.",
            risk_level=self._risk_level(request, "single"),
            summary=f"Single model route using {card.model_ref.key}.",
        )

    def _fallback(self, request: RequestEnvelope, candidates: list[CapabilityCard]) -> RoutePlan:
        ranked = candidates[:3]
        return RoutePlan(
            strategy="fallback",
            steps=[RouteStep(role="fallback", models=[card.model_ref.key for card in ranked])],
            estimated_cost=CostEstimate(0.0),
            estimated_latency_ms=self._latency_estimate(ranked[:1]),
            reason="Planned ordered fallback for availability and rate-limit resilience.",
            risk_level=self._risk_level(request, "fallback"),
            summary="Fallback route: " + " -> ".join(card.model_ref.key for card in ranked),
        )

    def _cascade(self, request: RequestEnvelope, candidates: list[CapabilityCard]) -> RoutePlan:
        cheap_first = sorted(self._rank(request, candidates), key=lambda card: self._cost_sort(card))
        first = cheap_first[0]
        best = self._rank(request, candidates)[0]
        steps = [RouteStep(role="primary", model=first.model_ref.key)]
        if best.model_ref.key != first.model_ref.key:
            steps.append(RouteStep(role="escalation", model=best.model_ref.key))
        return RoutePlan(
            strategy="cascade",
            steps=steps,
            estimated_cost=CostEstimate(0.0),
            estimated_latency_ms=self._latency_estimate([first]),
            reason="Start with a lower-cost candidate and escalate if validation or confidence fails.",
            risk_level=self._risk_level(request, "cascade"),
            summary="Cascade route: " + " -> ".join(step.model or "" for step in steps),
        )

    def _panel(self, request: RequestEnvelope, candidates: list[CapabilityCard]) -> RoutePlan:
        panel = self._rank(request, candidates)[: self._panel_size(request, candidates)]
        return RoutePlan(
            strategy="panel",
            steps=[RouteStep(role="panel", models=[card.model_ref.key for card in panel])],
            estimated_cost=CostEstimate(0.0),
            estimated_latency_ms=self._latency_estimate(panel),
            reason="Multiple independent model outputs requested without synthesis.",
            risk_level=self._risk_level(request, "panel"),
            summary="Panel route with " + ", ".join(card.model_ref.key for card in panel),
        )

    def _fusion(self, request: RequestEnvelope, candidates: list[CapabilityCard]) -> RoutePlan:
        if len(candidates) < 2:
            return self._single(request, candidates)
        ranked = self._rank(request, candidates)
        panel = ranked[: self._panel_size(request, candidates)]
        judge = self._prefer_different_provider(panel[0], ranked[1:]) or panel[0]
        writer = ranked[0]
        return RoutePlan(
            strategy="fusion",
            steps=[
                RouteStep(role="panel", models=[card.model_ref.key for card in panel]),
                RouteStep(role="judge", model=judge.model_ref.key),
                RouteStep(role="final_writer", model=writer.model_ref.key),
            ],
            estimated_cost=CostEstimate(0.0),
            estimated_latency_ms=self._latency_estimate(panel) + self._latency_estimate([judge, writer]),
            reason="Research/high-uncertainty route benefits from independent perspectives plus synthesis.",
            risk_level=self._risk_level(request, "fusion"),
            summary="Fusion route with panel, judge, and final writer.",
        )

    def _critique_repair(self, request: RequestEnvelope, candidates: list[CapabilityCard]) -> RoutePlan:
        ranked = self._rank(request, candidates)
        generator = ranked[0]
        critic = self._prefer_different_provider(generator, ranked[1:]) or generator
        repair = generator
        return RoutePlan(
            strategy="critique_repair",
            steps=[
                RouteStep(role="generator", model=generator.model_ref.key),
                RouteStep(role="critic", model=critic.model_ref.key),
                RouteStep(role="repair", model=repair.model_ref.key),
            ],
            estimated_cost=CostEstimate(0.0),
            estimated_latency_ms=self._latency_estimate([generator, critic, repair]),
            reason="Agentic/tool-heavy request benefits from a separate critique before final output.",
            risk_level=self._risk_level(request, "critique_repair"),
            summary=f"Critique-repair route using generator {generator.model_ref.key} and critic {critic.model_ref.key}.",
        )

    def _local_first(self, request: RequestEnvelope, candidates: list[CapabilityCard]) -> RoutePlan:
        ranked = self._rank(request, candidates)
        local = [card for card in ranked if card.model_ref.provider == "ollama"]
        first = local[0] if local else ranked[0]
        fallback = next((card for card in ranked if card.model_ref.key != first.model_ref.key), None)
        steps = [RouteStep(role="primary", model=first.model_ref.key)]
        if fallback:
            steps.append(RouteStep(role="fallback", model=fallback.model_ref.key))
        return RoutePlan(
            strategy="local_first",
            steps=steps,
            estimated_cost=CostEstimate(0.0),
            estimated_latency_ms=self._latency_estimate([first]),
            reason="Private/local-first profile prefers configured Ollama candidates before other providers.",
            risk_level=self._risk_level(request, "local_first"),
            summary="Local-first route: " + " -> ".join(step.model or "" for step in steps),
        )

    def _delegate(self, request: RequestEnvelope, candidates: list[CapabilityCard]) -> RoutePlan:
        anchor = self._rank(request, candidates)[0]
        max_depth = self._max_depth(request)
        return RoutePlan(
            strategy="delegate",
            steps=[
                RouteStep(
                    role="delegate",
                    model=anchor.model_ref.key,
                    timeout_ms=self.config.routing.max_latency_ms,
                    params={
                        "task": request.task,
                        "mode": request.mode or self.config.project.default_profile,
                        "strategy": "orchestrated",
                        "max_depth": max_depth,
                    },
                )
            ],
            estimated_cost=CostEstimate(0.0),
            estimated_latency_ms=self._latency_estimate([anchor]),
            reason="Delegated workflow requested; nested route will plan with inherited context and reduced depth.",
            risk_level=self._risk_level(request, "delegate"),
            summary=f"Delegate route anchored on {anchor.model_ref.key}.",
        )

    def _rank(self, request: RequestEnvelope, candidates: list[CapabilityCard]) -> list[CapabilityCard]:
        return self.selector.rank(request, candidates)

    def _attach_selection_scores(self, plan: RoutePlan, context: PlanningContext) -> None:
        if context.deterministic_scores:
            plan.selection_scores = list(context.deterministic_scores)
            return
        limit = int(context.request.constraints.get("selection_trace_limit", 5))
        plan.selection_scores = [
            score.to_dict() for score in self.selector.score_all(context.request, context.candidates)[:limit]
        ]

    @staticmethod
    def _attach_input_plan(plan: RoutePlan, request: RequestEnvelope) -> None:
        if request.file_plan is None:
            return
        plan.input_plan = {"files": request.file_plan.to_dict()}
        representations = ", ".join(
            f"{item.kind}->{item.representation}" for item in request.file_plan.representations
        )
        if representations:
            plan.summary = (plan.summary + f" Input plan: {representations}.").strip()

    @staticmethod
    def _prefer_different_provider(reference: CapabilityCard, candidates: list[CapabilityCard]) -> CapabilityCard | None:
        for card in candidates:
            if card.model_ref.provider != reference.model_ref.provider:
                return card
        return candidates[0] if candidates else None

    @staticmethod
    def _cost_sort(card: CapabilityCard) -> tuple[int, str]:
        order = {"low": 0, "medium": 1, "unknown": 2, "high": 3}
        return (order.get(card.cost_tier, 2), card.model_ref.key)

    @staticmethod
    def _latency_estimate(cards: list[CapabilityCard]) -> int:
        if not cards:
            return 0
        tiers = {"fast": 2500, "medium": 6000, "slow": 12000, "unknown": 8000}
        return max(tiers.get(card.latency_tier, 8000) for card in cards)

    @staticmethod
    def _panel_size(request: RequestEnvelope, candidates: list[CapabilityCard]) -> int:
        requested = int(request.constraints.get("max_panel_size", 3))
        return max(1, min(requested, len(candidates)))

    @staticmethod
    def _risk_level(request: RequestEnvelope, strategy: str) -> str:
        if "risk_level" in request.constraints:
            return str(request.constraints["risk_level"])
        if strategy in {"fusion", "critique_repair", "delegate"} or request.tools:
            return "high"
        if request.mode in {"cheap", "fast"}:
            return "low"
        return "medium"

    def _max_depth(self, request: RequestEnvelope) -> int:
        value = request.constraints.get("max_depth", self.config.routing.max_depth)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return max(0, int(self.config.routing.max_depth))


class ModelOrchestrator:
    """Model-powered planner with deterministic fallback."""

    ALLOWED_STRATEGIES = ALLOWED_STRATEGIES

    def __init__(
        self,
        config: CrupierConfig,
        *,
        adapters: dict[str, ProviderAdapter],
        fallback: DeterministicOrchestrator | None = None,
        selector: ModelSelector | None = None,
    ):
        self.config = config
        self.adapters = adapters
        self.fallback = fallback or DeterministicOrchestrator(config, selector=selector)
        self.policy = PolicyEngine(config)

    def plan(self, context: PlanningContext) -> RoutePlan:
        model_key = self.config.orchestrator.model
        if not model_key:
            return self._deterministic_fallback(context, "no orchestrator model is configured")

        attempted_models: list[str] = []
        last_error = ""
        for candidate_model in self._orchestrator_models():
            attempted_models.append(candidate_model)
            try:
                plan = self._plan_with_model(candidate_model, context)
                if candidate_model != model_key:
                    plan.reason = (
                        plan.reason
                        + f" Fallback orchestrator {candidate_model} was used after primary orchestrator failure."
                    ).strip()
                return plan
            except Exception as exc:  # noqa: BLE001 - invalid model plans fall back safely
                last_error = str(exc)

        attempted = " -> ".join(attempted_models)
        reason = last_error or "model plan was invalid"
        if attempted:
            reason = f"{reason}; attempted orchestrators: {attempted}"
        return self._deterministic_fallback(context, reason)

    def _plan_with_model(self, model_key: str, context: PlanningContext) -> RoutePlan:
        max_repairs = max(0, int(self.config.orchestrator.max_repairs))
        raw_text = ""
        last_error = ""
        raw_text = self._call_orchestrator(model_key, self._planning_prompt(context), context)
        for attempt in range(max_repairs + 1):
            try:
                plan = self._plan_from_text(raw_text)
                self._validate_model_plan(plan, context)
                plan = self.fallback._finalize_plan(plan, context)
                plan.reason = (plan.reason + " Model orchestrator proposed and validated this route.").strip()
                return plan
            except Exception as exc:  # noqa: BLE001 - invalid model plans are repaired or escalated
                last_error = str(exc)
                if attempt >= max_repairs:
                    break
                raw_text = self._call_orchestrator(
                    model_key,
                    self._repair_prompt(context, raw_text=raw_text, error=last_error),
                    context,
                )
        raise CrupierRouteValidationError(last_error or "model plan was invalid")

    def _orchestrator_models(self) -> list[str]:
        models: list[str] = []
        for value in (self.config.orchestrator.model, self.config.orchestrator.fallback_model):
            if not value:
                continue
            key = ModelRef.parse(str(value)).key
            if key not in models:
                models.append(key)
        return models

    def _deterministic_fallback(self, context: PlanningContext, reason: str) -> RoutePlan:
        plan = self.fallback.plan(context)
        plan.reason = (
            plan.reason + f" Model orchestrator unavailable or invalid; deterministic fallback used ({reason})."
        ).strip()
        return plan

    def _call_orchestrator(self, model_key: str, prompt: str, context: PlanningContext) -> str:
        model_ref = ModelRef.parse(model_key)
        adapter = self.adapters.get(model_ref.provider)
        if adapter is None:
            raise CrupierRouteValidationError(
                f"No adapter is configured for orchestrator provider {model_ref.provider!r}."
            )
        request = RequestEnvelope(
            task="Produce a validated Crupier RoutePlan JSON object.",
            input=context.to_dict(summary=True),
            mode="structured",
            constraints={
                "temperature": self.config.orchestrator.temperature,
                "max_output_tokens": 1200,
                "timeout_seconds": 60,
            },
            metadata={"purpose": "crupier_orchestrator"},
        )
        response = adapter.generate(model=model_ref.model, prompt=prompt, request=request)
        return response.text

    def _planning_prompt(self, context: PlanningContext) -> str:
        return build_orchestrator_planning_prompt(self._planning_payload(context))

    def _repair_prompt(self, context: PlanningContext, *, raw_text: str, error: str) -> str:
        return build_orchestrator_repair_prompt(self._planning_payload(context), raw_text=raw_text, error=error)

    def _planning_payload(self, context: PlanningContext) -> dict[str, Any]:
        payload = context.to_dict(summary=True)
        payload["candidate_cards"] = [_candidate_summary(card) for card in context.candidates]
        payload["max_calls"] = int(context.request.constraints.get("max_calls", self.config.routing.max_calls))
        payload["allowed_strategies"] = sorted(self.ALLOWED_STRATEGIES)
        payload["prompt_version"] = ORCHESTRATOR_ROUTE_PLAN_PROMPT_VERSION
        sensitive = self._sensitive_allowed_strategies(context)
        if sensitive:
            payload["strategy_guardrail"] = {
                "reason": "agentic high-risk/tool request",
                "allowed_strategies": sorted(sensitive),
            }
        return payload

    def _plan_from_text(self, text: str) -> RoutePlan:
        data = _extract_json_object(text)
        if isinstance(data.get("route_plan"), dict):
            data = data["route_plan"]
        data = _normalize_common_route_shape(data)
        return RoutePlan.from_dict(data)

    def _validate_model_plan(self, plan: RoutePlan, context: PlanningContext) -> None:
        max_calls = int(context.request.constraints.get("max_calls", self.config.routing.max_calls))
        validate_route_plan_shape(plan, max_calls=max_calls)
        expected_strategy = self._profile_strategy(context)
        if expected_strategy and plan.strategy != expected_strategy:
            # Profile strategies are project policy, not model suggestions.
            raise CrupierRouteValidationError(
                f"Route strategy {plan.strategy!r} does not match required profile strategy {expected_strategy!r}."
            )
        allowed_sensitive_strategies = self._sensitive_allowed_strategies(context)
        if allowed_sensitive_strategies and plan.strategy not in allowed_sensitive_strategies:
            raise CrupierRouteValidationError(
                f"Route strategy {plan.strategy!r} is not allowed for this sensitive request; "
                f"expected one of {sorted(allowed_sensitive_strategies)!r}."
            )
        policy_result = PolicyResult(
            allowed=list(context.candidates),
            filters_applied=list(context.filters_applied),
        )
        self.policy.validate_route(plan, policy_result, context.request)

    def _profile_strategy(self, context: PlanningContext) -> str | None:
        if context.request.strategy:
            return context.request.strategy
        mode = context.request.mode or self.config.project.default_profile
        profile = self.config.profiles.get(mode)
        if profile and profile.strategy != "orchestrated":
            return profile.strategy
        return None

    @staticmethod
    def _sensitive_allowed_strategies(context: PlanningContext) -> set[str] | None:
        request = context.request
        if request.mode != "agentic":
            return None
        if request.tools or request.constraints.get("risk_level") == "high":
            return {"critique_repair", "single"}
        return None


def _candidate_summary(card: CapabilityCard) -> dict[str, Any]:
    best_skills = {
        key: value
        for key, value in sorted(
            card.skill_scores.items(),
            key=lambda item: (float(item[1]) if isinstance(item[1], int | float) else 0.0, item[0]),
            reverse=True,
        )[:8]
        if isinstance(value, int | float)
    }
    return {
        "model": card.model_ref.key,
        "provider": card.model_ref.provider,
        "stability": card.model_ref.stability,
        "model_kind": card.model_kind,
        "modalities_input": card.modalities_input,
        "modalities_output": card.modalities_output,
        "supports_embeddings": card.supports_embeddings,
        "supports_tools": card.supports_tools,
        "supports_structured_output": card.supports_structured_output,
        "supports_streaming": card.supports_streaming,
        "supports_file_input": card.supports_file_input,
        "cost_tier": card.cost_tier,
        "latency_tier": card.latency_tier,
        "quality_tier": card.quality_tier,
        "routing_status": card.routing_hints.get("routing_status"),
        "lifecycle": card.routing_hints.get("lifecycle"),
        "production_default": card.routing_hints.get("production_default"),
        "requires_opt_in": card.routing_hints.get("requires_opt_in"),
        "natural_summary": card.natural_profile.get("summary"),
        "status_reason": card.natural_profile.get("status_reason"),
        "best_skills": best_skills,
        "routing_hints": {
            "preferred_when": card.routing_hints.get("preferred_when", []),
            "avoid_when": card.routing_hints.get("avoid_when", []),
            "strategy_bias": card.routing_hints.get("strategy_bias", []),
        },
        "strengths": card.strengths,
        "local_eval_scores": card.local_eval_scores,
        "capability_status": card.capability_status,
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise CrupierRouteValidationError("Orchestrator response did not contain a JSON object.")
        try:
            data = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise CrupierRouteValidationError(f"Orchestrator returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CrupierRouteValidationError("Orchestrator response must be a JSON object.")
    return data


def _normalize_common_route_shape(data: dict[str, Any]) -> dict[str, Any]:
    """Repair common role/strategy naming slips without changing model choices."""

    strategy = data.get("strategy")
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        return data
    normalized_steps: list[dict[str, Any]] = []
    changed = False
    for step in steps:
        if not isinstance(step, dict):
            normalized_steps.append(step)
            continue
        role = step.get("role")
        models = list(step.get("models", []))
        model = step.get("model")
        if strategy == "cascade" and role == "fallback" and models:
            normalized_steps.append({"role": "primary", "model": models[0]})
            normalized_steps.extend({"role": "escalation", "model": item} for item in models[1:])
            changed = True
            continue
        if strategy == "cascade" and role == "fallback" and model:
            normalized_steps.append({"role": "primary", "model": model})
            changed = True
            continue
        if strategy == "single" and role == "fallback" and models:
            normalized_steps.append({"role": "primary", "model": models[0]})
            changed = True
            continue
        if strategy == "single" and role == "fallback" and model:
            normalized_steps.append({"role": "primary", "model": model})
            changed = True
            continue
        if strategy == "fallback" and role == "primary" and models:
            normalized_steps.append({"role": "fallback", "models": models})
            changed = True
            continue
        if strategy == "cascade" and role == "escalation" and models:
            normalized_steps.extend({"role": "escalation", "model": item} for item in models)
            changed = True
            continue
        if strategy in {"single", "cascade", "local_first"} and role == "primary" and models and not model:
            normalized_steps.append({"role": "primary", "model": models[0]})
            changed = True
            continue
        normalized_steps.append(step)
    if strategy == "cascade":
        seen_primary = False
        cascade_steps: list[dict[str, Any]] = []
        for step in normalized_steps:
            if not isinstance(step, dict):
                cascade_steps.append(step)
                continue
            if step.get("role") == "primary":
                if seen_primary:
                    step = {**step, "role": "escalation"}
                    changed = True
                seen_primary = True
            cascade_steps.append(step)
        normalized_steps = cascade_steps
    if changed:
        data = dict(data)
        data["steps"] = normalized_steps
        reason = str(data.get("reason", ""))
        note = " Crupier normalized common route role shape before validation."
        data["reason"] = (reason + note).strip()
    return data
