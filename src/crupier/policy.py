"""Policy filtering and route validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .capabilities import capability_evidence, capability_reason
from .config import CrupierConfig, PolicyRule
from .constraints import request_allows_parallel
from .errors import (
    CrupierBudgetExceededError,
    CrupierConfigError,
    CrupierPolicyError,
    CrupierRouteValidationError,
)
from .models import CapabilityCard, ModelRef, RequestEnvelope, RoutePlan
from .route_schema import planned_call_count, validate_route_plan_shape


@dataclass(slots=True)
class Exclusion:
    model: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"model": self.model, "reason": self.reason}


@dataclass(slots=True)
class PolicyResult:
    allowed: list[CapabilityCard] = field(default_factory=list)
    excluded: list[Exclusion] = field(default_factory=list)
    filters_applied: list[str] = field(default_factory=list)

    def excluded_dicts(self) -> list[dict[str, str]]:
        return [item.to_dict() for item in self.excluded]


class PolicyEngine:
    """Applies hard constraints before and after route planning."""

    def __init__(self, config: CrupierConfig):
        self.config = config

    def filter_candidates(self, request: RequestEnvelope, candidates: list[CapabilityCard]) -> PolicyResult:
        self._require_well_formed_policy()
        result = PolicyResult()
        deny = {ModelRef.parse(model).key for model in self.config.models.deny}
        constraints = request.constraints
        require_zdr = bool(constraints.get("require_zdr", False))
        allow_latest = bool(constraints.get("allow_latest_aliases", self.config.routing.allow_latest_aliases))
        allow_preview = bool(constraints.get("allow_preview_models", self.config.routing.allow_preview_models))
        allow_deprecated = bool(constraints.get("allow_deprecated_models", False))
        require_verified_capabilities = bool(constraints.get("require_verified_capabilities", False))
        has_tools = bool(request.tools) or bool(constraints.get("requires_tools"))
        wants_structured = request.response_schema is not None or bool(constraints.get("response_schema"))
        wants_streaming = bool(constraints.get("stream", False) or constraints.get("require_streaming", False))
        requested_model_kind = str(
            constraints.get("model_kind") or ("embedding" if request.mode == "embedding" else "chat")
        )
        offline_planning = bool(request.metadata.get("_crupier_offline_planning"))
        allowed_models = _model_keys(constraints.get("allowed_models"))
        allowed_providers = _string_set(constraints.get("allowed_providers"))

        for card in candidates:
            key = card.model_ref.key
            provider = card.model_ref.provider
            provider_settings = self.config.providers.get(provider)

            if allowed_models and key not in allowed_models:
                self._exclude(
                    result,
                    key,
                    "model is outside request allowed_models",
                    "request_allowed_models",
                )
                continue
            if allowed_providers and provider not in allowed_providers:
                self._exclude(
                    result,
                    key,
                    "provider is outside request allowed_providers",
                    "request_allowed_providers",
                )
                continue
            if card.model_kind != requested_model_kind:
                self._exclude(
                    result,
                    key,
                    f"model kind {card.model_kind!r} cannot execute a {requested_model_kind!r} request",
                    "model_kind",
                )
                continue
            if key in deny:
                self._exclude(result, key, "model is explicitly denied", "deny_list")
                continue
            if provider_settings is None and not offline_planning:
                self._exclude(result, key, f"provider {provider!r} is not configured", "provider_configured")
                continue
            if provider == "openrouter" and (provider_settings is None or not provider_settings.enabled):
                self._exclude(result, key, "OpenRouter is optional BYOK and not enabled", "openrouter_byok")
                continue
            if provider_settings is not None and not provider_settings.enabled:
                self._exclude(result, key, f"provider {provider!r} is disabled", "provider_enabled")
                continue
            lifecycle = card.routing_hints.get("lifecycle") or card.natural_profile.get("lifecycle")
            routing_status = card.routing_hints.get("routing_status") or card.natural_profile.get("routing_status")
            if (card.deprecation or lifecycle in {"deprecated", "shutdown"} or routing_status in {"deprecated", "shutdown"}) and not allow_deprecated:
                self._exclude(result, key, "model is deprecated or shut down", "deprecated_models")
                continue
            if card.model_ref.stability == "latest" and not allow_latest:
                self._exclude(result, key, "latest aliases are disabled", "stable_models_only")
                continue
            if card.model_ref.stability in {"preview", "experimental"} and not allow_preview:
                self._exclude(result, key, f"{card.model_ref.stability} models are disabled", "stable_models_only")
                continue
            if require_zdr and card.zdr_eligible is not True:
                self._exclude(result, key, "ZDR is required and card is not marked ZDR eligible", "require_zdr")
                continue
            if has_tools:
                evidence = capability_evidence(card, "tool_call", declared=card.supports_tools)
                if not evidence.supported or (require_verified_capabilities and evidence.status != "verified"):
                    self._exclude(result, key, f"request has tools but {capability_reason(evidence)}", "tool_support")
                    continue
            if wants_structured:
                evidence = capability_evidence(
                    card,
                    "structured_output",
                    declared=card.supports_structured_output,
                )
                if not evidence.supported or (require_verified_capabilities and evidence.status != "verified"):
                    self._exclude(
                        result,
                        key,
                        f"structured output is required but {capability_reason(evidence)}",
                        "structured_output",
                    )
                    continue
            if wants_streaming:
                evidence = capability_evidence(card, "streaming", declared=card.supports_streaming)
                if not evidence.supported or (require_verified_capabilities and evidence.status != "verified"):
                    self._exclude(result, key, f"streaming is required but {capability_reason(evidence)}", "streaming")
                    continue
            if request.file_plan is not None:
                file_reason = self._file_input_rejection_reason(card, request, require_verified_capabilities)
                if file_reason:
                    self._exclude(result, key, file_reason, "file_input")
                    continue
            declarative_reason = self._declarative_rule_rejection_reason(card, request)
            if declarative_reason:
                rule_name, reason = declarative_reason
                self._exclude(result, key, reason, f"policy_rule:{rule_name}")
                continue
            result.allowed.append(card)

        if not result.allowed:
            reasons = "; ".join(f"{item.model}: {item.reason}" for item in result.excluded)
            raise CrupierPolicyError(f"No models remain after policy filtering. {reasons}")
        return result

    def _require_well_formed_policy(self) -> None:
        rules = self.config.policy.rules
        if not isinstance(rules, list):
            raise CrupierConfigError("[policy].rules must be an array of rule objects.")
        for index, rule in enumerate(rules):
            if not isinstance(rule, PolicyRule):
                raise CrupierConfigError(
                    f"[policy].rules[{index}] must be a table/object, not {type(rule).__name__}."
                )

    def validate_route(self, plan: RoutePlan, policy_result: PolicyResult, request: RequestEnvelope) -> None:
        max_calls = int(request.constraints.get("max_calls", self.config.routing.max_calls))
        validate_route_plan_shape(plan, max_calls=max_calls)

        allowed = {card.model_ref.key for card in policy_result.allowed}
        for model in plan.models:
            if model not in allowed:
                raise CrupierRouteValidationError(f"Route uses {model!r}, but it is not policy-allowed.")

        max_cost = request.constraints.get("max_cost_usd", self.config.routing.max_cost_per_request_usd)
        if max_cost is not None and plan.estimated_cost.estimated_usd > float(max_cost):
            raise CrupierBudgetExceededError(
                f"Route estimated cost ${plan.estimated_cost.estimated_usd:.4f} exceeds max ${float(max_cost):.4f}."
            )

        if plan.strategy == "fusion" and not self.config.routing.allow_fusion:
            raise CrupierRouteValidationError("Route uses fusion, but fusion is disabled by routing policy.")

        if (
            not request_allows_parallel(self.config, request)
            and plan.strategy in {"fusion", "panel"}
            and "sequential_panel_execution" not in plan.policy_filters_applied
        ):
            plan.policy_filters_applied.append("sequential_panel_execution")

        if plan.strategy in {"fusion", "panel"}:
            self._validate_panel_size_constraints(plan, request)

        planned_calls = planned_call_count(plan)
        if planned_calls > max_calls:
            raise CrupierRouteValidationError(f"Route plans {planned_calls} calls, above max_calls={max_calls}.")

    @staticmethod
    def _validate_panel_size_constraints(plan: RoutePlan, request: RequestEnvelope) -> None:
        panel = next((step for step in plan.steps if step.role == "panel"), None)
        if panel is None:
            return
        count = len(panel.models) if panel.models else int(panel.model is not None)
        try:
            minimum = max(2, int(request.constraints.get("min_panel_size", 2)))
            raw_maximum = request.constraints.get("max_panel_size")
            maximum = int(raw_maximum) if raw_maximum is not None else None
        except (TypeError, ValueError) as exc:
            raise CrupierRouteValidationError(
                "min_panel_size and max_panel_size must be integers."
            ) from exc
        if maximum is not None and maximum < 2:
            raise CrupierRouteValidationError("max_panel_size must be at least 2 for panel or fusion routes.")
        if maximum is not None and minimum > maximum:
            raise CrupierRouteValidationError("min_panel_size cannot exceed max_panel_size.")
        if count < minimum:
            raise CrupierRouteValidationError(
                f"Route plans {count} panel models, below min_panel_size={minimum}."
            )
        if maximum is not None and count > maximum:
            raise CrupierRouteValidationError(
                f"Route plans {count} panel models, above max_panel_size={maximum}."
            )

    def _declarative_rule_rejection_reason(
        self,
        card: CapabilityCard,
        request: RequestEnvelope,
    ) -> tuple[str, str] | None:
        for rule in self.config.policy.rules:
            if not self._rule_matches(rule, card, request):
                continue
            reason = rule.reason or f"blocked by policy rule {rule.name!r}"
            if rule.effect == "deny":
                return rule.name, reason
            if rule.effect in {"require_capability", "require_verified_capability"}:
                for capability in rule.capabilities:
                    evidence = capability_evidence(card, capability, declared=_declared_capability(card, capability))
                    if not evidence.supported:
                        return rule.name, f"{reason}: {capability_reason(evidence)}"
                    if rule.effect == "require_verified_capability" and evidence.status != "verified":
                        return rule.name, f"{reason}: {capability_reason(evidence)}"
        return None

    @staticmethod
    def _rule_matches(rule: PolicyRule, card: CapabilityCard, request: RequestEnvelope) -> bool:
        mode = request.mode
        if rule.modes and mode not in rule.modes:
            return False
        if rule.providers and card.model_ref.provider not in rule.providers:
            return False
        return not rule.models or card.model_ref.key in {ModelRef.parse(model).key for model in rule.models}

    @staticmethod
    def _exclude(result: PolicyResult, model: str, reason: str, filter_name: str) -> None:
        result.excluded.append(Exclusion(model=model, reason=reason))
        if filter_name not in result.filters_applied:
            result.filters_applied.append(filter_name)

    @staticmethod
    def _file_input_rejection_reason(
        card: CapabilityCard,
        request: RequestEnvelope,
        require_verified_capabilities: bool,
    ) -> str | None:
        if request.file_plan is None:
            return None
        for modality in request.file_plan.required_model_modalities:
            if modality == "text":
                continue
            if modality == "file":
                declared = card.supports_file_input or "file" in card.modalities_input
                capability = "file_input"
            elif modality == "image":
                declared = "image" in card.modalities_input
                capability = "vision_input"
            else:
                declared = modality in card.modalities_input
                capability = f"{modality}_input"
            evidence = capability_evidence(card, capability, declared=declared)
            if not evidence.supported or (require_verified_capabilities and evidence.status != "verified"):
                return f"file input requires {modality} but {capability_reason(evidence)}"

        for capability in request.file_plan.required_model_capabilities:
            declared = _declared_file_capability(card, capability)
            evidence = capability_evidence(card, capability, declared=declared)
            if not evidence.supported or (require_verified_capabilities and evidence.status != "verified"):
                return f"file input requires {capability} but {capability_reason(evidence)}"
        return None


def _declared_file_capability(card: CapabilityCard, capability: str) -> bool:
    if capability == "vision_input":
        return "image" in card.modalities_input
    if capability == "audio_input":
        return "audio" in card.modalities_input
    if capability == "video_input":
        return "video" in card.modalities_input
    if capability == "file_input":
        return card.supports_file_input or "file" in card.modalities_input
    if capability == "pdf_native_input":
        return card.supports_file_input or "pdf" in card.modalities_input
    return False


def _declared_capability(card: CapabilityCard, capability: str) -> bool:
    if capability == "tool_call":
        return card.supports_tools
    if capability == "structured_output":
        return card.supports_structured_output
    if capability == "streaming":
        return card.supports_streaming
    if capability == "embeddings":
        return card.supports_embeddings
    if capability in {"vision_input", "audio_input", "video_input", "file_input", "pdf_native_input"}:
        return _declared_file_capability(card, capability)
    return False


def _string_set(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if not isinstance(value, Iterable) or isinstance(value, Mapping):
        raise CrupierRouteValidationError(
            "allowed_models and allowed_providers must be a string or list of strings."
        )
    return {str(item) for item in value}


def _model_keys(value: object) -> set[str]:
    return {ModelRef.parse(model).key for model in _string_set(value)}
