"""Explainable model selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .capabilities import CapabilityEvidence, capability_evidence
from .config import CrupierConfig, ScoringSettings, ollama_is_local
from .costs import estimate_model_cost, estimate_tokens
from .model_profiles import classify_task_signal_weights
from .models import CapabilityCard, RequestEnvelope


@dataclass(slots=True)
class ScoreTerm:
    name: str
    value: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SelectionScore:
    model: str
    score: float
    terms: list[ScoreTerm] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "score": self.score,
            "terms": [term.to_dict() for term in self.terms],
        }


class ModelSelector:
    """Ranks policy-allowed models with a human-readable breakdown."""

    def __init__(self, config: CrupierConfig):
        self.config = config
        self.scoring = config.scoring

    def rank(self, request: RequestEnvelope, candidates: list[CapabilityCard]) -> list[CapabilityCard]:
        scores = self.score_all(request, candidates)
        by_key = {card.model_ref.key: card for card in candidates}
        return [by_key[score.model] for score in scores]

    def score_all(self, request: RequestEnvelope, candidates: list[CapabilityCard]) -> list[SelectionScore]:
        scores = [self.score(request, card) for card in candidates]
        return sorted(scores, key=lambda item: (item.score, item.model), reverse=True)

    def score(self, request: RequestEnvelope, card: CapabilityCard) -> SelectionScore:
        terms: list[ScoreTerm] = []
        mode = request.mode or self.config.project.default_profile
        profile = self.config.profiles.get(mode)
        preferences = set(profile.prefer if profile else [])
        task_signal_weights = self._task_signal_weights(request)
        task_signals = set(task_signal_weights)

        self._add(terms, "quality_tier", self._tier_weight("quality", card.quality_tier), f"quality={card.quality_tier}")

        matched_preferences = sorted(set(card.strengths).intersection(preferences))
        if matched_preferences:
            self._add(
                terms,
                "profile_preferences",
                self.scoring.profile_preference_weight * len(matched_preferences),
                "matches " + ", ".join(matched_preferences),
            )

        matched_task = sorted(set(card.strengths).intersection(task_signals))
        if matched_task:
            value = self.scoring.task_signal_weight * sum(task_signal_weights.get(signal, 1.0) for signal in matched_task)
            self._add(
                terms,
                "task_signals",
                value,
                "task suggests "
                + ", ".join(f"{signal}={task_signal_weights.get(signal, 1.0):.2g}" for signal in matched_task),
            )

        skill_terms = []
        for signal in sorted(task_signals):
            skill_value = card.skill_scores.get(signal)
            if isinstance(skill_value, int | float) and float(skill_value) >= self.scoring.skill_fit_min_score:
                skill_terms.append((signal, float(skill_value), task_signal_weights.get(signal, 1.0)))
        if skill_terms:
            skill_fit_value = min(
                self.scoring.skill_fit_cap,
                sum(
                    (score - self.scoring.skill_fit_baseline) * self.scoring.skill_fit_multiplier * weight
                    for _, score, weight in skill_terms
                ),
            )
            detail = ", ".join(f"{signal}={score:g}x{weight:.2g}" for signal, score, weight in skill_terms[:6])
            self._add(terms, "skill_fit", skill_fit_value, "decision profile skills " + detail)

        if mode == "cheap":
            self._add(
                terms,
                "cheap_mode_cost",
                self._tier_weight("cost", card.cost_tier) * self.scoring.cheap_mode_cost_multiplier,
                f"cost={card.cost_tier}",
            )
        if mode == "fast":
            self._add(
                terms,
                "fast_mode_latency",
                self._tier_weight("latency", card.latency_tier) * self.scoring.fast_mode_latency_multiplier,
                f"latency={card.latency_tier}",
            )
        if mode == "private" and card.model_ref.provider == "ollama" and ollama_is_local(self.config):
            self._add(
                terms,
                "private_mode_local",
                self.scoring.private_mode_ollama_bonus,
                "private mode prefers configured Ollama candidates",
            )
        if mode in {"quality", "research", "agentic"}:
            self._add(
                terms,
                f"{mode}_mode_quality",
                self._tier_weight("quality", card.quality_tier),
                f"{mode} mode values model quality",
            )

        budget_score = self._budget_fit_score(request, card)
        if budget_score:
            self._add(terms, budget_score[0], budget_score[1], budget_score[2])

        if request.tools:
            self._add_capability_term(
                terms,
                "tool_support",
                capability_evidence(card, "tool_call", declared=card.supports_tools),
            )
        if request.response_schema is not None or bool(request.constraints.get("response_schema")):
            self._add_capability_term(
                terms,
                "structured_output_support",
                capability_evidence(card, "structured_output", declared=card.supports_structured_output),
            )
        if bool(request.constraints.get("stream", False) or request.constraints.get("require_streaming", False)):
            self._add_capability_term(
                terms,
                "streaming_support",
                capability_evidence(card, "streaming", declared=card.supports_streaming),
            )
        if request.file_plan is not None:
            scored_file_capabilities: set[str] = set()
            for modality in request.file_plan.required_model_modalities:
                if modality == "text":
                    continue
                capability = "file_input" if modality == "file" else f"{modality}_input"
                if modality == "image":
                    capability = "vision_input"
                scored_file_capabilities.add(capability)
                declared = _declared_file_capability(card, capability)
                self._add_capability_term(
                    terms,
                    f"{modality}_input_support",
                    capability_evidence(card, capability, declared=declared),
                )
            for capability in request.file_plan.required_model_capabilities:
                if capability in scored_file_capabilities:
                    continue
                declared = _declared_file_capability(card, capability)
                self._add_capability_term(
                    terms,
                    f"{capability}_support",
                    capability_evidence(card, capability, declared=declared),
                )

        eval_score = self._local_eval_score(card, mode)
        if eval_score:
            self._add(terms, "local_eval", eval_score * self.scoring.local_eval_weight, f"local eval signal for {mode}")

        human_feedback_score = self._human_feedback_score(card, mode)
        if human_feedback_score:
            self._add(
                terms,
                "human_feedback",
                human_feedback_score * self.scoring.human_feedback_weight,
                f"project human feedback signal for {mode}",
            )

        if card.deprecation:
            self._add(terms, "deprecation_penalty", self.scoring.deprecation_penalty, "model card marks model as deprecated")
        routing_status = card.routing_hints.get("routing_status")
        if routing_status in {"legacy", "unknown"}:
            self._add(
                terms,
                "routing_status_penalty",
                self.scoring.routing_status_penalty,
                f"routing_status={routing_status}",
            )
        if card.routing_hints.get("requires_opt_in") and not card.routing_hints.get("production_default"):
            self._add(terms, "opt_in_penalty", self.scoring.opt_in_penalty, "model requires explicit opt-in for default routing")
        if mode == "cheap" and card.cost_tier == "high":
            self._add(terms, "high_cost_penalty", self.scoring.cheap_high_cost_penalty, "cheap mode penalizes high-cost models")
        if mode == "fast" and card.latency_tier not in {"fast", "unknown"}:
            self._add(terms, "latency_penalty", self.scoring.fast_latency_penalty, "fast mode penalizes slower models")
        if card.model_ref.stability in {"preview", "experimental"}:
            self._add(
                terms,
                "stability_penalty",
                self.scoring.preview_stability_penalty,
                f"model stability={card.model_ref.stability}",
            )

        total = sum(term.value for term in terms)
        return SelectionScore(model=card.model_ref.key, score=total, terms=terms)

    @staticmethod
    def _add(terms: list[ScoreTerm], name: str, value: float, reason: str) -> None:
        if value:
            terms.append(ScoreTerm(name=name, value=float(value), reason=reason))

    def _add_capability_term(self, terms: list[ScoreTerm], name: str, evidence: CapabilityEvidence) -> None:
        weights = {
            "verified": self.scoring.verified_capability_weight,
            "inferred": self.scoring.inferred_capability_weight,
            "unknown": 0.0,
            "failed": self.scoring.failed_capability_penalty,
        }
        value = weights.get(evidence.status, 0)
        if value:
            self._add(
                terms,
                name,
                value,
                f"{evidence.capability} support is {evidence.status} via {evidence.source}",
            )

    @staticmethod
    def _local_eval_score(card: CapabilityCard, mode: str) -> float:
        total = 0.0
        for key in (mode, "overall", f"eval:{mode}", "eval:overall"):
            try:
                total += float(card.local_eval_scores.get(key, 0))
            except (TypeError, ValueError):
                continue
        return total

    @staticmethod
    def _human_feedback_score(card: CapabilityCard, mode: str) -> float:
        total = 0.0
        for key in (f"human:{mode}", "human:overall"):
            try:
                total += float(card.local_eval_scores.get(key, 0))
            except (TypeError, ValueError):
                continue
        return total

    def _budget_fit_score(self, request: RequestEnvelope, card: CapabilityCard) -> tuple[str, float, str] | None:
        max_cost = request.constraints.get("max_cost_usd", self.config.routing.max_cost_per_request_usd)
        if max_cost is None:
            return None
        try:
            budget = float(max_cost)
        except (TypeError, ValueError):
            return None
        if budget <= 0:
            return None
        input_text = f"{request.task} {request.input if isinstance(request.input, str) else ''}"
        input_tokens = estimate_tokens(input_text)
        output_tokens = int(request.constraints.get("max_output_tokens", request.constraints.get("max_tokens", 1024)))
        estimate = estimate_model_cost(card, input_tokens=input_tokens, output_tokens=output_tokens)
        if estimate > budget:
            return (
                "budget_fit_penalty",
                self.scoring.budget_over_penalty,
                f"single-call estimate ${estimate:.4f} exceeds budget ${budget:.4f}",
            )
        if estimate <= budget * 0.5:
            return (
                "budget_fit",
                self.scoring.budget_comfort_bonus,
                f"single-call estimate ${estimate:.4f} is comfortably under budget ${budget:.4f}",
            )
        return (
            "budget_fit",
            self.scoring.budget_within_bonus,
            f"single-call estimate ${estimate:.4f} is within budget ${budget:.4f}",
        )

    def _tier_weight(self, family: str, tier: str) -> float:
        scoring: ScoringSettings = self.scoring
        if family == "quality":
            return float(scoring.quality_weight.get(tier, 0.0))
        if family == "cost":
            return float(scoring.cost_weight.get(tier, 0.0))
        if family == "latency":
            return float(scoring.latency_weight.get(tier, 0.0))
        return 0.0

    @staticmethod
    def _task_signal_weights(request: RequestEnvelope) -> dict[str, float]:
        return classify_task_signal_weights(request)


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
