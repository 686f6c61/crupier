"""Explainable model selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .capabilities import CapabilityEvidence, capability_evidence
from .config import CrupierConfig
from .costs import estimate_model_cost, estimate_tokens
from .model_profiles import classify_task_signals
from .models import CapabilityCard, RequestEnvelope


QUALITY_WEIGHT = {"unknown": 0, "strong": 2, "frontier": 4}
COST_WEIGHT = {"unknown": 0, "low": 4, "medium": 2, "high": 0}
LATENCY_WEIGHT = {"unknown": 0, "fast": 4, "medium": 2, "slow": 0}


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
        task_signals = self._task_signals(request)

        self._add(terms, "quality_tier", QUALITY_WEIGHT.get(card.quality_tier, 0), f"quality={card.quality_tier}")

        matched_preferences = sorted(set(card.strengths).intersection(preferences))
        if matched_preferences:
            self._add(
                terms,
                "profile_preferences",
                3 * len(matched_preferences),
                "matches " + ", ".join(matched_preferences),
            )

        matched_task = sorted(set(card.strengths).intersection(task_signals))
        if matched_task:
            self._add(terms, "task_signals", 2 * len(matched_task), "task suggests " + ", ".join(matched_task))

        skill_terms = []
        for signal in sorted(task_signals):
            value = card.skill_scores.get(signal)
            if isinstance(value, int | float) and float(value) >= 6.0:
                skill_terms.append((signal, float(value)))
        if skill_terms:
            value = min(12.0, sum((score - 5.0) * 0.8 for _, score in skill_terms))
            detail = ", ".join(f"{signal}={score:g}" for signal, score in skill_terms[:6])
            self._add(terms, "skill_fit", value, "decision profile skills " + detail)

        if mode == "cheap":
            self._add(terms, "cheap_mode_cost", COST_WEIGHT.get(card.cost_tier, 0) * 2, f"cost={card.cost_tier}")
        if mode == "fast":
            self._add(
                terms,
                "fast_mode_latency",
                LATENCY_WEIGHT.get(card.latency_tier, 0) * 2,
                f"latency={card.latency_tier}",
            )
        if mode == "private" and card.model_ref.provider == "ollama":
            self._add(terms, "private_mode_local", 10, "private mode prefers configured Ollama candidates")
        if mode in {"quality", "research", "agentic"}:
            self._add(
                terms,
                f"{mode}_mode_quality",
                QUALITY_WEIGHT.get(card.quality_tier, 0),
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
            self._add(terms, "local_eval", eval_score, f"local eval signal for {mode}")

        human_feedback_score = self._human_feedback_score(card, mode)
        if human_feedback_score:
            self._add(
                terms,
                "human_feedback",
                human_feedback_score,
                f"project human feedback signal for {mode}",
            )

        if card.deprecation:
            self._add(terms, "deprecation_penalty", -100, "model card marks model as deprecated")
        routing_status = card.routing_hints.get("routing_status")
        if routing_status in {"legacy", "unknown"}:
            self._add(terms, "routing_status_penalty", -6, f"routing_status={routing_status}")
        if card.routing_hints.get("requires_opt_in") and not card.routing_hints.get("production_default"):
            self._add(terms, "opt_in_penalty", -4, "model requires explicit opt-in for default routing")
        if mode == "cheap" and card.cost_tier == "high":
            self._add(terms, "high_cost_penalty", -4, "cheap mode penalizes high-cost models")
        if mode == "fast" and card.latency_tier not in {"fast", "unknown"}:
            self._add(terms, "latency_penalty", -3, "fast mode penalizes slower models")
        if card.model_ref.stability in {"preview", "experimental"}:
            self._add(terms, "stability_penalty", -5, f"model stability={card.model_ref.stability}")

        total = sum(term.value for term in terms)
        return SelectionScore(model=card.model_ref.key, score=total, terms=terms)

    @staticmethod
    def _add(terms: list[ScoreTerm], name: str, value: float, reason: str) -> None:
        if value:
            terms.append(ScoreTerm(name=name, value=float(value), reason=reason))

    def _add_capability_term(self, terms: list[ScoreTerm], name: str, evidence: CapabilityEvidence) -> None:
        weights = {"verified": 6, "inferred": 2, "unknown": 0, "failed": -20}
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
            return ("budget_fit_penalty", -30.0, f"single-call estimate ${estimate:.4f} exceeds budget ${budget:.4f}")
        if estimate <= budget * 0.5:
            return ("budget_fit", 3.0, f"single-call estimate ${estimate:.4f} is comfortably under budget ${budget:.4f}")
        return ("budget_fit", 1.0, f"single-call estimate ${estimate:.4f} is within budget ${budget:.4f}")

    @staticmethod
    def _task_signals(request: RequestEnvelope) -> set[str]:
        return classify_task_signals(request)


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
