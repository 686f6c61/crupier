"""Project-local scoring suggestions from eval and feedback evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

from .config import CrupierConfig, write_scoring_settings
from .models import CapabilityCard
from .registry import ModelRegistry


@dataclass(slots=True)
class ScoringSuggestion:
    field: str
    current: float
    suggested: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScoringTuningReport:
    applied: bool
    evidence: dict[str, Any]
    suggestions: list[ScoringSuggestion] = field(default_factory=list)
    written_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "evidence": self.evidence,
            "suggestions": [item.to_dict() for item in self.suggestions],
            "written_path": self.written_path,
        }

    @property
    def updates(self) -> dict[str, float]:
        return {item.field: item.suggested for item in self.suggestions}


def suggest_scoring_from_project(
    config: CrupierConfig,
    *,
    registry: ModelRegistry | None = None,
    apply: bool = False,
    min_samples: int = 2,
) -> ScoringTuningReport:
    """Suggest conservative scoring-weight updates from applied eval/feedback scores."""

    registry = registry or ModelRegistry(config)
    cards = registry.list(allowed_only=True)
    eval_values = _score_values(cards, prefixes=("eval:",), include_plain=True)
    human_values = _score_values(cards, prefixes=("human:",), include_plain=False)
    suggestions: list[ScoringSuggestion] = []
    if len(eval_values) >= min_samples:
        suggested = _suggest_weight(config.scoring.local_eval_weight, eval_values)
        if suggested != config.scoring.local_eval_weight:
            suggestions.append(
                ScoringSuggestion(
                    field="local_eval_weight",
                    current=config.scoring.local_eval_weight,
                    suggested=suggested,
                    reason=f"{len(eval_values)} eval signals show project-specific ranking separation.",
                )
            )
    if len(human_values) >= min_samples:
        suggested = _suggest_weight(config.scoring.human_feedback_weight, human_values)
        if suggested != config.scoring.human_feedback_weight:
            suggestions.append(
                ScoringSuggestion(
                    field="human_feedback_weight",
                    current=config.scoring.human_feedback_weight,
                    suggested=suggested,
                    reason=f"{len(human_values)} human feedback signals are available for ranking calibration.",
                )
            )
    evidence = {
        "allowed_models": [card.model_ref.key for card in cards],
        "eval_signal_count": len(eval_values),
        "human_feedback_signal_count": len(human_values),
        "eval_mean_abs": _mean_abs(eval_values),
        "human_feedback_mean_abs": _mean_abs(human_values),
        "min_samples": min_samples,
    }
    written_path = None
    if apply and suggestions:
        written_path = str(write_scoring_settings(config.root, {item.field: item.suggested for item in suggestions}))
    return ScoringTuningReport(
        applied=bool(apply and suggestions),
        evidence=evidence,
        suggestions=suggestions,
        written_path=written_path,
    )


def _score_values(
    cards: list[CapabilityCard],
    *,
    prefixes: tuple[str, ...],
    include_plain: bool,
) -> list[float]:
    values: list[float] = []
    ignored_prefixes = ("probe_", "human:")
    for card in cards:
        for key, raw in card.local_eval_scores.items():
            if key.startswith(ignored_prefixes) and not key.startswith(prefixes):
                continue
            if not key.startswith(prefixes) and (not include_plain or ":" in key):
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
    return values


def _suggest_weight(current: float, values: list[float]) -> float:
    signal = min(1.0, _mean_abs(values) / 5.0)
    spread = min(1.0, (max(values) - min(values)) / 10.0) if values else 0.0
    multiplier = 1.0 + (signal * 0.35) + (spread * 0.25)
    return round(_clamp(current * multiplier, 0.25, 3.0), 3)


def _mean_abs(values: list[float]) -> float:
    return round(mean(abs(value) for value in values), 3) if values else 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
