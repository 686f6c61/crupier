"""Route planning facade backed by orchestrators."""

from __future__ import annotations

from .config import CrupierConfig
from .models import CapabilityCard, PlanningContext, RequestEnvelope, RoutePlan
from .orchestrator import DeterministicOrchestrator, Orchestrator
from .selector import ModelSelector


class RoutePlanner:
    """Compatibility facade for route planning.

    Public callers keep using ``RoutePlanner.plan(...)`` while the actual
    planning logic lives behind the Orchestrator contract.
    """

    def __init__(self, config: CrupierConfig, *, orchestrator: Orchestrator | None = None):
        self.config = config
        self.selector = ModelSelector(config)
        self.orchestrator = orchestrator or DeterministicOrchestrator(config, selector=self.selector)

    def build_context(
        self,
        request: RequestEnvelope,
        candidates: list[CapabilityCard],
        filters_applied: list[str],
    ) -> PlanningContext:
        limit = int(request.constraints.get("selection_trace_limit", 5))
        deterministic_scores = [
            score.to_dict() for score in self.selector.score_all(request, candidates)[:limit]
        ]
        return PlanningContext(
            request=request,
            candidates=list(candidates),
            filters_applied=list(filters_applied),
            deterministic_scores=deterministic_scores,
            orchestrator_mode=self.config.orchestrator.mode,
            metadata={
                "configured_orchestrator_model": self.config.orchestrator.model,
            },
        )

    def plan(
        self,
        request: RequestEnvelope,
        candidates: list[CapabilityCard],
        filters_applied: list[str],
    ) -> RoutePlan:
        context = self.build_context(request, candidates, filters_applied)
        return self.orchestrator.plan(context)
