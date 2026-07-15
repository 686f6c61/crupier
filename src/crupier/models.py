"""Core data models for Crupier.

The project intentionally uses dataclasses instead of a runtime validation
dependency in the base package. Provider-specific and schema-specific layers
can add richer validation later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _compact_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


PROVIDER_ALIASES = {
    "claude": "anthropic",
}


@dataclass(frozen=True, slots=True)
class ModelRef:
    """Normalized provider/model reference.

    The canonical string form is ``provider:model``. The model segment may
    contain additional colons, which matters for Ollama tags.
    """

    provider: str
    model: str
    stability: str = "stable"
    source: str = "native"
    alias: str | None = None

    @classmethod
    def parse(cls, value: str) -> "ModelRef":
        if ":" not in value:
            raise ValueError(f"Model reference must be provider:model, got {value!r}")
        provider, model = value.split(":", 1)
        provider = provider.strip().lower()
        provider = PROVIDER_ALIASES.get(provider, provider)
        model = model.strip()
        if not provider or not model:
            raise ValueError(f"Model reference must be provider:model, got {value!r}")
        stability = "latest" if "latest" in model else "stable"
        if "preview" in model:
            stability = "preview"
        if "experimental" in model or "exp" in model:
            stability = "experimental"
        return cls(provider=provider, model=model, stability=stability)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRef":
        return cls(
            provider=data["provider"],
            model=data["model"],
            stability=data.get("stability", "stable"),
            source=data.get("source", "native"),
            alias=data.get("alias"),
        )

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    def to_dict(self) -> dict[str, Any]:
        return _compact_none(asdict(self))

    def __str__(self) -> str:
        return self.key


@dataclass(slots=True)
class CostEstimate:
    estimated_usd: float = 0.0
    actual_usd: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CostEstimate":
        if not data:
            return cls()
        return cls(
            estimated_usd=float(data.get("estimated_usd", 0.0)),
            actual_usd=data.get("actual_usd"),
        )

    def to_dict(self) -> dict[str, Any]:
        return _compact_none(asdict(self))


@dataclass(slots=True)
class CapabilityCard:
    """Local model card used by policy, planning, and update reports."""

    model_ref: ModelRef
    last_updated: str
    context_window: int | None = None
    max_output_tokens: int | None = None
    model_kind: str = "chat"
    modalities_input: list[str] = field(default_factory=lambda: ["text"])
    modalities_output: list[str] = field(default_factory=lambda: ["text"])
    supports_embeddings: bool = False
    embedding_dimensions: int | None = None
    embedding_input_modalities: list[str] = field(default_factory=list)
    supports_tools: bool = False
    supports_structured_output: bool = False
    supports_streaming: bool = True
    supports_web: bool = False
    supports_file_input: bool = False
    supports_code_execution: bool = False
    pricing: dict[str, Any] = field(default_factory=dict)
    latency_profile: dict[str, Any] = field(default_factory=dict)
    data_retention: str | None = None
    zdr_eligible: bool | None = None
    regions: list[str] = field(default_factory=list)
    unsupported_params: list[str] = field(default_factory=list)
    known_edge_cases: list[str] = field(default_factory=list)
    benchmarks: dict[str, Any] = field(default_factory=dict)
    skill_scores: dict[str, Any] = field(default_factory=dict)
    natural_profile: dict[str, Any] = field(default_factory=dict)
    routing_hints: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    local_eval_scores: dict[str, Any] = field(default_factory=dict)
    capability_status: dict[str, Any] = field(default_factory=dict)
    probe_results: dict[str, Any] = field(default_factory=dict)
    deprecation: dict[str, Any] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    cost_tier: str = "unknown"
    latency_tier: str = "unknown"
    quality_tier: str = "unknown"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityCard":
        model_ref_data = data["model_ref"]
        model_ref = (
            ModelRef.from_dict(model_ref_data)
            if isinstance(model_ref_data, dict)
            else ModelRef.parse(str(model_ref_data))
        )
        return cls(
            model_ref=model_ref,
            last_updated=data.get("last_updated", "unknown"),
            context_window=data.get("context_window"),
            max_output_tokens=data.get("max_output_tokens"),
            model_kind=data.get("model_kind", "chat"),
            modalities_input=list(data.get("modalities_input", ["text"])),
            modalities_output=list(data.get("modalities_output", ["text"])),
            supports_embeddings=bool(data.get("supports_embeddings", False)),
            embedding_dimensions=data.get("embedding_dimensions"),
            embedding_input_modalities=list(data.get("embedding_input_modalities", [])),
            supports_tools=bool(data.get("supports_tools", False)),
            supports_structured_output=bool(data.get("supports_structured_output", False)),
            supports_streaming=bool(data.get("supports_streaming", True)),
            supports_web=bool(data.get("supports_web", False)),
            supports_file_input=bool(data.get("supports_file_input", False)),
            supports_code_execution=bool(data.get("supports_code_execution", False)),
            pricing=dict(data.get("pricing", {})),
            latency_profile=dict(data.get("latency_profile", {})),
            data_retention=data.get("data_retention"),
            zdr_eligible=data.get("zdr_eligible"),
            regions=list(data.get("regions", [])),
            unsupported_params=list(data.get("unsupported_params", [])),
            known_edge_cases=list(data.get("known_edge_cases", [])),
            benchmarks=dict(data.get("benchmarks", {})),
            skill_scores=dict(data.get("skill_scores", {})),
            natural_profile=dict(data.get("natural_profile", {})),
            routing_hints=dict(data.get("routing_hints", {})),
            evidence=dict(data.get("evidence", {})),
            local_eval_scores=dict(data.get("local_eval_scores", {})),
            capability_status=dict(data.get("capability_status", {})),
            probe_results=dict(data.get("probe_results", {})),
            deprecation=dict(data.get("deprecation") or {}),
            strengths=list(data.get("strengths", [])),
            cost_tier=data.get("cost_tier", "unknown"),
            latency_tier=data.get("latency_tier", "unknown"),
            quality_tier=data.get("quality_tier", "unknown"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model_ref"] = self.model_ref.to_dict()
        return data


@dataclass(slots=True)
class FileAsset:
    """Normalized file metadata.

    The object can keep a local URI/path for future execution, but public
    route/trace dictionaries omit it by default to avoid leaking local paths.
    """

    kind: str
    name: str | None = None
    uri: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    page_count: int | None = None
    duration_seconds: float | None = None
    exists: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileAsset":
        return cls(
            kind=str(data.get("kind", "unknown")),
            name=data.get("name"),
            uri=data.get("uri"),
            mime_type=data.get("mime_type"),
            size_bytes=data.get("size_bytes"),
            page_count=data.get("page_count"),
            duration_seconds=data.get("duration_seconds"),
            exists=data.get("exists"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self, *, include_uri: bool = False) -> dict[str, Any]:
        data = {
            "kind": self.kind,
            "name": self.name,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "page_count": self.page_count,
            "duration_seconds": self.duration_seconds,
            "exists": self.exists,
            "metadata": self.metadata,
        }
        if include_uri:
            data["uri"] = self.uri
        return _compact_none(data)


@dataclass(slots=True)
class FileRepresentation:
    asset_name: str | None
    kind: str
    representation: str
    required_model_modalities: list[str] = field(default_factory=list)
    required_model_capabilities: list[str] = field(default_factory=list)
    pipeline: list[str] = field(default_factory=list)
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileRepresentation":
        return cls(
            asset_name=data.get("asset_name"),
            kind=str(data.get("kind", "unknown")),
            representation=str(data.get("representation", "metadata_only")),
            required_model_modalities=list(data.get("required_model_modalities", [])),
            required_model_capabilities=list(data.get("required_model_capabilities", [])),
            pipeline=list(data.get("pipeline", [])),
            reason=data.get("reason", ""),
            warnings=list(data.get("warnings", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return _compact_none(asdict(self))


@dataclass(slots=True)
class FileRoutingPlan:
    assets: list[FileAsset] = field(default_factory=list)
    representations: list[FileRepresentation] = field(default_factory=list)
    required_model_modalities: list[str] = field(default_factory=list)
    required_model_capabilities: list[str] = field(default_factory=list)
    extraction_required: bool = False
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileRoutingPlan":
        return cls(
            assets=[FileAsset.from_dict(item) for item in data.get("assets", [])],
            representations=[
                FileRepresentation.from_dict(item) for item in data.get("representations", [])
            ],
            required_model_modalities=list(data.get("required_model_modalities", [])),
            required_model_capabilities=list(data.get("required_model_capabilities", [])),
            extraction_required=bool(data.get("extraction_required", False)),
            warnings=list(data.get("warnings", [])),
        )

    def to_dict(self, *, include_uri: bool = False) -> dict[str, Any]:
        return {
            "assets": [asset.to_dict(include_uri=include_uri) for asset in self.assets],
            "representations": [item.to_dict() for item in self.representations],
            "required_model_modalities": self.required_model_modalities,
            "required_model_capabilities": self.required_model_capabilities,
            "extraction_required": self.extraction_required,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class RequestEnvelope:
    task: str
    input: Any = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    files: list[FileAsset] = field(default_factory=list)
    file_plan: FileRoutingPlan | None = None
    tools: list[Any] = field(default_factory=list)
    response_schema: Any = None
    mode: str | None = None
    strategy: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    tenant_id: str | None = None
    user_id_hash: str | None = None


@dataclass(slots=True)
class PlanningContext:
    """Structured input passed to route orchestrators."""

    request: RequestEnvelope
    candidates: list[CapabilityCard]
    filters_applied: list[str] = field(default_factory=list)
    deterministic_scores: list[dict[str, Any]] = field(default_factory=list)
    orchestrator_mode: str = "deterministic"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_models(self) -> list[str]:
        return [card.model_ref.key for card in self.candidates]

    def to_dict(self, *, summary: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "task": self.request.task,
            "mode": self.request.mode,
            "strategy": self.request.strategy,
            "constraints": dict(self.request.constraints),
            "candidate_models": self.candidate_models,
            "filters_applied": list(self.filters_applied),
            "deterministic_scores": list(self.deterministic_scores),
            "orchestrator_mode": self.orchestrator_mode,
            "metadata": dict(self.metadata),
        }
        if self.request.file_plan is not None:
            data["input_plan"] = {"files": self.request.file_plan.to_dict()}
        if not summary:
            data["candidates"] = [card.to_dict() for card in self.candidates]
        return data


@dataclass(slots=True)
class RouteStep:
    role: str
    model: str | None = None
    models: list[str] = field(default_factory=list)
    timeout_ms: int | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouteStep":
        return cls(
            role=data["role"],
            model=data.get("model"),
            models=list(data.get("models", [])),
            timeout_ms=data.get("timeout_ms"),
            params=dict(data.get("params", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return _compact_none(asdict(self))


@dataclass(slots=True)
class RoutePlan:
    strategy: str
    steps: list[RouteStep]
    estimated_cost: CostEstimate = field(default_factory=CostEstimate)
    estimated_latency_ms: int | None = None
    reason: str = ""
    policy_filters_applied: list[str] = field(default_factory=list)
    risk_level: str = "medium"
    requires_user_confirmation: bool = False
    summary: str = ""
    selection_scores: list[dict[str, Any]] = field(default_factory=list)
    input_plan: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoutePlan":
        return cls(
            strategy=data["strategy"],
            steps=[RouteStep.from_dict(step) for step in data.get("steps", [])],
            estimated_cost=CostEstimate.from_dict(data.get("estimated_cost")),
            estimated_latency_ms=data.get("estimated_latency_ms"),
            reason=data.get("reason", ""),
            policy_filters_applied=list(data.get("policy_filters_applied", [])),
            risk_level=data.get("risk_level", "medium"),
            requires_user_confirmation=bool(data.get("requires_user_confirmation", False)),
            summary=data.get("summary", ""),
            selection_scores=list(data.get("selection_scores", [])),
            input_plan=dict(data.get("input_plan", {})),
        )

    @property
    def models(self) -> list[str]:
        seen: list[str] = []
        for step in self.steps:
            for model in [step.model, *step.models]:
                if model and model not in seen:
                    seen.append(model)
        return seen

    @property
    def model_summary(self) -> str:
        return ", ".join(self.models)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "steps": [step.to_dict() for step in self.steps],
            "estimated_cost": self.estimated_cost.to_dict(),
            "estimated_latency_ms": self.estimated_latency_ms,
            "reason": self.reason,
            "policy_filters_applied": self.policy_filters_applied,
            "risk_level": self.risk_level,
            "requires_user_confirmation": self.requires_user_confirmation,
            "summary": self.summary,
            "selection_scores": self.selection_scores,
            "input_plan": self.input_plan,
        }


@dataclass(slots=True)
class DecisionTrace:
    trace_id: str
    request_summary: str
    candidate_models: list[str] = field(default_factory=list)
    excluded_models: list[dict[str, Any]] = field(default_factory=list)
    policy_filters: list[str] = field(default_factory=list)
    orchestrator_model: str | None = None
    route_plan: RoutePlan | None = None
    provider_calls: list[dict[str, Any]] = field(default_factory=list)
    fallbacks: list[dict[str, Any]] = field(default_factory=list)
    cost: CostEstimate = field(default_factory=CostEstimate)
    latency_ms: int | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)
    storage_decision: dict[str, Any] = field(default_factory=dict)
    final_quality_signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, summary: bool = False) -> dict[str, Any]:
        data = {
            "trace_id": self.trace_id,
            "request_summary": self.request_summary,
            "candidate_models": self.candidate_models,
            "excluded_models": self.excluded_models,
            "policy_filters": self.policy_filters,
            "orchestrator_model": self.orchestrator_model,
            "route_plan": self.route_plan.to_dict() if self.route_plan else None,
            "cost": self.cost.to_dict(),
            "latency_ms": self.latency_ms,
            "storage_decision": self.storage_decision,
        }
        if not summary:
            data.update(
                {
                    "provider_calls": self.provider_calls,
                    "fallbacks": self.fallbacks,
                    "errors": self.errors,
                    "final_quality_signals": self.final_quality_signals,
                }
            )
        return data


@dataclass(slots=True)
class CrupierResult:
    output_text: str = ""
    output_json: Any = None
    raw_outputs: list[Any] = field(default_factory=list)
    route: RoutePlan | None = None
    trace: DecisionTrace | None = None
    cost: CostEstimate = field(default_factory=CostEstimate)
    latency_ms: int | None = None
    warnings: list[str] = field(default_factory=list)
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, trace_summary: bool = True) -> dict[str, Any]:
        return {
            "output_text": self.output_text,
            "output_json": self.output_json,
            "raw_outputs": self.raw_outputs,
            "route": self.route.to_dict() if self.route else None,
            "trace": self.trace.to_dict(summary=trace_summary) if self.trace else None,
            "cost": self.cost.to_dict(),
            "latency_ms": self.latency_ms,
            "warnings": self.warnings,
            "provider_metadata": self.provider_metadata,
        }


@dataclass(slots=True)
class OperationResult:
    operation: str
    model: str
    data: Any = None
    raw: Any = None
    route: RoutePlan | None = None
    trace: DecisionTrace | None = None
    cost: CostEstimate = field(default_factory=CostEstimate)
    latency_ms: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self, *, trace_summary: bool = True) -> dict[str, Any]:
        data = {"bytes": len(self.data)} if isinstance(self.data, bytes | bytearray) else self.data
        return {
            "operation": self.operation,
            "model": self.model,
            "data": data,
            "route": self.route.to_dict() if self.route else None,
            "trace": self.trace.to_dict(summary=trace_summary) if self.trace else None,
            "cost": self.cost.to_dict(),
            "latency_ms": self.latency_ms,
            "usage": self.usage,
            "provider_metadata": self.provider_metadata,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class StreamEvent:
    type: str
    delta: str | None = None
    route: RoutePlan | None = None
    result: CrupierResult | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UpdateReport:
    changed_models: list[str] = field(default_factory=list)
    added_models: list[str] = field(default_factory=list)
    removed_models: list[str] = field(default_factory=list)
    modified_models: list[str] = field(default_factory=list)
    unchanged_models: list[str] = field(default_factory=list)
    deprecated_models: list[str] = field(default_factory=list)
    price_changes: list[dict[str, Any]] = field(default_factory=list)
    profile_changes: list[dict[str, Any]] = field(default_factory=list)
    diff: dict[str, Any] = field(default_factory=dict)
    model_states: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    written_files: list[str] = field(default_factory=list)
    requires_confirmation: bool = False
    dry_run: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
