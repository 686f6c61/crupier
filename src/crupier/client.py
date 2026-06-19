"""High-level Crupier SDK client."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from .adapters import ProviderAdapter, ProviderModel, build_default_adapters
from .config import CrupierConfig, write_models_allow
from .evals import RoutingEvalRunner
from .executor import RouteExecutor
from .feedback import HumanFeedbackStore
from .models import CapabilityCard, CrupierResult, DecisionTrace, RequestEnvelope, StreamEvent, UpdateReport
from .multimodal import can_execute_native_images, normalize_files, plan_file_representations, prepare_extracted_file_context
from .orchestrator import ModelOrchestrator
from .planner import RoutePlanner
from .policy import PolicyEngine
from .project_audit import ProjectAuditRunner
from .probes import CapabilityProbeRunner, ProbeReport, ReadinessReport
from .registry import ModelRegistry
from .trace_store import TraceStore


class ModelManager:
    def __init__(self, registry: ModelRegistry, adapters: dict[str, ProviderAdapter], config: CrupierConfig):
        self._registry = registry
        self._adapters = adapters
        self._config = config

    def list(self, *, allowed_only: bool = False) -> list[CapabilityCard]:
        return self._registry.list(allowed_only=allowed_only)

    def get(self, model: str) -> CapabilityCard:
        return self._registry.get(model)

    def discover(self, *, provider: str | None = None) -> list[ProviderModel]:
        providers = [provider] if provider else sorted(self._adapters)
        models: list[ProviderModel] = []
        for provider_name in providers:
            adapter = self._adapters.get(provider_name)
            if adapter is None:
                continue
            models.extend(adapter.list_models())
        return sorted(models, key=lambda item: (item.provider, item.id))

    def allow(self, models: list[str], *, replace: bool = False) -> None:
        write_models_allow(self._config.root, models, replace=replace)
        self._config = CrupierConfig.from_toml(self._config.root)
        self._registry.config = self._config
        self._registry._cards = None


class CapabilityManager:
    def __init__(self, registry: ModelRegistry, adapters: dict[str, ProviderAdapter]):
        self._runner = CapabilityProbeRunner(registry, adapters)

    def probe(
        self,
        models: list[str],
        *,
        probes: list[str] | None = None,
        apply: bool = False,
        dry_run: bool = False,
    ) -> ProbeReport:
        return self._runner.probe(models, probes=probes, apply=apply, dry_run=dry_run)

    def readiness(self, models: list[str], *, strict: bool = False) -> ReadinessReport:
        return self._runner.readiness(models, strict=strict)


class ResponsesFacade:
    """Small OpenAI-like compatibility facade."""

    def __init__(self, client: "Crupier"):
        self._client = client

    def create(self, *, input: Any, mode: str | None = None, **kwargs: Any) -> CrupierResult:
        task = kwargs.pop("task", "Respond to the provided input.")
        return self._client.deal(task=task, input=input, mode=mode, **kwargs)


class Crupier:
    """Main SDK entrypoint."""

    def __init__(self, config: CrupierConfig, *, adapters: dict[str, ProviderAdapter] | None = None):
        self.config = config
        self.registry = ModelRegistry(config)
        self.policy = PolicyEngine(config)
        self.adapters = adapters if adapters is not None else build_default_adapters(config)
        self.planner = RoutePlanner(config, orchestrator=self._build_orchestrator())
        self.executor = RouteExecutor(config, adapters=self.adapters)
        self.models = ModelManager(self.registry, self.adapters, config)
        self.capabilities = CapabilityManager(self.registry, self.adapters)
        self.evals = RoutingEvalRunner(self)
        self.audit = ProjectAuditRunner(self)
        self.traces = TraceStore(config.traces_dir)
        self.feedback = HumanFeedbackStore(config.feedback_dir)
        self.responses = ResponsesFacade(self)

    def _build_orchestrator(self):
        mode = self.config.orchestrator.mode
        if mode in {"model", "hybrid"}:
            return ModelOrchestrator(self.config, adapters=self.adapters)
        return None

    @classmethod
    def from_project(cls, path: str | Path = ".") -> "Crupier":
        return cls(CrupierConfig.from_toml(Path(path)))

    @classmethod
    def from_toml(cls, path: str | Path) -> "Crupier":
        return cls(CrupierConfig.from_toml(path))

    @classmethod
    def from_config(cls, config: CrupierConfig | dict[str, Any]) -> "Crupier":
        if isinstance(config, CrupierConfig):
            return cls(config)
        return cls(CrupierConfig.from_dict(config))

    def deal(
        self,
        task: str,
        input: Any = None,
        *,
        mode: str | None = None,
        strategy: str | None = None,
        constraints: dict[str, Any] | None = None,
        files: list[Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
        response_schema: Any = None,
        metadata: dict[str, Any] | None = None,
        trace: bool | str = False,
        dry_run: bool | None = None,
    ) -> CrupierResult:
        constraints = dict(constraints or {})
        if dry_run is None:
            dry_run = bool(constraints.pop("dry_run", True))
        file_assets = normalize_files(files)
        file_plan = plan_file_representations(file_assets, task=task, constraints=constraints)
        metadata = dict(metadata or {})
        execution_files = list(file_assets)
        if file_assets and not dry_run and not can_execute_native_images(file_plan):
            file_context = prepare_extracted_file_context(
                file_assets,
                file_plan,
                max_file_bytes=int(constraints.get("max_file_bytes", 2_000_000)),
                max_chars=int(constraints.get("max_file_context_chars", 80_000)),
            )
            metadata["extracted_file_context"] = file_context
            execution_files = []
        request = RequestEnvelope(
            task=task,
            input=input,
            messages=list(messages or []),
            files=execution_files,
            file_plan=file_plan,
            tools=list(tools or []),
            response_schema=response_schema,
            mode=mode or self.config.project.default_profile,
            strategy=strategy,
            constraints=constraints,
            metadata=metadata,
            tenant_id=metadata.get("tenant_id"),
            user_id_hash=metadata.get("user_id_hash"),
        )

        cards = self.registry.allowed_cards()
        policy_result = self.policy.filter_candidates(request, cards)
        plan = self.planner.plan(request, policy_result.allowed, policy_result.filters_applied)
        self.policy.validate_route(plan, policy_result, request)

        trace_obj = DecisionTrace(
            trace_id=f"trc_{uuid4().hex[:16]}",
            request_summary=self._summarize_task(task),
            candidate_models=[card.model_ref.key for card in cards],
            excluded_models=policy_result.excluded_dicts(),
            policy_filters=policy_result.filters_applied,
            orchestrator_model=self.config.orchestrator.model,
            route_plan=plan,
            storage_decision=self._storage_decision(constraints),
        )

        result = self.executor.execute(request, plan, trace_obj, dry_run=dry_run)
        stored_trace_path = self.traces.write(
            project=self.config.project.name,
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

    async def adeal(self, *args: Any, **kwargs: Any) -> CrupierResult:
        return await asyncio.to_thread(self.deal, *args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any):
        yield StreamEvent(type="route_started", data={"message": "Planning route"})
        result = self.deal(*args, **kwargs)
        if result.route:
            yield StreamEvent(type="route_selected", route=result.route)
        yield StreamEvent(type="final", result=result)

    def update(
        self,
        *,
        dry_run: bool = False,
        apply: bool | None = None,
        online: bool = False,
        provider: str | None = None,
    ) -> UpdateReport:
        if apply is not None:
            dry_run = not apply
        if not online:
            return self.registry.update(dry_run=dry_run)
        return self.registry.update_from_provider_models(
            self.models.discover(provider=provider),
            dry_run=dry_run,
            provider=provider,
        )

    def _storage_decision(self, constraints: dict[str, Any]) -> dict[str, Any]:
        return {
            "logging_mode": constraints.get("logging_mode", self.config.logging.mode),
            "store_trace": bool(constraints.get("store_trace", self.config.logging.persist_traces)),
            "store_prompt": bool(constraints.get("store_prompt", self.config.logging.store_prompts)),
            "store_response": bool(constraints.get("store_response", self.config.logging.store_responses)),
            "redact_secrets": bool(self.config.logging.redact_secrets),
        }

    @staticmethod
    def _summarize_task(task: str) -> str:
        task = " ".join(task.split())
        return task if len(task) <= 180 else task[:177] + "..."
