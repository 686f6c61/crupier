"""High-level Crupier SDK client."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from .adapters import ProviderAdapter, ProviderModel, build_default_adapters
from .config import CrupierConfig, write_models_allow, write_orchestrator_settings
from .errors import (
    CrupierConfigError,
    CrupierPolicyError,
    CrupierProviderAuthError,
    CrupierProviderRateLimitError,
    CrupierProviderUnavailableError,
)
from .evals import RoutingEvalRunner
from .executor import RouteExecutor
from .feedback import HumanFeedbackStore
from .models import CapabilityCard, CrupierResult, DecisionTrace, ModelRef, RequestEnvelope, StreamEvent, UpdateReport
from .multimodal import can_execute_native_images, normalize_files, plan_file_representations, prepare_extracted_file_context
from .orchestrator import ModelOrchestrator
from .planner import RoutePlanner
from .policy import Exclusion, PolicyEngine
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

    def discover(
        self,
        *,
        provider: str | None = None,
        skip_unavailable: bool = False,
        warnings: list[str] | None = None,
    ) -> list[ProviderModel]:
        providers = [provider] if provider else sorted(self._adapters)
        models: list[ProviderModel] = []
        for provider_name in providers:
            adapter = self._adapters.get(provider_name)
            if adapter is None:
                continue
            try:
                models.extend(adapter.list_models())
            except (CrupierProviderAuthError, CrupierProviderRateLimitError, CrupierProviderUnavailableError) as exc:
                if not skip_unavailable:
                    raise
                if warnings is not None:
                    warnings.append(
                        f"Skipped provider {provider_name!r} because its API key or endpoint is not operational: {exc}"
                    )
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
        self._provider_visibility_cache: dict[str, tuple[set[str] | None, str | None]] = {}

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

    def configure_orchestrator(
        self,
        *,
        mode: str | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
        fallback: str | None = None,
        temperature: float | None = None,
        require_validated_plan: bool | None = None,
        max_repairs: int | None = None,
        allow_prompt_summary_only: bool | None = None,
        persist: bool = False,
    ) -> "Crupier":
        """Configure the model-powered route orchestrator.

        Set ``persist=True`` to write the change into ``crupier.toml``.
        """

        if mode is not None and mode not in {"deterministic", "model", "hybrid"}:
            raise CrupierConfigError("orchestrator mode must be one of: deterministic, model, hybrid.")
        if model is not None:
            self.config.orchestrator.model = ModelRef.parse(model).key
        if fallback_model is not None:
            self.config.orchestrator.fallback_model = ModelRef.parse(fallback_model).key
        if mode is not None:
            self.config.orchestrator.mode = mode
        if fallback is not None:
            self.config.orchestrator.fallback = fallback
        if temperature is not None:
            self.config.orchestrator.temperature = float(temperature)
        if require_validated_plan is not None:
            self.config.orchestrator.require_validated_plan = bool(require_validated_plan)
        if max_repairs is not None:
            self.config.orchestrator.max_repairs = int(max_repairs)
        if allow_prompt_summary_only is not None:
            self.config.orchestrator.allow_prompt_summary_only = bool(allow_prompt_summary_only)
        if persist:
            write_orchestrator_settings(
                self.config.root,
                mode=self.config.orchestrator.mode,
                model=self.config.orchestrator.model,
                fallback_model=self.config.orchestrator.fallback_model,
                fallback=self.config.orchestrator.fallback,
                temperature=self.config.orchestrator.temperature,
                require_validated_plan=self.config.orchestrator.require_validated_plan,
                max_repairs=self.config.orchestrator.max_repairs,
                allow_prompt_summary_only=self.config.orchestrator.allow_prompt_summary_only,
            )
            self.config = CrupierConfig.from_toml(self.config.root)
            self.registry.config = self.config
            self.models._config = self.config
            self.policy.config = self.config
            self.executor.config = self.config
        self.planner = RoutePlanner(self.config, orchestrator=self._build_orchestrator())
        return self

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
        cards, provider_exclusions, provider_filters = self._filter_operational_candidates(request, cards, dry_run=dry_run)
        cards, circuit_exclusions, circuit_filters = self._filter_circuit_breaker_candidates(cards)
        provider_exclusions.extend(circuit_exclusions)
        provider_filters.extend(circuit_filters)
        policy_result = self.policy.filter_candidates(request, cards)
        policy_result.excluded.extend(provider_exclusions)
        for filter_name in provider_filters:
            if filter_name not in policy_result.filters_applied:
                policy_result.filters_applied.append(filter_name)
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

    def _filter_operational_candidates(
        self,
        request: RequestEnvelope,
        cards: list[CapabilityCard],
        *,
        dry_run: bool,
    ) -> tuple[list[CapabilityCard], list[Exclusion], list[str]]:
        explicit = "require_operational_providers" in request.constraints
        require_operational = bool(
            request.constraints.get(
                "require_operational_providers",
                self.config.routing.require_operational_providers,
            )
        )
        if dry_run and not explicit:
            require_operational = False
        if not require_operational:
            return cards, [], []

        allowed: list[CapabilityCard] = []
        excluded: list[Exclusion] = []
        filters: set[str] = set()
        for card in cards:
            key = card.model_ref.key
            provider = card.model_ref.provider
            visible_models, error = self._provider_visible_models(provider)
            if error is not None:
                excluded.append(Exclusion(key, error))
                filters.add("provider_operational")
                continue
            if visible_models is not None and key not in visible_models:
                excluded.append(Exclusion(key, "model is not visible to the configured provider API key"))
                filters.add("provider_model_visibility")
                continue
            allowed.append(card)

        if not allowed:
            reasons = "; ".join(f"{item.model}: {item.reason}" for item in excluded)
            raise CrupierPolicyError(f"No models remain after provider operational checks. {reasons}")
        return allowed, excluded, sorted(filters)

    def _filter_circuit_breaker_candidates(
        self,
        cards: list[CapabilityCard],
    ) -> tuple[list[CapabilityCard], list[Exclusion], list[str]]:
        allowed: list[CapabilityCard] = []
        excluded: list[Exclusion] = []
        for card in cards:
            reason = self.executor.provider_circuit_open_reason(card.model_ref.provider)
            if reason is None:
                allowed.append(card)
                continue
            excluded.append(Exclusion(card.model_ref.key, reason))
        if not excluded:
            return cards, [], []
        if not allowed:
            return cards, [], []
        return allowed, excluded, ["provider_circuit_breaker"]

    def _provider_visible_models(self, provider: str) -> tuple[set[str] | None, str | None]:
        cached = self._provider_visibility_cache.get(provider)
        if cached is not None:
            return cached
        adapter = self.adapters.get(provider)
        if adapter is None:
            result = (None, f"provider {provider!r} has no configured adapter")
            self._provider_visibility_cache[provider] = result
            return result
        list_models = getattr(adapter, "list_models", None)
        if not callable(list_models):
            result = (None, None)
            self._provider_visibility_cache[provider] = result
            return result
        try:
            models = list_models()
        except (CrupierProviderAuthError, CrupierProviderRateLimitError, CrupierProviderUnavailableError) as exc:
            result = (None, f"provider {provider!r} is not operational with the configured API key: {exc}")
            self._provider_visibility_cache[provider] = result
            return result
        except Exception as exc:  # noqa: BLE001 - provider SDK boundaries vary
            result = (None, f"provider {provider!r} model discovery failed: {exc}")
            self._provider_visibility_cache[provider] = result
            return result
        result = ({ModelRef.parse(model.model_ref).key for model in models}, None)
        self._provider_visibility_cache[provider] = result
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
        warnings: list[str] = []
        provider_models = self.models.discover(
            provider=provider,
            skip_unavailable=provider is None,
            warnings=warnings,
        )
        discovered_providers = sorted({model.provider for model in provider_models})
        return self.registry.update_from_provider_models(
            provider_models,
            dry_run=dry_run,
            provider=provider,
            discovered_providers=discovered_providers,
            warnings=warnings,
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
