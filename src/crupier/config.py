"""Project configuration loading and defaults."""

from __future__ import annotations

import json
import os
import tomllib
import warnings
from dataclasses import dataclass, field
from ipaddress import ip_address
from math import isfinite
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .errors import CrupierConfigError
from .models import ModelRef

OLLAMA_CLOUD_HOST = "https://ollama.com/api"
OPENROUTER_DEFAULT_HOST = "https://openrouter.ai/api/v1"
INFERENCE_DEFAULT_HOST = "http://127.0.0.1:8000/v1"
NAN_DEFAULT_HOST = "https://api.nan.builders/v1"
CANONICAL_PROVIDER_HOSTNAMES = {
    "openai": frozenset({"api.openai.com"}),
    "anthropic": frozenset({"api.anthropic.com"}),
    "openrouter": frozenset({"openrouter.ai"}),
    "ollama": frozenset({"ollama.com"}),
    "nan": frozenset({"api.nan.builders"}),
}
CANONICAL_ENV_KEYS = frozenset(
    {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "OLLAMA_API_KEY",
        "NAN_API_KEY",
    }
)
DEFAULT_CREDENTIAL_ENV_KEYS = CANONICAL_ENV_KEYS | {"INFERENCE_API_KEY"}
_ROUTE_STRATEGIES = {
    "single",
    "fallback",
    "cascade",
    "panel",
    "fusion",
    "critique_repair",
    "local_first",
    "delegate",
    "orchestrated",
}


@dataclass(slots=True)
class ProjectSettings:
    name: str = "crupier-project"
    default_profile: str = "agentic"


@dataclass(slots=True)
class LoggingSettings:
    mode: str = "metadata"
    persist_traces: bool = False
    store_prompts: bool = False
    store_responses: bool = False
    redact_secrets: bool = True
    ttl_days: int | None = None


@dataclass(slots=True)
class ProviderSettings:
    enabled: bool = False
    env_key: str | None = None
    host: str | None = None
    mode: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    env_values: dict[str, str] = field(default_factory=dict, repr=False)

    def env_value(self, key: str) -> str | None:
        """Devuelve primero una credencial exportada y después la del `.env` local."""

        return os.environ.get(key) or self.env_values.get(key)


@dataclass(slots=True)
class ModelSettings:
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RoutingSettings:
    default_strategy: str = "orchestrated"
    allow_fusion: bool = True
    allow_parallel: bool = True
    allow_latest_aliases: bool = False
    allow_preview_models: bool = False
    max_cost_per_request_usd: float | None = None
    max_latency_ms: int | None = 120000
    max_depth: int = 8
    max_calls: int = 40
    max_tool_rounds: int = 3
    max_tool_calls_per_round: int = 8
    max_tool_result_chars: int = 50_000
    max_provider_retries: int = 1
    retry_backoff_seconds: float = 0.2
    retry_jitter_seconds: float = 0.0
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_cooldown_seconds: float = 60.0
    require_operational_providers: bool = True


@dataclass(slots=True)
class OrchestratorSettings:
    mode: str = "deterministic"
    model: str | None = None
    fallback_model: str | None = None
    fallback: str = "deterministic"
    temperature: float = 0.0
    require_validated_plan: bool = True
    max_repairs: int = 1
    candidate_limit: int = 6
    allow_prompt_summary_only: bool = False


@dataclass(slots=True)
class ScoringSettings:
    quality_weight: dict[str, float] = field(
        default_factory=lambda: {"unknown": 0.0, "strong": 2.0, "frontier": 4.0}
    )
    cost_weight: dict[str, float] = field(
        default_factory=lambda: {"unknown": 0.0, "low": 4.0, "medium": 2.0, "high": 0.0}
    )
    latency_weight: dict[str, float] = field(
        default_factory=lambda: {"unknown": 0.0, "fast": 4.0, "medium": 2.0, "slow": 0.0}
    )
    profile_preference_weight: float = 3.0
    task_signal_weight: float = 2.0
    skill_fit_min_score: float = 6.0
    skill_fit_baseline: float = 5.0
    skill_fit_multiplier: float = 0.8
    skill_fit_cap: float = 12.0
    cheap_mode_cost_multiplier: float = 2.0
    fast_mode_latency_multiplier: float = 2.0
    private_mode_ollama_bonus: float = 10.0
    verified_capability_weight: float = 6.0
    inferred_capability_weight: float = 2.0
    failed_capability_penalty: float = -20.0
    local_eval_weight: float = 1.0
    human_feedback_weight: float = 1.0
    deprecation_penalty: float = -100.0
    routing_status_penalty: float = -6.0
    opt_in_penalty: float = -4.0
    cheap_high_cost_penalty: float = -4.0
    fast_latency_penalty: float = -3.0
    preview_stability_penalty: float = -5.0
    budget_over_penalty: float = -30.0
    budget_comfort_bonus: float = 3.0
    budget_within_bonus: float = 1.0


@dataclass(slots=True)
class PromotionSettings:
    action: str = "recommend"
    min_samples: int = 200
    max_error_rate: float = 0.02
    max_error_rate_delta: float = 0.02
    min_quality_delta: float = 0.0
    quality_check: str = "quality_delta"
    require_quality_evaluator: bool = True
    max_cost_ratio: float | None = 1.25
    max_p95_latency_ratio: float | None = 1.20
    confidence: float = 0.95


@dataclass(slots=True)
class ExperimentSettings:
    name: str
    enabled: bool = True
    traffic: str = "shadow"
    sample_rate: float = 0.0
    execution: str = "plan_only"
    candidate_models: list[str] = field(default_factory=list)
    candidate_strategy: str | None = None
    candidate_constraints: dict[str, Any] = field(default_factory=dict)
    sticky_by: str = "session_id"
    max_shadow_cost_usd: float | None = None
    max_concurrency: int = 2
    store_outputs: bool = False
    allow_side_effecting_tools: bool = False
    promotion: PromotionSettings = field(default_factory=PromotionSettings)


@dataclass(slots=True)
class PolicyRule:
    name: str
    effect: str
    reason: str = ""
    modes: list[str] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PolicySettings:
    rules: list[PolicyRule] = field(default_factory=list)


@dataclass(slots=True)
class ProfileSettings:
    name: str
    prefer: list[str] = field(default_factory=list)
    strategy: str = "orchestrated"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CrupierConfig:
    project: ProjectSettings = field(default_factory=ProjectSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    providers: dict[str, ProviderSettings] = field(default_factory=dict)
    models: ModelSettings = field(default_factory=ModelSettings)
    routing: RoutingSettings = field(default_factory=RoutingSettings)
    orchestrator: OrchestratorSettings = field(default_factory=OrchestratorSettings)
    scoring: ScoringSettings = field(default_factory=ScoringSettings)
    experiments: dict[str, ExperimentSettings] = field(default_factory=dict)
    policy: PolicySettings = field(default_factory=PolicySettings)
    profiles: dict[str, ProfileSettings] = field(default_factory=dict)
    root: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_toml(cls, path: str | Path) -> CrupierConfig:
        toml_path = Path(path)
        if toml_path.is_dir():
            toml_path = toml_path / "crupier.toml"
        if not toml_path.exists():
            raise CrupierConfigError(
                f"No crupier.toml found at {toml_path}. Run `crupier init` first or pass a config explicitly."
            )
        with toml_path.open("rb") as handle:
            data = tomllib.load(handle)
        config = cls.from_dict(data)
        config.root = toml_path.parent.resolve()
        load_profile_files(config)
        allowed_env_keys = DEFAULT_CREDENTIAL_ENV_KEYS | {
            settings.env_key for settings in config.providers.values() if settings.env_key
        }
        local_env = load_env_file(config.root, allowed_keys=allowed_env_keys)
        for settings in config.providers.values():
            settings.env_values = local_env.copy()
        apply_env_overrides(config)
        config.validate()
        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrupierConfig:
        if not isinstance(data, dict):
            raise CrupierConfigError("Crupier configuration must be an object/table.")
        try:
            providers: dict[str, ProviderSettings] = {}
            for name, provider_data in data.get("providers", {}).items():
                if not isinstance(provider_data, dict):
                    raise CrupierConfigError(f"Provider {name!r} configuration must be a table/object.")
                known = {"enabled", "env_key", "host", "mode"}
                providers[name] = ProviderSettings(
                    enabled=bool(provider_data.get("enabled", False)),
                    env_key=provider_data.get("env_key"),
                    host=provider_data.get("host"),
                    mode=provider_data.get("mode"),
                    options={key: value for key, value in provider_data.items() if key not in known},
                )

            profiles: dict[str, ProfileSettings] = {}
            for name, profile_data in data.get("profiles", {}).items():
                if not isinstance(profile_data, dict):
                    raise CrupierConfigError(f"Profile {name!r} configuration must be a table/object.")
                known = {"prefer", "strategy"}
                profiles[name] = ProfileSettings(
                    name=name,
                    prefer=list(profile_data.get("prefer", [])),
                    strategy=profile_data.get("strategy", "orchestrated"),
                    options={key: value for key, value in profile_data.items() if key not in known},
                )

            experiments = _experiment_settings_from_dict(data.get("experiments", {}))
            config = cls(
                project=ProjectSettings(**data.get("project", {})),
                logging=LoggingSettings(**data.get("logging", {})),
                providers=providers,
                models=ModelSettings(**data.get("models", {})),
                routing=RoutingSettings(**data.get("routing", {})),
                orchestrator=OrchestratorSettings(**data.get("orchestrator", {})),
                scoring=_scoring_settings_from_dict(data.get("scoring", {})),
                experiments=experiments,
                policy=_policy_settings_from_dict(data.get("policy", {})),
                profiles=profiles,
            )
        except CrupierConfigError:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise CrupierConfigError(f"Invalid Crupier configuration: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.models.allow, list) or not isinstance(self.models.deny, list):
            raise CrupierConfigError("[models].allow and [models].deny must be arrays of provider:model strings.")
        for model in [*self.models.allow, *self.models.deny]:
            try:
                ModelRef.parse(str(model))
            except ValueError as exc:
                raise CrupierConfigError(str(exc)) from exc

        if self.routing.default_strategy not in _ROUTE_STRATEGIES:
            raise CrupierConfigError(f"Unsupported routing.default_strategy {self.routing.default_strategy!r}.")
        _require_int_at_least("routing.max_calls", self.routing.max_calls, 1)
        _require_int_at_least("routing.max_depth", self.routing.max_depth, 0)
        _require_int_at_least("routing.max_tool_rounds", self.routing.max_tool_rounds, 1)
        _require_int_at_least("routing.max_tool_calls_per_round", self.routing.max_tool_calls_per_round, 1)
        _require_int_at_least("routing.max_tool_result_chars", self.routing.max_tool_result_chars, 256)
        _require_int_at_least("routing.max_provider_retries", self.routing.max_provider_retries, 0)
        _require_int_at_least(
            "routing.circuit_breaker_failure_threshold",
            self.routing.circuit_breaker_failure_threshold,
            0,
        )
        for name, value, allow_zero in (
            ("routing.max_cost_per_request_usd", self.routing.max_cost_per_request_usd, True),
            ("routing.max_latency_ms", self.routing.max_latency_ms, False),
            ("routing.retry_backoff_seconds", self.routing.retry_backoff_seconds, True),
            ("routing.retry_jitter_seconds", self.routing.retry_jitter_seconds, True),
            ("routing.circuit_breaker_cooldown_seconds", self.routing.circuit_breaker_cooldown_seconds, True),
        ):
            _require_finite_number(name, value, allow_none=True, allow_zero=allow_zero)

        if self.orchestrator.mode not in {"deterministic", "model", "hybrid"}:
            raise CrupierConfigError(f"Unsupported orchestrator.mode {self.orchestrator.mode!r}.")
        if self.orchestrator.fallback not in {"deterministic", "error"}:
            raise CrupierConfigError("orchestrator.fallback must be 'deterministic' or 'error'.")
        _require_int_at_least("orchestrator.max_repairs", self.orchestrator.max_repairs, 0)
        _require_int_at_least("orchestrator.candidate_limit", self.orchestrator.candidate_limit, 2)
        if self.orchestrator.candidate_limit > 32:
            raise CrupierConfigError("orchestrator.candidate_limit must be at most 32.")
        _require_finite_number("orchestrator.temperature", self.orchestrator.temperature)
        for orchestrator_model in (self.orchestrator.model, self.orchestrator.fallback_model):
            if orchestrator_model:
                try:
                    ModelRef.parse(orchestrator_model)
                except ValueError as exc:
                    raise CrupierConfigError(str(exc)) from exc

        for name, experiment in self.experiments.items():
            if experiment.traffic not in {"shadow", "canary"}:
                raise CrupierConfigError(
                    f"experiments.{name}.traffic must be 'shadow' or 'canary'."
                )
            if experiment.execution not in {"plan_only", "sync", "async"}:
                raise CrupierConfigError(
                    f"experiments.{name}.execution must be plan_only, sync, or async."
                )
            if not 0.0 <= experiment.sample_rate <= 1.0:
                raise CrupierConfigError(
                    f"experiments.{name}.sample_rate must be between 0 and 1."
                )
            _require_int_at_least(
                f"experiments.{name}.max_concurrency",
                experiment.max_concurrency,
                1,
            )
            if experiment.max_concurrency > 64:
                raise CrupierConfigError(
                    f"experiments.{name}.max_concurrency must be at most 64."
                )
            for model in experiment.candidate_models:
                try:
                    ModelRef.parse(model)
                except ValueError as exc:
                    raise CrupierConfigError(str(exc)) from exc
            if experiment.candidate_strategy and experiment.candidate_strategy not in _ROUTE_STRATEGIES:
                raise CrupierConfigError(
                    f"experiments.{name}.candidate_strategy is unsupported."
                )
            promotion = experiment.promotion
            if promotion.action not in {"recommend", "auto"}:
                raise CrupierConfigError(
                    f"experiments.{name}.promotion.action must be recommend or auto."
                )
            _require_int_at_least(
                f"experiments.{name}.promotion.min_samples",
                promotion.min_samples,
                1,
            )
            for field_name in (
                "max_error_rate",
                "max_error_rate_delta",
                "confidence",
            ):
                value = float(getattr(promotion, field_name))
                if not 0.0 <= value <= 1.0:
                    raise CrupierConfigError(
                        f"experiments.{name}.promotion.{field_name} must be between 0 and 1."
                    )
            _require_finite_number(
                f"experiments.{name}.promotion.min_quality_delta",
                promotion.min_quality_delta,
            )
            if not promotion.quality_check.strip():
                raise CrupierConfigError(
                    f"experiments.{name}.promotion.quality_check cannot be empty."
                )
            for field_name in ("max_cost_ratio", "max_p95_latency_ratio"):
                _require_finite_number(
                    f"experiments.{name}.promotion.{field_name}",
                    getattr(promotion, field_name),
                    allow_none=True,
                    allow_zero=False,
                )

        for name, profile in self.profiles.items():
            if profile.strategy not in _ROUTE_STRATEGIES:
                raise CrupierConfigError(f"Profile {name!r} has unsupported strategy {profile.strategy!r}.")
        for rule in self.policy.rules:
            if rule.effect not in {"deny", "require_capability", "require_verified_capability"}:
                raise CrupierConfigError(f"Policy rule {rule.name!r} has unsupported effect {rule.effect!r}.")
        _validate_scoring(self.scoring)
        for name, settings in self.providers.items():
            if settings.enabled:
                validate_provider_endpoint(name, settings)

    @property
    def crupier_dir(self) -> Path:
        return self.root / ".crupier"

    @property
    def registry_dir(self) -> Path:
        return self.crupier_dir / "registry"

    @property
    def capability_cards_dir(self) -> Path:
        return self.registry_dir / "capability-cards"

    @property
    def registry_snapshots_dir(self) -> Path:
        return self.registry_dir / "snapshots"

    @property
    def traces_dir(self) -> Path:
        return self.crupier_dir / "traces"

    @property
    def evals_dir(self) -> Path:
        return self.crupier_dir / "evals"

    @property
    def feedback_dir(self) -> Path:
        return self.crupier_dir / "feedback"

    @property
    def audits_dir(self) -> Path:
        return self.crupier_dir / "audits"

    @property
    def profiles_dir(self) -> Path:
        return self.crupier_dir / "profiles"

    @property
    def state_db_path(self) -> Path:
        return self.crupier_dir / "state.sqlite3"

    def ensure_project_dirs(self) -> None:
        for directory in [
            self.crupier_dir,
            self.registry_dir,
            self.capability_cards_dir,
            self.registry_snapshots_dir,
            self.traces_dir,
            self.evals_dir,
            self.feedback_dir,
            self.audits_dir,
            self.profiles_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


def _scoring_settings_from_dict(data: dict[str, Any]) -> ScoringSettings:
    defaults = ScoringSettings()
    if not isinstance(data, dict):
        return defaults
    known_maps = {"quality_weight", "cost_weight", "latency_weight"}
    settings: dict[str, Any] = {}
    for key in known_maps:
        value = data.get(key)
        default_value = getattr(defaults, key)
        if isinstance(value, dict):
            settings[key] = _float_map(value, default_value)
    for field_name in ScoringSettings.__dataclass_fields__:
        if field_name in known_maps or field_name not in data:
            continue
        try:
            settings[field_name] = float(data[field_name])
        except (TypeError, ValueError):
            settings[field_name] = getattr(defaults, field_name)
    return ScoringSettings(**settings)


def _experiment_settings_from_dict(data: Any) -> dict[str, ExperimentSettings]:
    if not isinstance(data, dict):
        return {}
    experiments: dict[str, ExperimentSettings] = {}
    for name, raw in data.items():
        if not isinstance(raw, dict):
            raise CrupierConfigError(f"Experiment {name!r} must be a table/object.")
        promotion_raw = raw.get("promotion", raw.get("promote_after", {}))
        promotion = (
            PromotionSettings(**promotion_raw)
            if isinstance(promotion_raw, dict)
            else PromotionSettings()
        )
        percent = raw.get("percent")
        sample_rate = (
            float(percent) / 100.0
            if percent is not None and "sample_rate" not in raw
            else float(raw.get("sample_rate", 0.0))
        )
        known = {
            "enabled",
            "traffic",
            "sample_rate",
            "percent",
            "execution",
            "candidate_models",
            "allow",
            "candidate_strategy",
            "strategy",
            "candidate_constraints",
            "sticky_by",
            "max_shadow_cost_usd",
            "max_concurrency",
            "store_outputs",
            "allow_side_effecting_tools",
            "promotion",
            "promote_after",
        }
        unknown = sorted(set(raw).difference(known))
        if unknown:
            raise CrupierConfigError(
                f"Experiment {name!r} has unknown field(s): {', '.join(unknown)}."
            )
        candidate_models_raw = raw.get("candidate_models", raw.get("allow", [])) or []
        if isinstance(candidate_models_raw, str):
            candidate_models_raw = [candidate_models_raw]
        if not isinstance(candidate_models_raw, list):
            raise CrupierConfigError(
                f"Experiment {name!r} candidate_models must be an array."
            )
        experiments[str(name)] = ExperimentSettings(
            name=str(name),
            enabled=bool(raw.get("enabled", True)),
            traffic=str(raw.get("traffic", "shadow")),
            sample_rate=sample_rate,
            execution=str(raw.get("execution", "plan_only")),
            candidate_models=[
                ModelRef.parse(str(model)).key
                for model in candidate_models_raw
            ],
            candidate_strategy=raw.get("candidate_strategy", raw.get("strategy")),
            candidate_constraints=dict(raw.get("candidate_constraints", {})),
            sticky_by=str(raw.get("sticky_by", "session_id")),
            max_shadow_cost_usd=(
                float(raw["max_shadow_cost_usd"])
                if raw.get("max_shadow_cost_usd") is not None
                else None
            ),
            max_concurrency=int(raw.get("max_concurrency", 2)),
            store_outputs=bool(raw.get("store_outputs", False)),
            allow_side_effecting_tools=bool(raw.get("allow_side_effecting_tools", False)),
            promotion=promotion,
        )
    return experiments


def _policy_settings_from_dict(data: Any) -> PolicySettings:
    if data is None:
        raise CrupierConfigError("policy must be a table/object.")
    if not isinstance(data, dict):
        raise CrupierConfigError("policy must be a table/object.")
    if "rules" not in data:
        return PolicySettings()
    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list):
        raise CrupierConfigError("[policy].rules must be an array of rule objects.")
    rules: list[PolicyRule] = []
    for index, item in enumerate(raw_rules):
        if not isinstance(item, dict):
            raise CrupierConfigError(
                f"[policy].rules[{index}] must be a table/object, not {type(item).__name__}."
            )
        rules.append(_policy_rule_from_dict(item))
    return PolicySettings(rules=rules)


def _policy_rule_from_dict(data: dict[str, Any]) -> PolicyRule:
    known = {"name", "effect", "reason", "modes", "mode", "providers", "provider", "models", "model", "capabilities"}
    modes = _string_list(data.get("modes", data.get("mode")))
    providers = _string_list(data.get("providers", data.get("provider")))
    models = _string_list(data.get("models", data.get("model")))
    capabilities = _string_list(data.get("capabilities"))
    return PolicyRule(
        name=str(data.get("name") or data.get("effect") or "policy_rule"),
        effect=str(data.get("effect", "deny")),
        reason=str(data.get("reason", "")),
        modes=modes,
        providers=providers,
        models=models,
        capabilities=capabilities,
        options={key: value for key, value in data.items() if key not in known},
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _float_map(value: dict[str, Any], default: dict[str, float]) -> dict[str, float]:
    merged = dict(default)
    for key, raw in value.items():
        try:
            merged[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return merged


def _require_int_at_least(name: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool):
        raise CrupierConfigError(f"{name} must be an integer >= {minimum}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CrupierConfigError(f"{name} must be an integer >= {minimum}.") from exc
    if parsed != value or parsed < minimum:
        raise CrupierConfigError(f"{name} must be an integer >= {minimum}.")


def _require_finite_number(
    name: str,
    value: Any,
    *,
    allow_none: bool = False,
    allow_zero: bool = True,
) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool):
        raise CrupierConfigError(f"{name} must be a finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CrupierConfigError(f"{name} must be a finite number.") from exc
    if not isfinite(parsed) or parsed < 0 or (not allow_zero and parsed == 0):
        qualifier = "positive finite" if not allow_zero else "non-negative finite"
        raise CrupierConfigError(f"{name} must be a {qualifier} number.")


def _validate_scoring(scoring: ScoringSettings) -> None:
    for field_name in ScoringSettings.__dataclass_fields__:
        value = getattr(scoring, field_name)
        values = value.values() if isinstance(value, dict) else [value]
        for item in values:
            if isinstance(item, bool):
                raise CrupierConfigError(f"scoring.{field_name} must contain finite numbers.")
            try:
                parsed = float(item)
            except (TypeError, ValueError) as exc:
                raise CrupierConfigError(f"scoring.{field_name} must contain finite numbers.") from exc
            if not isfinite(parsed):
                raise CrupierConfigError(f"scoring.{field_name} must contain finite numbers.")


def host_is_loopback(host: str) -> bool:
    """Indica si *host* apunta a loopback (localhost, 127.0.0.1 o ::1)."""

    hostname = (urlparse(host).hostname or "").lower()
    if not hostname:
        return False
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def is_generic_inference_provider(provider: str, settings: ProviderSettings) -> bool:
    """Indica si el proveedor es un endpoint genérico compatible con OpenAI."""

    if provider == "inference":
        return True
    return settings.mode == "openai_compatible" and provider not in CANONICAL_PROVIDER_HOSTNAMES


def is_official_provider_host(provider: str, host: str | None) -> bool:
    """Indica si *host* es el API oficial o, en Ollama, un demonio local."""

    if not host:
        return True
    hostname = (urlparse(host).hostname or "").lower()
    if not hostname:
        return False
    official = CANONICAL_PROVIDER_HOSTNAMES.get(provider, frozenset())
    if hostname in official:
        return True
    return provider == "ollama" and host_is_loopback(host)


def validate_provider_endpoint(provider: str, settings: ProviderSettings) -> None:
    """Impide enviar credenciales canónicas a un host no oficial sin opt-in.

    Los endpoints genéricos de inferencia pueden usar un host propio, pero no
    reutilizar la clave de un proveedor canónico. Fuera de loopback el host
    tiene que ser HTTPS.
    """

    host = settings.host
    if is_generic_inference_provider(provider, settings):
        env_key = settings.env_key
        if env_key and env_key in CANONICAL_ENV_KEYS:
            raise CrupierConfigError(
                f"Provider {provider!r} cannot reuse the canonical credential {env_key!r}. "
                "Configure a dedicated env_key for this generic inference endpoint."
            )
        if host:
            _require_https_unless_loopback(host)
        return

    if host:
        _require_https_unless_loopback(host)
    if provider not in CANONICAL_PROVIDER_HOSTNAMES:
        return
    if is_official_provider_host(provider, host):
        return
    if settings.options.get("allow_custom_host") is True:
        return
    raise CrupierConfigError(
        f"Provider {provider!r} host {host!r} is not an official endpoint. "
        "Set allow_custom_host = true to send credentials to a custom host."
    )


def _require_https_unless_loopback(host: str) -> None:
    scheme = (urlparse(host).scheme or "").lower()
    if host_is_loopback(host):
        if scheme not in {"http", "https"}:
            raise CrupierConfigError(f"Provider host {host!r} must use http or https.")
        return
    if scheme != "https":
        raise CrupierConfigError(f"Provider host {host!r} must use HTTPS outside loopback.")


def ollama_is_local(config: CrupierConfig) -> bool:
    settings = config.providers.get("ollama")
    if settings is None:
        return False
    if settings.mode == "local":
        return True
    host = settings.host or os.environ.get("OLLAMA_HOST") or OLLAMA_CLOUD_HOST
    return host_is_loopback(host)


def load_profile_files(config: CrupierConfig) -> None:
    """Load optional shared profiles from .crupier/profiles/*.toml|*.json."""

    profiles_dir = config.profiles_dir
    if not profiles_dir.exists():
        return
    for path in sorted([*profiles_dir.glob("*.toml"), *profiles_dir.glob("*.json")]):
        try:
            if path.suffix == ".toml":
                with path.open("rb") as handle:
                    data = tomllib.load(handle)
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
            raise CrupierConfigError(f"Could not load profile file {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise CrupierConfigError(f"Profile file {path} must contain an object.")
        profile_data = data.get("profile", data)
        if not isinstance(profile_data, dict):
            raise CrupierConfigError(f"Profile file {path} must contain a profile object.")
        profile = _profile_from_data(path.stem, profile_data)
        config.profiles[profile.name] = profile


def _profile_from_data(default_name: str, data: dict[str, Any]) -> ProfileSettings:
    known = {"name", "prefer", "strategy"}
    name = str(data.get("name") or default_name)
    return ProfileSettings(
        name=name,
        prefer=list(data.get("prefer", [])),
        strategy=str(data.get("strategy", "orchestrated")),
        options={key: value for key, value in data.items() if key not in known},
    )


DEFAULT_TOML = """[project]
name = "crupier-project"
default_profile = "agentic"

[providers.openai]
enabled = true
env_key = "OPENAI_API_KEY"

[providers.anthropic]
enabled = false
env_key = "ANTHROPIC_API_KEY"

[providers.google]
enabled = false
env_key = "GOOGLE_API_KEY"

[providers.ollama]
enabled = false
host = "https://ollama.com/api"
env_key = "OLLAMA_API_KEY"

[providers.openrouter]
enabled = false
mode = "byok"
host = "https://openrouter.ai/api/v1"
env_key = "OPENROUTER_API_KEY"

[providers.inference]
enabled = false
mode = "openai_compatible"
host = "http://127.0.0.1:8000/v1"
auth = "none"

[models]
allow = ["openai:gpt-5.5", "openai:gpt-5.4-mini"]
deny = []

[routing]
default_strategy = "orchestrated"
allow_fusion = true
allow_parallel = true
allow_latest_aliases = false
allow_preview_models = false
max_cost_per_request_usd = 1.00
max_latency_ms = 120000
max_depth = 8
max_calls = 40
max_tool_rounds = 3
max_tool_calls_per_round = 8
max_tool_result_chars = 50000
max_provider_retries = 1
retry_backoff_seconds = 0.2
retry_jitter_seconds = 0
circuit_breaker_failure_threshold = 3
circuit_breaker_cooldown_seconds = 60
require_operational_providers = true

[orchestrator]
mode = "model"
model = "openai:gpt-5.4-mini"
fallback_model = "openai:gpt-5.4-mini"
fallback = "deterministic"
temperature = 0
require_validated_plan = true
max_repairs = 1
candidate_limit = 6
allow_prompt_summary_only = false

[scoring]
quality_weight = { unknown = 0, strong = 2, frontier = 4 }
cost_weight = { unknown = 0, low = 4, medium = 2, high = 0 }
latency_weight = { unknown = 0, fast = 4, medium = 2, slow = 0 }
profile_preference_weight = 3
task_signal_weight = 2
skill_fit_min_score = 6
skill_fit_baseline = 5
skill_fit_multiplier = 0.8
skill_fit_cap = 12
cheap_mode_cost_multiplier = 2
fast_mode_latency_multiplier = 2
private_mode_ollama_bonus = 10
verified_capability_weight = 6
inferred_capability_weight = 2
failed_capability_penalty = -20
local_eval_weight = 1
human_feedback_weight = 1
deprecation_penalty = -100
routing_status_penalty = -6
opt_in_penalty = -4
cheap_high_cost_penalty = -4
fast_latency_penalty = -3
preview_stability_penalty = -5
budget_over_penalty = -30
budget_comfort_bonus = 3
budget_within_bonus = 1

[experiments]

[policy]
rules = []

[logging]
mode = "metadata"
persist_traces = false
store_prompts = false
store_responses = false
redact_secrets = true

[profiles.agentic]
prefer = ["tool_use", "coding", "long_horizon", "reliability"]
strategy = "orchestrated"

[profiles.cheap]
prefer = ["low_cost"]
strategy = "orchestrated"

[profiles.fast]
prefer = ["low_latency"]
strategy = "orchestrated"

[profiles.private]
prefer = ["local", "zdr", "no_prompt_logging"]
strategy = "local_first"

[profiles.research]
prefer = ["consensus", "critique"]
strategy = "orchestrated"

[profiles.structured]
prefer = ["structured_output", "schema_validity"]
strategy = "orchestrated"
"""

DEFAULT_ENV_EXAMPLE = """OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
GEMINI_API_KEY=
OLLAMA_API_KEY=
OLLAMA_HOST=https://ollama.com/api
OPENROUTER_API_KEY=
INFERENCE_API_KEY=
"""

DEFAULT_GITIGNORE_ENTRIES = [
    ".env",
    ".env.*",
    "!.env.example",
    ".ruff_cache/",
    ".coverage",
    "coverage.xml",
    "htmlcov/",
    ".crupier/registry/models.json",
    ".crupier/registry/capability-cards/",
    ".crupier/traces/",
    ".crupier/audits/",
    ".crupier/code-comments/",
    ".crupier/evals/history/",
    ".crupier/evals/results/",
    ".crupier/evals/runs/",
    ".crupier/evals/live-routing-validation.json",
    ".crupier/evals/live-operations-validation.json",
    ".crupier/feedback/",
    ".crupier/handoffs/",
    ".crupier/packages/",
    ".crupier/test-assets/",
    ".crupier/audit-dist/",
]


def write_default_project(path: str | Path, *, force: bool = False) -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    toml_path = root / "crupier.toml"
    if toml_path.exists() and not force:
        raise CrupierConfigError(f"{toml_path} already exists. Pass force=True or use `crupier init --force`.")
    toml_path.write_text(DEFAULT_TOML, encoding="utf-8")
    env_example_path = root / ".env.example"
    if force or not env_example_path.exists():
        env_example_path.write_text(DEFAULT_ENV_EXAMPLE, encoding="utf-8")
    _ensure_gitignore_entries(root / ".gitignore", DEFAULT_GITIGNORE_ENTRIES)
    config = CrupierConfig.from_toml(toml_path)
    config.ensure_project_dirs()
    return toml_path


def _ensure_gitignore_entries(path: Path, entries: list[str]) -> None:
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    normalized = {line.strip() for line in existing}
    missing = [entry for entry in entries if entry not in normalized]
    if not missing:
        return
    lines = list(existing)
    if lines and lines[-1].strip():
        lines.append("")
    if "Crupier local artifacts" not in normalized:
        lines.append("# Crupier local artifacts")
    lines.extend(missing)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def load_env_file(root: str | Path, *, allowed_keys: set[str] | frozenset[str]) -> dict[str, str]:
    """Carga credenciales permitidas de un `.env` sin mutar el entorno del proceso."""

    path = Path(root) / ".env"
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        if key not in allowed_keys:
            warnings.warn(
                f"Se ha ignorado {key!r} del .env porque no es una credencial permitida.",
                UserWarning,
                stacklevel=2,
            )
            continue
        value = _env_value(value.strip())
        if key not in os.environ:
            loaded[key] = value
    return loaded


def apply_env_overrides(config: CrupierConfig) -> None:
    """Apply supported provider runtime overrides from environment variables."""

    ollama_host = os.environ.get("OLLAMA_HOST")
    if ollama_host and "ollama" in config.providers:
        config.providers["ollama"].host = ollama_host


def _env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def write_models_allow(path: str | Path, models: list[str], *, replace: bool = False) -> Path:
    from .models import ModelRef

    toml_path = Path(path)
    if toml_path.is_dir():
        toml_path = toml_path / "crupier.toml"
    if not toml_path.exists():
        raise CrupierConfigError(f"No crupier.toml found at {toml_path}.")

    config = CrupierConfig.from_toml(toml_path)
    existing = [] if replace else list(config.models.allow)
    merged: list[str] = []
    for model in [*existing, *models]:
        normalized = ModelRef.parse(model).key
        if normalized not in merged:
            merged.append(normalized)

    text = toml_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = None
    end = len(lines)
    for index, line in enumerate(lines):
        if line.strip() == "[models]":
            start = index
            continue
        if start is not None and index > start and line.strip().startswith("[") and line.strip().endswith("]"):
            end = index
            break

    deny = list(config.models.deny)
    section = ["[models]", "allow = ["]
    section.extend(f'  "{model}",' for model in merged)
    section.append("]")
    if deny:
        section.append("deny = [")
        section.extend(f'  "{ModelRef.parse(model).key}",' for model in deny)
        section.append("]")
    else:
        section.append("deny = []")

    if start is None:
        new_lines = [*lines, "", *section]
    else:
        new_lines = [*lines[:start], *section, *lines[end:]]
    toml_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
    return toml_path


def write_orchestrator_settings(
    path: str | Path,
    *,
    mode: str | None = None,
    model: str | None = None,
    fallback_model: str | None = None,
    temperature: float | None = None,
    fallback: str | None = None,
    require_validated_plan: bool | None = None,
    max_repairs: int | None = None,
    candidate_limit: int | None = None,
    allow_prompt_summary_only: bool | None = None,
) -> Path:
    from .models import ModelRef

    toml_path = Path(path)
    if toml_path.is_dir():
        toml_path = toml_path / "crupier.toml"
    if not toml_path.exists():
        raise CrupierConfigError(f"No crupier.toml found at {toml_path}.")

    config = CrupierConfig.from_toml(toml_path)
    if mode is not None and mode not in {"deterministic", "model", "hybrid"}:
        raise CrupierConfigError("orchestrator mode must be one of: deterministic, model, hybrid.")

    settings = {
        "mode": mode or config.orchestrator.mode,
        "model": ModelRef.parse(model).key if model else config.orchestrator.model,
        "fallback_model": ModelRef.parse(fallback_model).key if fallback_model else config.orchestrator.fallback_model,
        "fallback": fallback or config.orchestrator.fallback,
        "temperature": config.orchestrator.temperature if temperature is None else float(temperature),
        "require_validated_plan": config.orchestrator.require_validated_plan
        if require_validated_plan is None
        else bool(require_validated_plan),
        "max_repairs": config.orchestrator.max_repairs if max_repairs is None else int(max_repairs),
        "candidate_limit": config.orchestrator.candidate_limit if candidate_limit is None else int(candidate_limit),
        "allow_prompt_summary_only": config.orchestrator.allow_prompt_summary_only
        if allow_prompt_summary_only is None
        else bool(allow_prompt_summary_only),
    }
    for key, value in settings.items():
        setattr(config.orchestrator, key, value)
    config.validate()

    text = toml_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = None
    end = len(lines)
    for index, line in enumerate(lines):
        if line.strip() == "[orchestrator]":
            start = index
            continue
        if start is not None and index > start and line.strip().startswith("[") and line.strip().endswith("]"):
            end = index
            break

    section = ["[orchestrator]"]
    for key, value in settings.items():
        section.append(f"{key} = {_toml_value(value)}")

    if start is None:
        new_lines = [*lines, "", *section]
    else:
        new_lines = [*lines[:start], *section, *lines[end:]]
    toml_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
    return toml_path


def write_scoring_settings(path: str | Path, updates: dict[str, Any]) -> Path:
    toml_path = Path(path)
    if toml_path.is_dir():
        toml_path = toml_path / "crupier.toml"
    if not toml_path.exists():
        raise CrupierConfigError(f"No crupier.toml found at {toml_path}.")

    config = CrupierConfig.from_toml(toml_path)
    current = {
        field_name: getattr(config.scoring, field_name)
        for field_name in ScoringSettings.__dataclass_fields__
    }
    current.update(updates)

    text = toml_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = None
    end = len(lines)
    for index, line in enumerate(lines):
        if line.strip() == "[scoring]":
            start = index
            continue
        if start is not None and index > start and line.strip().startswith("[") and line.strip().endswith("]"):
            end = index
            break

    section = ["[scoring]"]
    for key, value in current.items():
        section.append(f"{key} = {_toml_value(value)}")

    if start is None:
        new_lines = [*lines, "", *section]
    else:
        new_lines = [*lines[:start], *section, *lines[end:]]
    toml_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
    return toml_path


def _toml_value(value: Any) -> str:
    if value is None:
        return '""'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, dict):
        items = ", ".join(f"{key} = {_toml_value(item)}" for key, item in value.items())
        return "{ " + items + " }"
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
