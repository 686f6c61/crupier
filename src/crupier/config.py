"""Project configuration loading and defaults."""

from __future__ import annotations

import tomllib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import CrupierConfigError

OLLAMA_CLOUD_HOST = "https://ollama.com/api"


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
    max_latency_ms: int | None = 30000
    max_depth: int = 8
    max_calls: int = 40


@dataclass(slots=True)
class OrchestratorSettings:
    mode: str = "deterministic"
    model: str | None = None
    fallback_model: str | None = None
    fallback: str = "deterministic"
    temperature: float = 0.0
    require_validated_plan: bool = True
    max_repairs: int = 1
    allow_prompt_summary_only: bool = True


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
    profiles: dict[str, ProfileSettings] = field(default_factory=dict)
    root: Path = field(default_factory=lambda: Path(".").resolve())

    @classmethod
    def from_toml(cls, path: str | Path) -> "CrupierConfig":
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
        load_env_file(config.root)
        apply_env_overrides(config)
        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrupierConfig":
        providers: dict[str, ProviderSettings] = {}
        for name, provider_data in data.get("providers", {}).items():
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
            known = {"prefer", "strategy"}
            profiles[name] = ProfileSettings(
                name=name,
                prefer=list(profile_data.get("prefer", [])),
                strategy=profile_data.get("strategy", "orchestrated"),
                options={key: value for key, value in profile_data.items() if key not in known},
            )

        return cls(
            project=ProjectSettings(**data.get("project", {})),
            logging=LoggingSettings(**data.get("logging", {})),
            providers=providers,
            models=ModelSettings(**data.get("models", {})),
            routing=RoutingSettings(**data.get("routing", {})),
            orchestrator=OrchestratorSettings(**data.get("orchestrator", {})),
            profiles=profiles,
        )

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
env_key = "OPENROUTER_API_KEY"

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
max_latency_ms = 30000
max_depth = 8
max_calls = 40

[orchestrator]
mode = "deterministic"
model = "openai:gpt-5.4-mini"
fallback_model = "openai:gpt-5.4-mini"
fallback = "deterministic"
temperature = 0
require_validated_plan = true
max_repairs = 1
allow_prompt_summary_only = true

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
strategy = "cascade"

[profiles.fast]
prefer = ["low_latency"]
strategy = "single"

[profiles.private]
prefer = ["local", "zdr", "no_prompt_logging"]
strategy = "local_first"

[profiles.research]
prefer = ["consensus", "critique"]
strategy = "fusion"

[profiles.structured]
prefer = ["structured_output", "schema_validity"]
strategy = "cascade"
"""

DEFAULT_ENV_EXAMPLE = """OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
GEMINI_API_KEY=
OLLAMA_API_KEY=
OLLAMA_HOST=https://ollama.com/api
"""

DEFAULT_GITIGNORE_ENTRIES = [
    ".env",
    ".env.*",
    "!.env.example",
    ".ruff_cache/",
    ".crupier/registry/models.json",
    ".crupier/registry/capability-cards/",
    ".crupier/traces/",
    ".crupier/audits/",
    ".crupier/code-comments/",
    ".crupier/evals/history/",
    ".crupier/evals/results/",
    ".crupier/evals/runs/",
    ".crupier/feedback/",
    ".crupier/handoffs/",
    ".crupier/packages/",
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


def load_env_file(root: str | Path) -> dict[str, str]:
    """Load a local .env file into os.environ without overwriting exported values."""

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
        value = _env_value(value.strip())
        if key not in os.environ:
            os.environ[key] = value
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
