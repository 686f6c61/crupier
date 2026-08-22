import os
import subprocess

import pytest

from crupier import Crupier
from crupier.config import (
    DEFAULT_GITIGNORE_ENTRIES,
    OLLAMA_CLOUD_HOST,
    OPENROUTER_DEFAULT_HOST,
    CrupierConfig,
    write_default_project,
    write_models_allow,
    write_orchestrator_settings,
)
from crupier.errors import CrupierConfigError


def test_write_default_project_creates_config_and_dirs(tmp_path):
    toml_path = write_default_project(tmp_path)

    assert toml_path.exists()
    assert (tmp_path / ".env.example").read_text(encoding="utf-8").startswith("OPENAI_API_KEY=")
    gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert "!.env.example" in gitignore
    assert ".crupier/traces/" in gitignore
    assert ".crupier/evals/live-routing-validation.json" in gitignore
    assert ".crupier/evals/live-operations-validation.json" in gitignore
    assert (tmp_path / ".crupier" / "registry" / "capability-cards").is_dir()
    assert (tmp_path / ".crupier" / "registry" / "snapshots").is_dir()

    config = CrupierConfig.from_toml(tmp_path)
    assert config.project.default_profile == "agentic"
    assert config.providers["openai"].enabled is True
    assert config.providers["ollama"].host == OLLAMA_CLOUD_HOST
    assert config.providers["openrouter"].enabled is False
    assert config.providers["openrouter"].mode == "byok"
    assert config.providers["openrouter"].host == OPENROUTER_DEFAULT_HOST
    assert config.routing.max_provider_retries == 1
    assert config.routing.retry_backoff_seconds == 0.2
    assert config.routing.require_operational_providers is True
    assert config.orchestrator.mode == "model"
    assert config.orchestrator.model == "openai:gpt-5.4-mini"
    assert config.orchestrator.fallback == "deterministic"
    env_example = (tmp_path / ".env.example").read_text(encoding="utf-8")
    assert "OLLAMA_HOST=https://ollama.com/api" in env_example
    assert "OPENROUTER_API_KEY=" in env_example


def test_write_default_project_preserves_existing_gitignore_entries(tmp_path):
    (tmp_path / ".gitignore").write_text("node_modules/\n.env\n", encoding="utf-8")

    write_default_project(tmp_path)

    lines = (tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert lines.count(".env") == 1
    assert "node_modules/" in lines
    assert "!.env.example" in lines


def test_init_ignores_crupier_backlog(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    write_default_project(tmp_path)
    backlog = tmp_path / ".crupier" / "backlog" / "READY.jsonl"
    backlog.parent.mkdir(parents=True)
    backlog.write_text("{}\n", encoding="utf-8")

    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", ".crupier/backlog/READY.jsonl"], cwd=tmp_path, check=False
    )

    assert ignored.returncode == 0


def test_init_ignores_crupier_state_and_snapshots(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    write_default_project(tmp_path)
    state = tmp_path / ".crupier" / "state.sqlite3-wal"
    snapshot = tmp_path / ".crupier" / "registry" / "snapshots" / "baseline.json"
    state.write_bytes(b"")
    snapshot.write_text("{}\n", encoding="utf-8")

    for relative in (".crupier/state.sqlite3-wal", ".crupier/registry/snapshots/baseline.json"):
        ignored = subprocess.run(["git", "check-ignore", "--quiet", relative], cwd=tmp_path, check=False)
        assert ignored.returncode == 0


def test_default_gitignore_covers_profiles_dir():
    assert ".crupier/profiles/" in DEFAULT_GITIGNORE_ENTRIES


def test_config_loads_profiles_and_models():
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "x", "default_profile": "private"},
            "providers": {"ollama": {"enabled": True, "host": "http://localhost:11434"}},
            "models": {"allow": ["ollama:qwen3.5:122b"]},
            "scoring": {
                "quality_weight": {"frontier": 12},
                "skill_fit_min_score": 7,
                "human_feedback_weight": 2.5,
            },
            "profiles": {"private": {"prefer": ["local"], "strategy": "local_first"}},
        }
    )

    assert config.project.name == "x"
    assert config.providers["ollama"].host == "http://localhost:11434"
    assert config.profiles["private"].strategy == "local_first"
    assert config.scoring.quality_weight["frontier"] == 12
    assert config.scoring.quality_weight["strong"] == 2
    assert config.scoring.skill_fit_min_score == 7
    assert config.scoring.human_feedback_weight == 2.5


def test_config_loads_shared_profile_files(tmp_path):
    write_default_project(tmp_path)
    profiles_dir = tmp_path / ".crupier" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "legal.toml").write_text(
        """
name = "legal"
prefer = ["accuracy", "citations"]
strategy = "critique_repair"
review_level = "strict"
""",
        encoding="utf-8",
    )
    (profiles_dir / "agentic.json").write_text(
        '{"prefer":["tool_use","low_latency"],"strategy":"single","owner":"platform"}',
        encoding="utf-8",
    )

    config = CrupierConfig.from_toml(tmp_path)

    assert config.profiles["legal"].strategy == "critique_repair"
    assert config.profiles["legal"].options["review_level"] == "strict"
    assert config.profiles["agentic"].strategy == "single"
    assert config.profiles["agentic"].options["owner"] == "platform"


def test_write_models_allow_replaces_models_section(tmp_path):
    write_default_project(tmp_path)

    write_models_allow(tmp_path, ["claude:claude-opus-4-8", "ollama:gpt-oss:120b"], replace=True)

    config = CrupierConfig.from_toml(tmp_path)
    assert config.models.allow == ["anthropic:claude-opus-4-8", "ollama:gpt-oss:120b"]


def test_write_orchestrator_settings_and_sdk_configuration(tmp_path):
    write_default_project(tmp_path)

    write_orchestrator_settings(
        tmp_path,
        mode="model",
        model="ollama:glm-5.2",
        fallback_model="anthropic:claude-opus-4-8",
        temperature=0.1,
        fallback="deterministic",
        require_validated_plan=False,
        max_repairs=3,
        candidate_limit=8,
        allow_prompt_summary_only=False,
    )

    config = CrupierConfig.from_toml(tmp_path)
    assert config.orchestrator.mode == "model"
    assert config.orchestrator.model == "ollama:glm-5.2"
    assert config.orchestrator.fallback_model == "anthropic:claude-opus-4-8"
    assert config.orchestrator.temperature == 0.1
    assert config.orchestrator.require_validated_plan is False
    assert config.orchestrator.max_repairs == 3
    assert config.orchestrator.candidate_limit == 8
    assert config.orchestrator.allow_prompt_summary_only is False

    client = Crupier.from_project(tmp_path)
    client.update_orchestrator(
        mode="hybrid",
        model="anthropic:claude-opus-4-8",
        max_repairs=2,
        candidate_limit=7,
        allow_prompt_summary_only=True,
    )
    assert client.config.orchestrator.mode == "hybrid"
    assert client.config.orchestrator.model == "anthropic:claude-opus-4-8"
    assert client.config.orchestrator.max_repairs == 2
    assert client.config.orchestrator.candidate_limit == 7
    assert client.config.orchestrator.allow_prompt_summary_only is True
    assert client.planner.config.orchestrator.model == "anthropic:claude-opus-4-8"


def test_update_orchestrator_persists_and_rebuilds_components(tmp_path):
    write_default_project(tmp_path)
    client = Crupier.from_project(tmp_path)
    previous_planner = client.planner

    returned = client.update_orchestrator(
        mode="model",
        model="openai:gpt-5.5",
        persist=True,
    )

    persisted = CrupierConfig.from_toml(tmp_path)
    assert returned is client
    assert persisted.orchestrator.mode == "model"
    assert persisted.orchestrator.model == "openai:gpt-5.5"
    assert client.config.orchestrator.mode == "model"
    assert client.config.orchestrator.model == "openai:gpt-5.5"
    assert client.planner is not previous_planner
    assert client.planner.config is client.config
    assert client.registry.config is client.config
    assert client.models._config is client.config
    assert client.policy.config is client.config
    assert client.executor.config is client.config


def test_from_project_uses_environment_when_no_file(tmp_path, monkeypatch):
    project = tmp_path / "environment-project"
    project.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CRUPIER_PROJECT", str(project))

    client = Crupier.from_project()

    assert client.config == CrupierConfig(root=project.resolve())
    assert client.config.root == project.resolve()


def test_from_project_explicit_path_wins_over_environment(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    environment = tmp_path / "environment"
    write_default_project(explicit)
    environment.mkdir()
    monkeypatch.setenv("CRUPIER_PROJECT", str(environment))

    client = Crupier.from_project(explicit)

    assert client.config.root == explicit.resolve()


def test_configure_orchestrator_delegates_with_deprecation_warning(tmp_path):
    write_default_project(tmp_path)
    client = Crupier.from_project(tmp_path)

    with pytest.warns(DeprecationWarning, match="update_orchestrator"):
        returned = client.configure_orchestrator(mode="deterministic")

    assert returned is client
    assert client.config.orchestrator.mode == "deterministic"


def test_orchestrator_candidate_limit_is_validated_before_mutation_or_write(tmp_path):
    write_default_project(tmp_path)
    original = (tmp_path / "crupier.toml").read_text(encoding="utf-8")

    try:
        write_orchestrator_settings(tmp_path, candidate_limit=1)
    except CrupierConfigError as exc:
        assert "candidate_limit" in str(exc)
    else:
        raise AssertionError("invalid persisted candidate limit must fail")
    assert (tmp_path / "crupier.toml").read_text(encoding="utf-8") == original

    client = Crupier.from_project(tmp_path)
    previous = client.config.orchestrator.candidate_limit
    try:
        client.update_orchestrator(candidate_limit=33)
    except CrupierConfigError as exc:
        assert "candidate_limit" in str(exc)
    else:
        raise AssertionError("invalid SDK candidate limit must fail")
    assert client.config.orchestrator.candidate_limit == previous


def test_from_toml_loads_local_dotenv_without_overriding_existing_env(tmp_path, monkeypatch):
    (tmp_path / "crupier.toml").write_text(
        """
[providers.openai]
enabled = true
env_key = "OPENAI_API_KEY"

[providers.ollama]
enabled = true
host = "http://localhost:11434"
env_key = "OLLAMA_API_KEY"
""",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        """
OPENAI_API_KEY=from-dotenv
ANTHROPIC_API_KEY='quoted-value'
OLLAMA_HOST=https://ollama.com/api
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

    config = CrupierConfig.from_toml(tmp_path)

    assert config.providers["ollama"].host == "http://127.0.0.1:11434"
    assert config.providers["openai"].env_value("OPENAI_API_KEY") == "from-dotenv"
    assert config.providers["openai"].env_value("ANTHROPIC_API_KEY") == "quoted-value"
    assert "OPENAI_API_KEY" not in os.environ
    assert "ANTHROPIC_API_KEY" not in os.environ
