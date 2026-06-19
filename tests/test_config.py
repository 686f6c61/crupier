import os

from crupier.config import OLLAMA_CLOUD_HOST, CrupierConfig, write_default_project, write_models_allow


def test_write_default_project_creates_config_and_dirs(tmp_path):
    toml_path = write_default_project(tmp_path)

    assert toml_path.exists()
    assert (tmp_path / ".env.example").read_text(encoding="utf-8").startswith("OPENAI_API_KEY=")
    gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert "!.env.example" in gitignore
    assert ".crupier/traces/" in gitignore
    assert (tmp_path / ".crupier" / "registry" / "capability-cards").is_dir()
    assert (tmp_path / ".crupier" / "registry" / "snapshots").is_dir()

    config = CrupierConfig.from_toml(tmp_path)
    assert config.project.default_profile == "agentic"
    assert config.providers["openai"].enabled is True
    assert config.providers["ollama"].host == OLLAMA_CLOUD_HOST
    assert "OLLAMA_HOST=https://ollama.com/api" in (tmp_path / ".env.example").read_text(encoding="utf-8")


def test_write_default_project_preserves_existing_gitignore_entries(tmp_path):
    (tmp_path / ".gitignore").write_text("node_modules/\n.env\n", encoding="utf-8")

    write_default_project(tmp_path)

    lines = (tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert lines.count(".env") == 1
    assert "node_modules/" in lines
    assert "!.env.example" in lines


def test_config_loads_profiles_and_models():
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "x", "default_profile": "private"},
            "providers": {"ollama": {"enabled": True, "host": "http://localhost:11434"}},
            "models": {"allow": ["ollama:qwen3.5:122b"]},
            "profiles": {"private": {"prefer": ["local"], "strategy": "local_first"}},
        }
    )

    assert config.project.name == "x"
    assert config.providers["ollama"].host == "http://localhost:11434"
    assert config.profiles["private"].strategy == "local_first"


def test_write_models_allow_replaces_models_section(tmp_path):
    write_default_project(tmp_path)

    write_models_allow(tmp_path, ["claude:claude-opus-4-8", "ollama:gpt-oss:120b"], replace=True)

    config = CrupierConfig.from_toml(tmp_path)
    assert config.models.allow == ["anthropic:claude-opus-4-8", "ollama:gpt-oss:120b"]


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
    monkeypatch.setenv("OLLAMA_HOST", "http://already-set:11434")

    config = CrupierConfig.from_toml(tmp_path)

    assert config.providers["ollama"].host == "http://already-set:11434"
    assert os.environ["OPENAI_API_KEY"] == "from-dotenv"
    assert os.environ["ANTHROPIC_API_KEY"] == "quoted-value"
