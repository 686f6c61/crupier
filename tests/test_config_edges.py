from __future__ import annotations

import math
from pathlib import Path

import pytest

import crupier.config as config_module
from crupier.config import (
    CrupierConfig,
    PolicyRule,
    ProfileSettings,
    ProviderSettings,
    ScoringSettings,
)
from crupier.errors import CrupierConfigError


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([], "object/table"),
        ({"providers": {"openai": True}}, "Provider 'openai'"),
        ({"profiles": {"agentic": True}}, "Profile 'agentic'"),
        ({"project": {"unknown": True}}, "Invalid Crupier configuration"),
    ],
)
def test_from_dict_rejects_invalid_top_level_shapes(data, message: str) -> None:
    with pytest.raises(CrupierConfigError, match=message):
        CrupierConfig.from_dict(data)


def _invalid_config() -> CrupierConfig:
    return CrupierConfig.from_dict({})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda cfg: setattr(cfg.models, "allow", "openai:gpt-5.5"), "must be arrays"),
        (lambda cfg: cfg.models.allow.append("missing-provider"), "provider:model"),
        (lambda cfg: setattr(cfg.routing, "default_strategy", "magic"), "Unsupported routing.default_strategy"),
        (lambda cfg: setattr(cfg.orchestrator, "mode", "magic"), "Unsupported orchestrator.mode"),
        (lambda cfg: setattr(cfg.orchestrator, "fallback", "magic"), "orchestrator.fallback"),
        (lambda cfg: setattr(cfg.orchestrator, "model", "missing-provider"), "provider:model"),
        (
            lambda cfg: cfg.profiles.update({"bad": ProfileSettings(name="bad", strategy="magic")}),
            "unsupported strategy",
        ),
        (
            lambda cfg: cfg.policy.rules.append(PolicyRule(name="bad", effect="allow")),
            "unsupported effect",
        ),
    ],
)
def test_config_validate_rejects_invalid_routes_models_profiles_and_policies(mutate, message: str) -> None:
    config = _invalid_config()
    mutate(config)

    with pytest.raises(CrupierConfigError, match=message):
        config.validate()


def test_scoring_and_policy_parsers_handle_invalid_values() -> None:
    defaults = config_module._scoring_settings_from_dict("invalid")
    parsed = config_module._scoring_settings_from_dict(
        {
            "quality_weight": {"frontier": "12", "bad": object()},
            "skill_fit_cap": "invalid",
        }
    )
    assert defaults == ScoringSettings()
    assert parsed.quality_weight["frontier"] == 12.0
    assert "bad" not in parsed.quality_weight
    assert parsed.skill_fit_cap == ScoringSettings().skill_fit_cap

    assert config_module._policy_settings_from_dict([]).rules == []
    assert config_module._policy_settings_from_dict({"rules": "invalid"}).rules == []
    settings = config_module._policy_settings_from_dict(
        {
            "rules": [
                "ignored",
                {
                    "effect": "deny",
                    "mode": "private",
                    "provider": "openai",
                    "model": ["openai:gpt-5.5"],
                    "capabilities": "tools",
                    "owner": "platform",
                },
            ]
        }
    )
    rule = settings.rules[0]
    assert rule.name == "deny"
    assert rule.modes == ["private"]
    assert rule.providers == ["openai"]
    assert rule.models == ["openai:gpt-5.5"]
    assert rule.capabilities == ["tools"]
    assert rule.options == {"owner": "platform"}


def test_numeric_validators_cover_bool_type_range_and_non_finite_values() -> None:
    for value in (True, "not-an-int", 0):
        with pytest.raises(CrupierConfigError):
            config_module._require_int_at_least("value", value, 1)

    for value in (True, object(), math.inf, -1, 0):
        with pytest.raises(CrupierConfigError):
            config_module._require_finite_number("value", value, allow_zero=False)

    for value in (True, object(), math.nan):
        scoring = ScoringSettings()
        scoring.profile_preference_weight = value
        with pytest.raises(CrupierConfigError):
            config_module._validate_scoring(scoring)


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        (None, False),
        (ProviderSettings(mode="local"), True),
        (ProviderSettings(host="http://localhost:11434"), True),
        (ProviderSettings(host="http://127.0.0.1:11434"), True),
        (ProviderSettings(host="https://ollama.com/api"), False),
    ],
)
def test_ollama_local_detection(settings: ProviderSettings | None, expected: bool) -> None:
    config = CrupierConfig()
    if settings is not None:
        config.providers["ollama"] = settings

    assert config_module.ollama_is_local(config) is expected


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("bad.json", "{", "Could not load profile"),
        ("list.json", "[]", "must contain an object"),
        ("nested.json", '{"profile": []}', "profile object"),
    ],
)
def test_profile_files_reject_corrupt_shapes(tmp_path: Path, filename: str, content: str, message: str) -> None:
    config = CrupierConfig(root=tmp_path)
    config.profiles_dir.mkdir(parents=True)
    (config.profiles_dir / filename).write_text(content, encoding="utf-8")

    with pytest.raises(CrupierConfigError, match=message):
        config_module.load_profile_files(config)


def test_default_project_force_and_gitignore_idempotence(tmp_path: Path) -> None:
    config_module.write_default_project(tmp_path)
    with pytest.raises(CrupierConfigError, match="already exists"):
        config_module.write_default_project(tmp_path)

    before = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    config_module._ensure_gitignore_entries(tmp_path / ".gitignore", config_module.DEFAULT_GITIGNORE_ENTRIES)
    assert (tmp_path / ".gitignore").read_text(encoding="utf-8") == before

    (tmp_path / ".env.example").write_text("PRESERVE=yes\n", encoding="utf-8")
    config_module.write_default_project(tmp_path, force=True)
    assert "OPENAI_API_KEY=" in (tmp_path / ".env.example").read_text(encoding="utf-8")


def test_env_loader_skips_invalid_names_and_preserves_exported_values(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "# comment\n"
        "NO_EQUALS\n"
        "1INVALID=value\n"
        "BAD-NAME=value\n"
        "EXPORTED=from-file\n"
        'QUOTED="hello"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("EXPORTED", "from-shell")
    monkeypatch.delenv("QUOTED", raising=False)

    loaded = config_module.load_env_file(tmp_path)

    assert loaded == {"QUOTED": "hello"}
    assert config_module.load_env_file(tmp_path / "missing") == {}


def test_model_allow_writer_handles_missing_file_appended_section_and_denylist(tmp_path: Path) -> None:
    with pytest.raises(CrupierConfigError, match="No crupier.toml"):
        config_module.write_models_allow(tmp_path, ["openai:gpt-5.5"])

    toml = tmp_path / "crupier.toml"
    toml.write_text(
        '[project]\nname = "demo"\n\n[models]\nallow = []\ndeny = ["openai:o3"]\n',
        encoding="utf-8",
    )
    config_module.write_models_allow(toml, ["openai:gpt-5.5"])
    text = toml.read_text(encoding="utf-8")
    assert 'deny = [\n  "openai:o3",\n]' in text

    no_section = tmp_path / "without-models.toml"
    no_section.write_text('[project]\nname = "demo"\n', encoding="utf-8")
    config_module.write_models_allow(no_section, ["anthropic:claude-sonnet-4-6"])
    assert "[models]" in no_section.read_text(encoding="utf-8")


def test_orchestrator_writer_handles_missing_file_invalid_mode_and_appended_section(tmp_path: Path) -> None:
    with pytest.raises(CrupierConfigError, match="No crupier.toml"):
        config_module.write_orchestrator_settings(tmp_path, mode="model")

    toml = tmp_path / "crupier.toml"
    toml.write_text('[project]\nname = "demo"\n', encoding="utf-8")
    with pytest.raises(CrupierConfigError, match="must be one of"):
        config_module.write_orchestrator_settings(toml, mode="magic")

    config_module.write_orchestrator_settings(toml, mode="hybrid", model="openai:gpt-5.5")
    assert "[orchestrator]" in toml.read_text(encoding="utf-8")


def test_scoring_writer_handles_missing_file_appended_section_and_toml_values(tmp_path: Path) -> None:
    with pytest.raises(CrupierConfigError, match="No crupier.toml"):
        config_module.write_scoring_settings(tmp_path, {"skill_fit_cap": 10})

    toml = tmp_path / "crupier.toml"
    toml.write_text('[project]\nname = "demo"\n', encoding="utf-8")
    config_module.write_scoring_settings(toml, {"quality_weight": {"frontier": 9}, "skill_fit_cap": 10})
    text = toml.read_text(encoding="utf-8")
    assert "[scoring]" in text
    assert "skill_fit_cap = 10" in text
    assert config_module._toml_value(None) == '""'
    assert config_module._toml_value(["one", 2]) == '["one", 2]'
