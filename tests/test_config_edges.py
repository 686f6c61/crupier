from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

import crupier.config as config_module
from crupier.config import (
    CrupierConfig,
    ExperimentSettings,
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


def test_redact_secrets_false_is_rejected_instead_of_ignored() -> None:
    with pytest.raises(CrupierConfigError, match="logging.redact_secrets"):
        CrupierConfig.from_dict({"logging": {"redact_secrets": False}})


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
            "quality_weight": {"frontier": "12"},
            "skill_fit_cap": "invalid",
        }
    )
    assert defaults == ScoringSettings()
    assert parsed.quality_weight["frontier"] == 12.0
    assert parsed.skill_fit_cap == ScoringSettings().skill_fit_cap

    with pytest.raises(CrupierConfigError, match="policy must be a table"):
        config_module._policy_settings_from_dict([])
    with pytest.raises(CrupierConfigError, match="rules"):
        config_module._policy_settings_from_dict({"rules": "invalid"})
    settings = config_module._policy_settings_from_dict(
        {
            "rules": [
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


@pytest.mark.parametrize("value", ["mucho", [1], None])
def test_scoring_weight_rejects_non_numeric_value(value) -> None:
    with pytest.raises(CrupierConfigError, match=r"scoring\.cost_weight\.low"):
        CrupierConfig.from_dict({"scoring": {"cost_weight": {"low": value}}})


def test_scoring_weight_rejects_boolean_instead_of_coercing_it() -> None:
    with pytest.raises(CrupierConfigError, match=r"scoring\.cost_weight\.low.*True"):
        CrupierConfig.from_dict({"scoring": {"cost_weight": {"low": True}}})


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

    loaded = config_module.load_env_file(tmp_path, allowed_keys={"EXPORTED", "QUOTED"})

    assert loaded == {"QUOTED": "hello"}
    assert "QUOTED" not in os.environ
    assert config_module.load_env_file(tmp_path / "missing", allowed_keys=set()) == {}


def test_env_file_ignores_non_credential_variables(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "crupier.toml").write_text(
        "[providers.openai]\nenabled = true\nenv_key = \"OPENAI_API_KEY\"\n",
        encoding="utf-8",
    )
    blocked = {
        "HTTPS_PROXY": "http://attacker.invalid:8080",
        "PYTHONPATH": "/tmp/attacker",
        "SSL_CERT_FILE": "/tmp/attacker.pem",
        "LD_PRELOAD": "/tmp/attacker.so",
    }
    (tmp_path / ".env").write_text(
        "\n".join(f"{key}={value}" for key, value in blocked.items()),
        encoding="utf-8",
    )
    for key in blocked:
        monkeypatch.delenv(key, raising=False)

    with pytest.warns(UserWarning) as caught:
        CrupierConfig.from_toml(tmp_path)

    warning_text = " ".join(str(item.message) for item in caught)
    assert all(key not in os.environ for key in blocked)
    assert all(key in warning_text for key in blocked)


def test_env_file_loads_declared_provider_keys_only(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "crupier.toml").write_text(
        """
[providers.openai]
enabled = true
env_key = "OPENAI_API_KEY"

[providers.private_gateway]
enabled = true
mode = "openai_compatible"
host = "https://gateway.example/v1"
env_key = "PRIVATE_GATEWAY_TOKEN"
""",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=openai-local\nPRIVATE_GATEWAY_TOKEN=gateway-local\nIGNORED=value\n",
        encoding="utf-8",
    )
    for key in ("OPENAI_API_KEY", "PRIVATE_GATEWAY_TOKEN", "IGNORED"):
        monkeypatch.delenv(key, raising=False)

    with pytest.warns(UserWarning, match="IGNORED"):
        config = CrupierConfig.from_toml(tmp_path)

    assert config.providers["openai"].env_values["OPENAI_API_KEY"] == "openai-local"
    assert config.providers["private_gateway"].env_values["PRIVATE_GATEWAY_TOKEN"] == "gateway-local"
    assert all(key not in os.environ for key in ("OPENAI_API_KEY", "PRIVATE_GATEWAY_TOKEN", "IGNORED"))


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


@pytest.mark.parametrize("rules", ["deny-all", {"effect": "deny"}, None])
def test_policy_rules_wrong_shape_fails_closed(rules) -> None:
    with pytest.raises(CrupierConfigError, match="rules"):
        CrupierConfig.from_dict({"policy": {"rules": rules}})


def test_policy_rejects_non_object_rule_instead_of_skipping_it() -> None:
    with pytest.raises(CrupierConfigError, match=r"rules\[0\]"):
        CrupierConfig.from_dict(
            {
                "policy": {
                    "rules": [
                        "deny openai",
                        {"effect": "deny", "providers": ["openai"]},
                    ]
                }
            }
        )


def test_custom_host_requires_explicit_opt_in_and_https() -> None:
    remote_http = {
        "providers": {
            "openai": {
                "enabled": True,
                "host": "http://proxy.example/v1",
                "env_key": "OPENAI_API_KEY",
                "allow_custom_host": True,
            }
        }
    }
    with pytest.raises(CrupierConfigError, match="HTTPS"):
        CrupierConfig.from_dict(remote_http)

    remote_https_without_opt_in = {
        "providers": {
            "openai": {
                "enabled": True,
                "host": "https://proxy.example/v1",
                "env_key": "OPENAI_API_KEY",
            }
        }
    }
    with pytest.raises(CrupierConfigError, match="allow_custom_host"):
        CrupierConfig.from_dict(remote_https_without_opt_in)

    remote_https = CrupierConfig.from_dict(
        {
            "providers": {
                "openai": {
                    "enabled": True,
                    "host": "https://proxy.example/v1",
                    "env_key": "OPENAI_API_KEY",
                    "allow_custom_host": True,
                }
            }
        }
    )
    assert remote_https.providers["openai"].host == "https://proxy.example/v1"
    assert remote_https.providers["openai"].options.get("allow_custom_host") is True

    loopback_http = CrupierConfig.from_dict(
        {
            "providers": {
                "openai": {
                    "enabled": True,
                    "host": "http://127.0.0.1:8080/v1",
                    "env_key": "OPENAI_API_KEY",
                    "allow_custom_host": True,
                }
            }
        }
    )
    assert loopback_http.providers["openai"].host == "http://127.0.0.1:8080/v1"


@pytest.mark.parametrize(
    ("experiment", "message"),
    [
        ({"execution": "later"}, "execution"),
        ({"sample_rate": 1.1}, "sample_rate"),
        ({"candidate_models": ["missing-provider"]}, "provider:model"),
        ({"candidate_strategy": "magic"}, "candidate_strategy"),
        ({"promotion": {"action": "deploy"}}, "promotion.action"),
        ({"promotion": {"confidence": 1.1}}, "promotion.confidence"),
    ],
)
def test_experiment_config_rejects_residual_invalid_values(experiment, message: str) -> None:
    with pytest.raises(CrupierConfigError, match=message):
        CrupierConfig.from_dict({"experiments": {"edge": experiment}})


def test_experiment_config_parser_handles_absent_string_and_invalid_candidates() -> None:
    assert config_module._experiment_settings_from_dict(None) == {}
    with pytest.raises(CrupierConfigError, match="must be a table"):
        config_module._experiment_settings_from_dict({"edge": []})

    parsed = config_module._experiment_settings_from_dict(
        {"edge": {"candidate_models": "openai:gpt-5.5"}}
    )
    assert parsed["edge"].candidate_models == ["openai:gpt-5.5"]

    with pytest.raises(CrupierConfigError, match="must be an array"):
        config_module._experiment_settings_from_dict({"edge": {"candidate_models": 42}})

    config = CrupierConfig(experiments={"edge": ExperimentSettings("edge", candidate_models=["invalid"])})
    with pytest.raises(CrupierConfigError, match="provider:model"):
        config.validate()


def test_config_helpers_cover_absent_values_and_malformed_hosts() -> None:
    with pytest.raises(CrupierConfigError, match="policy must be a table"):
        config_module._policy_settings_from_dict(None)
    assert config_module._string_list(None) == []
    assert config_module.host_is_loopback("not a URL") is False
    assert config_module.is_official_provider_host("openai", "not a URL") is False
    with pytest.raises(CrupierConfigError, match="must use http or https"):
        config_module._require_https_unless_loopback("ftp://127.0.0.1")
