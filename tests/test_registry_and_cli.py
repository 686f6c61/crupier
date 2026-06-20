import json

from crupier import Crupier
from crupier.adapters import AdapterResponse, EmbeddingResponse, ProviderModel
from crupier.cli import _smoke_model_refs, _verify_provider, _verify_provider_names, main
from crupier.config import CrupierConfig, write_default_project, write_models_allow
from crupier.model_profiles import apply_decision_profile
from crupier.models import CapabilityCard, ModelRef, RequestEnvelope


def test_registry_update_writes_allowed_cards(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True}},
            "models": {"allow": ["openai:gpt-5.5"]},
        }
    )
    config.root = tmp_path
    client = Crupier(config)

    report = client.update(dry_run=False)

    assert "openai:gpt-5.5" in report.changed_models
    assert (tmp_path / ".crupier" / "registry" / "capability-cards" / "openai__gpt-5.5.json").exists()


class FakeDiscoveryAdapter:
    provider = "openai"

    def __init__(self, model_ids=None):
        self.model_ids = model_ids or ["gpt-5.5", "gpt-5.4-mini"]

    def list_models(self):
        return [ProviderModel(id=model_id, provider="openai") for model_id in self.model_ids]

    def generate(self, *, model, prompt, request):
        raise AssertionError("update online should not generate")


class FakeVerifyAdapter:
    provider = "openai"

    def list_models(self):
        return [ProviderModel(id="gpt-5.5", provider="openai")]

    def generate(self, *, model, prompt, request):
        assert isinstance(request, RequestEnvelope)
        return AdapterResponse(
            text="crupier-ok",
            usage={"input_tokens": 1, "output_tokens": 1},
            metadata={"provider": "openai", "model": model},
        )


class FakeEmbeddingVerifyAdapter(FakeVerifyAdapter):
    def list_models(self):
        return [ProviderModel(id="text-embedding-3-small", provider="openai")]

    def generate(self, *, model, prompt, request):
        raise AssertionError("embedding verification should not call chat generation")

    def embed(self, *, model, input):
        return EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            usage={"prompt_tokens": 3, "total_tokens": 3},
            metadata={"provider": "openai", "model": model},
        )


class FakeFailingSmokeAdapter(FakeVerifyAdapter):
    def generate(self, *, model, prompt, request):
        raise RuntimeError(
            "you (686f6c61) have reached your session usage limit "
            "(ref: c6615965-effa-4585-9664-463251607c52)"
        )


def test_registry_update_online_writes_discovered_cards(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True}},
            "models": {"allow": []},
        }
    )
    config.root = tmp_path
    client = Crupier(config, adapters={"openai": FakeDiscoveryAdapter()})

    report = client.update(dry_run=False, online=True, provider="openai")

    assert "openai:gpt-5.5" in report.changed_models
    assert "openai:gpt-5.4-mini" in report.changed_models
    assert (tmp_path / ".crupier" / "registry" / "capability-cards" / "openai__gpt-5.5.json").exists()
    assert (tmp_path / ".crupier" / "registry" / "models.json").exists()


def test_registry_update_online_classifies_embedding_models(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True}},
            "models": {"allow": []},
        }
    )
    config.root = tmp_path
    client = Crupier(config, adapters={"openai": FakeDiscoveryAdapter(["text-embedding-3-small"])})

    client.update(dry_run=False, online=True, provider="openai")

    card_path = tmp_path / ".crupier" / "registry" / "capability-cards" / "openai__text-embedding-3-small.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["model_kind"] == "embedding"
    assert card["supports_embeddings"] is True
    assert card["modalities_output"] == ["embedding"]
    assert card["embedding_dimensions"] == 1536
    assert card["routing_hints"]["routing_status"] == "specialized"


def test_registry_update_online_enriches_decision_profiles(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True}},
            "models": {"allow": []},
        }
    )
    config.root = tmp_path
    client = Crupier(config, adapters={"openai": FakeDiscoveryAdapter(["gpt-5.5", "o3", "gpt-5.2-chat-latest"])})

    client.update(dry_run=False, online=True, provider="openai")

    recommended = client.registry.get("openai:gpt-5.5")
    expensive = client.registry.get("openai:o3")
    deprecated = client.registry.get("openai:gpt-5.2-chat-latest")

    assert recommended.routing_hints["routing_status"] == "recommended"
    assert recommended.routing_hints["production_default"] is True
    assert expensive.routing_hints["routing_status"] == "opt_in"
    assert expensive.routing_hints["production_default"] is False
    assert deprecated.routing_hints["routing_status"] == "deprecated"
    assert deprecated.deprecation["replacement"] == "openai:gpt-5.5"


def test_decision_profiles_do_not_recommend_uncurated_discovered_models():
    card = apply_decision_profile(
        CapabilityCard(
            model_ref=ModelRef(provider="anthropic", model="claude-fable-5", source="discovered"),
            last_updated="2026-06-20",
        ),
        provider_metadata={"id": "claude-fable-5"},
    )

    assert card.routing_hints["routing_status"] == "unknown"
    assert card.routing_hints["production_default"] is False
    assert card.routing_hints["requires_opt_in"] is True


def test_date_snapshot_models_require_explicit_opt_in():
    card = apply_decision_profile(
        CapabilityCard(
            model_ref=ModelRef(
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
                source="discovered",
            ),
            last_updated="2026-06-20",
        ),
        provider_metadata={"id": "claude-sonnet-4-5-20250929"},
    )

    assert card.routing_hints["routing_status"] == "opt_in"
    assert card.routing_hints["lifecycle"] == "snapshot"
    assert card.routing_hints["production_default"] is False


def test_failed_capability_probe_overrides_inferred_model_family_support():
    card = apply_decision_profile(
        CapabilityCard(
            model_ref=ModelRef(provider="ollama", model="gpt-oss:120b", source="discovered"),
            last_updated="2026-06-20",
            supports_tools=True,
            strengths=["tool_use", "reasoning"],
            capability_status={"tool_call": {"status": "failed", "source": "probe:tool_call"}},
        )
    )

    assert card.supports_tools is False
    assert "tool_use" not in card.strengths
    assert "tool_use" not in card.skill_scores


def test_registry_update_online_dry_run_reports_diff_and_states(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True}},
            "models": {"allow": ["openai:gpt-5.4-mini"]},
        }
    )
    config.root = tmp_path
    client = Crupier(config, adapters={"openai": FakeDiscoveryAdapter(["gpt-5.5", "gpt-5.4-mini"])})
    client.update(dry_run=False, online=True, provider="openai")

    dry_client = Crupier(config, adapters={"openai": FakeDiscoveryAdapter(["gpt-5.5", "gpt-5.4-nano"])})
    report = dry_client.update(dry_run=True, online=True, provider="openai")

    assert report.added_models == ["openai:gpt-5.4-nano"]
    assert report.removed_models == ["openai:gpt-5.4-mini"]
    assert report.unchanged_models == ["openai:gpt-5.5"]
    assert report.diff["added"] == ["openai:gpt-5.4-nano"]
    assert report.diff["removed"] == ["openai:gpt-5.4-mini"]
    assert report.requires_confirmation is True
    assert report.written_files == []
    assert not (tmp_path / ".crupier" / "registry" / "capability-cards" / "openai__gpt-5.4-nano.json").exists()

    state_by_model = {item["model"]: item["states"] for item in report.model_states}
    assert state_by_model["openai:gpt-5.4-nano"] == ["discovered"]
    assert state_by_model["openai:gpt-5.4-mini"] == ["allowed", "stale"]


def test_registry_snapshot_create_diff_and_use(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True}},
            "models": {"allow": ["openai:gpt-5.5"]},
        }
    )
    config.root = tmp_path
    client = Crupier(config)
    client.update(dry_run=False)

    snapshot = client.registry.snapshot_create("baseline", allowed_only=True)
    assert snapshot["name"] == "baseline"
    assert snapshot["card_count"] == 1
    assert (tmp_path / ".crupier" / "registry" / "snapshots" / "baseline.json").exists()

    no_diff = client.registry.snapshot_diff("baseline", "current")
    assert no_diff["added"] == []
    assert no_diff["removed"] == []
    assert no_diff["changed"] == []
    reverse_no_diff = client.registry.snapshot_diff("current", "baseline")
    assert reverse_no_diff["added"] == []
    assert reverse_no_diff["removed"] == []
    assert reverse_no_diff["changed"] == []

    card_path = tmp_path / ".crupier" / "registry" / "capability-cards" / "openai__gpt-5.5.json"
    card_data = json.loads(card_path.read_text(encoding="utf-8"))
    card_data["local_eval_scores"] = {"agentic": 1.25}
    card_path.write_text(json.dumps(card_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    client.registry._cards = None

    diff = client.registry.snapshot_diff("baseline", "current")
    assert diff["changed"] == [{"model": "openai:gpt-5.5", "fields": ["evidence", "local_eval_scores"]}]

    restore = client.registry.snapshot_use("baseline")
    assert restore["restored_models"] == ["openai:gpt-5.5"]
    restored = json.loads(card_path.read_text(encoding="utf-8"))
    assert restored["local_eval_scores"] == {}
    states = client.registry.model_states(models=["openai:gpt-5.5"])[0]["states"]
    assert states == ["allowed", "locked"]


def test_registry_snapshot_use_restores_allowlist(tmp_path):
    write_default_project(tmp_path)
    write_models_allow(tmp_path, ["openai:gpt-5.5", "anthropic:claude-opus-4-8"], replace=True)
    config = CrupierConfig.from_toml(tmp_path)
    client = Crupier(config)
    client.update(dry_run=False)
    client.registry.snapshot_create("allowed", allowed_only=True)

    assert main(["--project", str(tmp_path), "models", "allow", "openai:gpt-5.4-mini", "--replace"]) == 0
    restore = client.registry.snapshot_use("allowed", restore_allowlist=True)

    assert restore["allowlist_restored"] is True
    reloaded = CrupierConfig.from_toml(tmp_path)
    assert reloaded.models.allow == ["openai:gpt-5.5", "anthropic:claude-opus-4-8"]


def test_cli_init_and_deal(tmp_path, capsys):
    assert main(["--project", str(tmp_path), "init"]) == 0
    assert main(["--project", str(tmp_path), "deal", "Plan this", "--mode", "agentic"]) == 0
    assert main(["--project", str(tmp_path), "route", "Plan this", "--mode", "agentic"]) == 0

    output = capsys.readouterr().out
    assert "Created" in output
    assert "Crupier dry-run planned" in output
    assert "scores:" in output


def test_cli_models_allow_updates_config(tmp_path):
    assert main(["--project", str(tmp_path), "init"]) == 0
    assert main(["--project", str(tmp_path), "models", "allow", "claude:claude-opus-4-8", "--replace"]) == 0

    config = CrupierConfig.from_toml(tmp_path)
    assert config.models.allow == ["anthropic:claude-opus-4-8"]


def test_cli_registry_snapshot_commands(tmp_path, capsys):
    assert main(["--project", str(tmp_path), "init"]) == 0
    assert main(["--project", str(tmp_path), "update"]) == 0
    assert main(["--project", str(tmp_path), "registry", "snapshot", "create", "baseline", "--allowed-only"]) == 0
    assert main(["--project", str(tmp_path), "registry", "snapshot", "list"]) == 0
    assert main(["--project", str(tmp_path), "registry", "snapshot", "diff", "baseline"]) == 0

    output = capsys.readouterr().out
    assert "Created registry snapshot baseline" in output
    assert "baseline" in output
    assert "changed: 0" in output


def test_cli_update_and_models_list_show_registry_states(tmp_path, capsys):
    assert main(["--project", str(tmp_path), "init"]) == 0
    assert main(["--project", str(tmp_path), "update", "--dry-run"]) == 0
    assert main(["--project", str(tmp_path), "models", "list"]) == 0

    output = capsys.readouterr().out
    assert "update: dry-run" in output
    assert "added:" in output
    assert "states=" in output


def test_cli_models_list_recommended_excludes_expensive_opt_in_models(tmp_path, capsys):
    assert main(["--project", str(tmp_path), "init"]) == 0
    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "models",
                "allow",
                "openai:gpt-5.5",
                "openai:o3",
                "openai:o4-mini",
                "--replace",
            ]
        )
        == 0
    )
    assert main(["--project", str(tmp_path), "models", "list", "--recommended"]) == 0

    output = capsys.readouterr().out
    assert "openai:gpt-5.5" in output
    assert "openai:o3" not in output
    assert "openai:o4-mini" not in output


def test_cli_models_show_outputs_decision_profile(tmp_path, capsys):
    assert main(["--project", str(tmp_path), "init"]) == 0
    assert main(["--project", str(tmp_path), "models", "allow", "openai:o3", "--replace"]) == 0
    assert main(["--project", str(tmp_path), "models", "show", "openai:o3"]) == 0

    output = capsys.readouterr().out
    assert "routing_status: opt_in" in output
    assert "requires_opt_in: True" in output


def test_cli_orchestrator_set_updates_config(tmp_path, capsys):
    assert main(["--project", str(tmp_path), "init"]) == 0
    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "orchestrator",
                "set",
                "--model",
                "ollama:glm-5.2",
                "--fallback-model",
                "anthropic:claude-opus-4-8",
            ]
        )
        == 0
    )

    config = CrupierConfig.from_toml(tmp_path)
    assert config.orchestrator.mode == "model"
    assert config.orchestrator.model == "ollama:glm-5.2"
    assert config.orchestrator.fallback_model == "anthropic:claude-opus-4-8"
    assert "Updated [orchestrator]" in capsys.readouterr().out


def test_smoke_model_refs_selects_one_per_enabled_provider(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "providers": {
                "openai": {"enabled": True},
                "anthropic": {"enabled": True},
                "ollama": {"enabled": False},
            },
            "models": {
                "allow": [
                    "openai:gpt-5.5",
                    "openai:gpt-5.4-mini",
                    "claude:claude-opus-4-8",
                    "ollama:gpt-oss:120b",
                ]
            },
        }
    )

    refs = _smoke_model_refs(config, provider=None, explicit=None, all_models=False)

    assert refs == ["openai:gpt-5.5", "anthropic:claude-opus-4-8"]


def test_smoke_model_refs_explicit_normalizes_aliases():
    config = CrupierConfig.from_dict({})

    refs = _smoke_model_refs(
        config,
        provider="anthropic",
        explicit=["claude:claude-opus-4-8", "openai:gpt-5.5"],
        all_models=False,
    )

    assert refs == ["anthropic:claude-opus-4-8"]


def test_verify_provider_names_adds_openai_baseline():
    config = CrupierConfig.from_dict(
        {
            "providers": {
                "openai": {"enabled": True},
                "anthropic": {"enabled": True},
                "ollama": {"enabled": True},
            }
        }
    )

    providers = _verify_provider_names(config, requested=["anthropic", "ollama"], include_openai_baseline=True)

    assert providers == ["openai", "anthropic", "ollama"]


def test_cli_verify_reports_missing_required_env_without_provider_calls(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    (tmp_path / "crupier.toml").write_text(
        """
[providers.openai]
enabled = true
env_key = "OPENAI_API_KEY"

[providers.anthropic]
enabled = true
env_key = "ANTHROPIC_API_KEY"

[providers.ollama]
enabled = true
host = "https://ollama.com/api"
env_key = "OLLAMA_API_KEY"

[models]
allow = ["openai:gpt-5.5", "anthropic:claude-opus-4-8", "ollama:gpt-oss:120b"]
""",
        encoding="utf-8",
    )

    status = main(["--project", str(tmp_path), "verify", "--provider", "anthropic", "--provider", "ollama"])

    output = capsys.readouterr().out
    assert status == 1
    assert "providers: openai, anthropic, ollama" in output
    assert "OPENAI_API_KEY=missing" in output
    assert "ANTHROPIC_API_KEY=missing" in output
    assert "OLLAMA_API_KEY=missing" in output


def test_cli_verify_reports_missing_google_env_without_provider_calls(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (tmp_path / "crupier.toml").write_text(
        """
[providers.openai]
enabled = true
env_key = "OPENAI_API_KEY"

[providers.google]
enabled = true
env_key = "GOOGLE_API_KEY"

[models]
allow = ["openai:gpt-5.5", "google:gemini-3.5-flash"]
""",
        encoding="utf-8",
    )

    status = main(["--project", str(tmp_path), "verify", "--provider", "google"])

    output = capsys.readouterr().out
    assert status == 1
    assert "providers: openai, google" in output
    assert "OPENAI_API_KEY=missing" in output
    assert "GOOGLE_API_KEY/GEMINI_API_KEY=missing" in output


def test_cli_verify_reports_missing_openrouter_env_without_provider_calls(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / "crupier.toml").write_text(
        """
[providers.openai]
enabled = true
env_key = "OPENAI_API_KEY"

[providers.openrouter]
enabled = true
mode = "byok"
host = "https://openrouter.ai/api/v1"
env_key = "OPENROUTER_API_KEY"

[models]
allow = ["openai:gpt-5.5", "openrouter:openai/gpt-4o"]
""",
        encoding="utf-8",
    )

    status = main(["--project", str(tmp_path), "verify", "--provider", "openrouter"])

    output = capsys.readouterr().out
    assert status == 1
    assert "providers: openai, openrouter" in output
    assert "OPENAI_API_KEY=missing" in output
    assert "OPENROUTER_API_KEY=missing" in output


def test_verify_provider_with_fake_adapter_runs_discovery_and_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True, "env_key": "OPENAI_API_KEY"}},
            "models": {"allow": ["openai:gpt-5.5"]},
            "routing": {"default_strategy": "single"},
        }
    )
    config.root = tmp_path
    client = Crupier(config, adapters={"openai": FakeVerifyAdapter()})

    item = _verify_provider(client, "openai", run_smoke=True, all_models=False)

    assert item["provider"] == "openai"
    assert item["discovered_count"] == 1
    assert item["smoke"][0]["ok"] is True
    assert item["status"] == "needs_probes"


def test_verify_provider_uses_embedding_smoke_for_embedding_models(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True, "env_key": "OPENAI_API_KEY"}},
            "models": {"allow": ["openai:text-embedding-3-small"]},
            "routing": {"default_strategy": "single"},
        }
    )
    config.root = tmp_path
    client = Crupier(config, adapters={"openai": FakeEmbeddingVerifyAdapter()})

    item = _verify_provider(client, "openai", run_smoke=True, all_models=False)

    assert item["smoke"][0]["ok"] is True
    assert item["smoke"][0]["kind"] == "embeddings"
    assert item["smoke"][0]["embedding_dimensions"] == 3
    assert item["status"] == "needs_probes"


def test_verify_provider_redacts_account_identifiers_from_smoke_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True, "env_key": "OPENAI_API_KEY"}},
            "models": {"allow": ["openai:gpt-5.5"]},
            "routing": {"default_strategy": "single"},
        }
    )
    config.root = tmp_path
    client = Crupier(config, adapters={"openai": FakeFailingSmokeAdapter()})

    item = _verify_provider(client, "openai", run_smoke=True, all_models=False)

    error = item["smoke"][0]["error"]
    assert "686f6c61" not in error
    assert "c6615965-effa-4585-9664-463251607c52" not in error
    assert "you ([redacted])" in error
    assert "ref: [redacted]" in error
    assert item["status"] == "failed"
