from __future__ import annotations

import json
from pathlib import Path

import pytest

from crupier.adapters import ProviderModel
from crupier.config import CrupierConfig
from crupier.errors import CrupierConfigError, CrupierModelUnsupportedError
from crupier.models import CapabilityCard, ModelRef, UpdateReport
import crupier.registry as registry_module
from crupier.registry import ModelRegistry


def _config(root: Path, allow: list[str] | None = None) -> CrupierConfig:
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "registry-test"},
            "models": {"allow": allow or []},
        }
    )
    config.root = root
    return config


def _card(model: str, *, source: str = "local", deprecated: bool = False) -> CapabilityCard:
    return CapabilityCard(
        model_ref=ModelRef(provider="custom", model=model, source=source),
        last_updated="2026-07-15",
        deprecation={"replacement": "custom:new"} if deprecated else {},
    )


def test_registry_helpers_cover_noop_change_and_retained_discovery() -> None:
    report = UpdateReport()
    registry_module._record_card_change_details(
        report,
        model="openai:gpt-5.5",
        old_data=None,
        new_data={"pricing": {"input": 1}},
    )
    assert report.price_changes == []

    assert registry_module._retained_discovery_index_keys({}, exclude_providers=set()) == set()
    retained = registry_module._retained_discovery_index_keys(
        {
            "source": "provider_discovery",
            "models": ["openai:gpt-5.5", "anthropic:claude-sonnet-4-6"],
        },
        exclude_providers={"openai"},
    )
    assert retained == {"anthropic:claude-sonnet-4-6"}


@pytest.mark.parametrize(
    ("provider", "model", "expected_kind", "expected_dimensions"),
    [
        ("anthropic", "claude-embedding-1", "embedding", None),
        ("ollama", "all-minilm", "embedding", 384),
        ("ollama", "nomic-embed-text", "embedding", 768),
        ("ollama", "mxbai-embed-large", "embedding", 1024),
        ("custom", "universal-embed-v1", "embedding", None),
    ],
)
def test_discovered_embedding_cards_are_specialized(
    provider: str,
    model: str,
    expected_kind: str,
    expected_dimensions: int | None,
) -> None:
    card = registry_module._card_from_provider_model(ProviderModel(id=model, provider=provider))

    assert card.model_kind == expected_kind
    assert card.embedding_dimensions == expected_dimensions
    assert card.supports_streaming is False
    assert card.modalities_output == ["embedding"]


@pytest.mark.parametrize(
    ("model", "expected_cost", "expected_stability", "expected_strength"),
    [
        ("vision-small-preview", "low", "preview", "multimodal"),
        ("coder-120b-experimental", "medium", "experimental", "coding"),
        ("reasoner-671b-pro", "medium", "stable", "quality"),
    ],
)
def test_discovered_ollama_chat_profiles_cover_vision_size_and_stability(
    model: str,
    expected_cost: str,
    expected_stability: str,
    expected_strength: str,
) -> None:
    card = registry_module._card_from_provider_model(ProviderModel(id=model, provider="ollama"))

    assert card.cost_tier == expected_cost
    assert card.model_ref.stability == expected_stability
    assert expected_strength in card.strengths


def test_discovered_anthropic_chat_and_unknown_curated_model() -> None:
    card = registry_module._card_from_provider_model(ProviderModel(id="claude-opus-custom", provider="anthropic"))

    assert card.modalities_input[:2] == ["text", "image"]
    assert card.supports_tools is True
    assert card.cost_tier == "high"
    assert registry_module._nan_builtin_card("not-a-built-in") is None
    assert registry_module._embedding_dimensions("unknown", "embedding-model") is None


def test_registry_loads_local_cards_caches_and_handles_allowed_modes(tmp_path: Path) -> None:
    config = _config(tmp_path, ["custom:allowed"])
    registry = ModelRegistry(config)
    config.ensure_project_dirs()
    local = _card("local")
    path = config.capability_cards_dir / "custom__local.json"
    path.write_text(json.dumps(local.to_dict()), encoding="utf-8")

    first = registry.load()
    second = registry.load()

    assert first is second
    assert "custom:local" in first
    assert [card.model_ref.key for card in registry.list(allowed_only=True)] == ["custom:allowed"]
    assert registry.allowed_cards()[0].model_ref.key == "custom:allowed"

    config.models.allow = []
    assert registry.allowed_cards() == registry.list()
    with pytest.raises(CrupierModelUnsupportedError):
        registry.get("custom:missing")


def test_save_card_dry_run_does_not_write(tmp_path: Path) -> None:
    registry = ModelRegistry(_config(tmp_path))

    assert registry.save_card(_card("dry"), dry_run=True) is None
    assert not (registry.config.capability_cards_dir / "custom__dry.json").exists()


def test_model_states_cover_discovered_locked_builtin_local_and_stale(tmp_path: Path) -> None:
    config = _config(tmp_path, ["custom:allowed"])
    registry = ModelRegistry(config)
    config.ensure_project_dirs()
    discovered = _card("discovered", source="discovered")
    registry.save_card(discovered)

    index = {
        "source": "provider_discovery",
        "models": ["custom:other"],
    }
    (config.registry_dir / "models.json").write_text(json.dumps(index), encoding="utf-8")

    states = {item["model"]: item["states"] for item in registry.model_states()}

    assert states["custom:discovered"] == ["stale"]
    assert states["custom:allowed"] == ["allowed"]
    assert states["custom:other"] == ["discovered"]
    assert states["openai:gpt-5.5"] == ["builtin"]

    explicit = registry.model_states(models=["custom:local"], locked_keys=["custom:local"])
    assert explicit[0]["states"] == ["locked"]

    snapshot_index = {"source": "registry_snapshot", "models": ["custom:locked"]}
    (config.registry_dir / "models.json").write_text(json.dumps(snapshot_index), encoding="utf-8")
    locked = registry.model_states(models=["custom:locked"])
    assert locked[0]["states"] == ["locked"]


def test_registry_index_and_previous_discovery_fallbacks(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = ModelRegistry(config)
    config.ensure_project_dirs()
    (config.registry_dir / "models.json").write_text("[]", encoding="utf-8")
    assert registry._read_registry_index() == {}
    assert registry._previous_discovered_keys([None]) == set()

    (config.registry_dir / "models.json").unlink()
    registry.save_card(_card("old", source="discovered"))
    registry.save_card(_card("other"))
    assert registry._previous_discovered_keys(["custom"]) == {"custom:old"}


def test_seed_update_covers_modified_unchanged_and_deprecated_cards(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path, ["custom:deprecated"])
    registry = ModelRegistry(config)
    seeded = _card("deprecated", deprecated=True)
    monkeypatch.setattr(registry, "load", lambda: {"custom:deprecated": seeded})
    config.ensure_project_dirs()
    path = config.capability_cards_dir / "custom__deprecated.json"
    path.write_text(json.dumps({"model_ref": {"provider": "custom", "model": "deprecated"}}), encoding="utf-8")

    modified = registry.update()
    unchanged = registry.update(dry_run=True)

    assert modified.modified_models == ["custom:deprecated"]
    assert modified.deprecated_models == ["custom:deprecated"]
    assert unchanged.unchanged_models == ["custom:deprecated"]


def test_provider_update_retains_other_providers_and_reports_empty_discovery(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = ModelRegistry(config)
    config.ensure_project_dirs()
    index = {
        "source": "provider_discovery",
        "models": ["openai:gpt-5.5", "anthropic:claude-sonnet-4-6"],
    }
    (config.registry_dir / "models.json").write_text(json.dumps(index), encoding="utf-8")

    report = registry.update_from_provider_models(
        [ProviderModel(id="gpt-5.4-mini", provider="openai")],
        discovered_providers=["openai"],
    )
    saved_index = json.loads((config.registry_dir / "models.json").read_text(encoding="utf-8"))

    assert saved_index["models"] == ["anthropic:claude-sonnet-4-6", "openai:gpt-5.4-mini"]
    assert report.removed_models == ["openai:gpt-5.5"]

    empty = registry.update_from_provider_models([], provider="openai", dry_run=True)
    assert "No provider models were discovered." in empty.warnings


def test_snapshot_validation_duplicate_default_name_and_cleanup(tmp_path: Path) -> None:
    config = _config(tmp_path, ["custom:one"])
    registry = ModelRegistry(config)
    registry.update()

    generated = registry.snapshot_create()
    assert generated["name"].startswith("reg_")
    registry.snapshot_create("baseline")
    with pytest.raises(CrupierConfigError, match="already exists"):
        registry.snapshot_create("baseline")

    extra = config.capability_cards_dir / "custom__extra.json"
    extra.write_text(json.dumps(_card("extra").to_dict()), encoding="utf-8")
    restored = registry.snapshot_use("baseline")
    assert str(extra) in restored["removed_files"]
    assert not extra.exists()

    current = registry._load_snapshot("current")
    assert current["name"] == "current"
    with pytest.raises(CrupierConfigError, match="does not exist"):
        registry._load_snapshot("missing")


@pytest.mark.parametrize("name", ["", ".", "..", "bad/name", "bad\\name", "bad name"])
def test_snapshot_name_validation(name: str) -> None:
    with pytest.raises(CrupierConfigError):
        ModelRegistry._normalize_snapshot_name(name)


def test_snapshot_file_validation_and_cards_shape(tmp_path: Path) -> None:
    bad_schema = tmp_path / "schema.json"
    bad_schema.write_text(json.dumps({"schema_version": 999, "name": "bad", "cards": {}}), encoding="utf-8")
    with pytest.raises(CrupierConfigError, match="schema"):
        ModelRegistry._read_snapshot_path(bad_schema)

    missing_fields = tmp_path / "missing.json"
    missing_fields.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(CrupierConfigError, match="Invalid registry snapshot"):
        ModelRegistry._read_snapshot_path(missing_fields)

    with pytest.raises(CrupierConfigError, match="cards must be an object"):
        ModelRegistry._snapshot_cards({"cards": []})
