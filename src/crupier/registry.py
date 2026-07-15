"""Model registry and project capability-card management."""

from __future__ import annotations

import json
import re
import builtins
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import CrupierConfig, write_models_allow
from .default_cards import BUILTIN_CAPABILITY_CARDS
from .errors import CrupierConfigError, CrupierModelUnsupportedError
from .model_profiles import apply_decision_profile
from .models import CapabilityCard, ModelRef, UpdateReport
from .adapters import ProviderModel

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _card_filename(model_key: str) -> str:
    safe = model_key.replace(":", "__").replace("/", "_").replace("\\", "_")
    return f"{safe}.json"


def _json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _diff_fields(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    return sorted(field for field in set(left) | set(right) if left.get(field) != right.get(field))


PROFILE_CHANGE_FIELDS = {
    "context_window",
    "max_output_tokens",
    "model_kind",
    "modalities_input",
    "modalities_output",
    "supports_embeddings",
    "embedding_dimensions",
    "embedding_input_modalities",
    "supports_tools",
    "supports_structured_output",
    "supports_streaming",
    "supports_web",
    "supports_file_input",
    "supports_code_execution",
    "cost_tier",
    "latency_tier",
    "quality_tier",
    "strengths",
    "routing_hints",
    "natural_profile",
    "skill_scores",
    "capability_status",
}


def _record_card_change_details(
    report: UpdateReport,
    *,
    model: str,
    old_data: dict[str, Any] | None,
    new_data: dict[str, Any],
) -> None:
    if old_data is None:
        return
    if old_data.get("pricing") != new_data.get("pricing"):
        report.price_changes.append(
            {
                "model": model,
                "before": old_data.get("pricing", {}),
                "after": new_data.get("pricing", {}),
            }
        )
    fields = [field for field in sorted(PROFILE_CHANGE_FIELDS) if old_data.get(field) != new_data.get(field)]
    if fields:
        report.profile_changes.append({"model": model, "fields": fields})


def _retained_discovery_index_keys(index: dict[str, Any], *, exclude_providers: set[str]) -> set[str]:
    if index.get("source") != "provider_discovery":
        return set()
    retained: set[str] = set()
    for model_key in index.get("models", []) or []:
        ref = ModelRef.parse(model_key)
        if ref.provider not in exclude_providers:
            retained.add(ref.key)
    return retained


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_card_for(model_key: str) -> CapabilityCard:
    model_ref = ModelRef.parse(model_key)
    strengths: list[str] = []
    source = "openai_compatible" if model_ref.provider == "openrouter" else "native"
    model_ref = ModelRef(
        provider=model_ref.provider,
        model=model_ref.model,
        stability=model_ref.stability,
        source=source,
    )
    card = CapabilityCard(
        model_ref=model_ref,
        last_updated="generated",
        context_window=None,
        max_output_tokens=None,
        modalities_input=["text"],
        modalities_output=["text"],
        supports_tools=False,
        supports_structured_output=False,
        supports_streaming=True,
        pricing={"confidence": "unknown"},
        data_retention="unknown",
        zdr_eligible=None,
        known_edge_cases=["Generated generic card; run `crupier update --online` in a future version for richer metadata."],
        strengths=strengths,
        cost_tier="unknown",
        latency_tier="unknown",
        quality_tier="unknown",
    )
    return apply_decision_profile(_apply_embedding_kind(card))


def _card_from_provider_model(provider_model: ProviderModel) -> CapabilityCard:
    model_ref = ModelRef.parse(provider_model.model_ref)
    model_id = model_ref.model.lower()
    provider = model_ref.provider
    if provider == "nan":
        curated = _nan_builtin_card(model_id)
        if curated is not None:
            curated.model_ref = ModelRef(
                provider="nan",
                model=model_ref.model,
                stability=curated.model_ref.stability,
                source="discovered",
            )
            curated.last_updated = date.today().isoformat()
            curated.evidence = {
                **curated.evidence,
                "provider_discovery": {"metadata": provider_model.metadata},
            }
            return apply_decision_profile(curated, provider_metadata=provider_model.metadata)
    strengths: list[str] = []
    supports_tools = False
    supports_structured_output = False
    supports_file_input = False
    supports_embeddings = False
    embedding_dimensions: int | None = None
    supports_streaming = True
    supports_web = False
    supports_code_execution = False
    modalities_input = ["text"]
    modalities_output = ["text"]
    model_kind = "chat"
    cost_tier = "unknown"
    latency_tier = "unknown"
    quality_tier = "unknown"

    if provider == "openai":
        if _is_embedding_model(provider, model_id):
            model_kind = "embedding"
            supports_embeddings = True
            supports_streaming = False
            modalities_output = ["embedding"]
            strengths.extend(["embeddings", "rag", "semantic_search"])
            embedding_dimensions = _embedding_dimensions(provider, model_id)
            cost_tier = "low"
            latency_tier = "fast"
            quality_tier = "embedding"
        else:
            supports_tools = model_id.startswith(("gpt-", "o"))
            supports_structured_output = model_id.startswith(("gpt-", "o"))
            supports_file_input = model_id.startswith(("gpt-", "o"))
            supports_web = model_id.startswith("gpt-")
        if model_kind == "chat" and model_id.startswith(("gpt-", "o")):
            modalities_input = ["text", "image"]
        if model_kind == "chat" and ("mini" in model_id or "nano" in model_id):
            strengths.extend(["low_cost", "low_latency", "tool_use", "structured_output"])
            cost_tier = "low"
            latency_tier = "fast"
            quality_tier = "strong"
        elif model_kind == "chat" and model_id.startswith(("gpt-", "o")):
            strengths.extend(["reasoning", "coding", "agentic", "quality", "tool_use", "structured_output"])
            cost_tier = "high"
            latency_tier = "medium"
            quality_tier = "frontier"
    elif provider == "anthropic":
        if _is_embedding_model(provider, model_id):
            model_kind = "embedding"
            supports_embeddings = True
            supports_streaming = False
            modalities_output = ["embedding"]
            strengths.extend(["embeddings", "rag", "semantic_search"])
            embedding_dimensions = _embedding_dimensions(provider, model_id)
            cost_tier = "unknown"
            latency_tier = "unknown"
            quality_tier = "embedding"
        else:
            supports_tools = model_id.startswith("claude")
            supports_structured_output = model_id.startswith("claude")
            supports_file_input = model_id.startswith("claude")
        if model_kind == "chat" and model_id.startswith("claude"):
            modalities_input = ["text", "image"]
            strengths.extend(["reasoning", "coding", "agentic", "critique", "quality"])
            cost_tier = "high" if "opus" in model_id else "medium"
            latency_tier = "medium"
            quality_tier = "frontier" if any(name in model_id for name in ["opus", "sonnet"]) else "strong"
    elif provider == "ollama":
        if _is_embedding_model(provider, model_id):
            model_kind = "embedding"
            supports_embeddings = True
            supports_streaming = False
            modalities_output = ["embedding"]
            strengths.extend(["embeddings", "rag", "semantic_search"])
            embedding_dimensions = _embedding_dimensions(provider, model_id)
            cost_tier = "low"
            latency_tier = "fast"
            quality_tier = "embedding"
        elif any(name in model_id for name in ["vision", "vl", "llava", "moondream", "bakllava", "minicpm-v"]):
            modalities_input = ["text", "image"]
            supports_file_input = True
            strengths.append("multimodal")
        is_small_parameter_model = any(
            re.search(rf"(?<!\d){size}(?!\d)", model_id) for size in ("20b", "24b")
        )
        if model_kind == "chat" and (
            any(name in model_id for name in ["flash", "small"]) or is_small_parameter_model
        ):
            strengths.extend(["low_latency", "low_cost"])
            cost_tier = "low"
            latency_tier = "fast"
            quality_tier = "strong"
        elif model_kind == "chat" and any(name in model_id for name in ["671b", "480b", "120b", "pro"]):
            strengths.extend(["reasoning", "coding", "quality"])
            cost_tier = "medium"
            latency_tier = "medium"
            quality_tier = "frontier"

    if model_kind == "chat" and any(name in model_id for name in ["code", "coder", "devstral"]):
        strengths.extend(["coding", "agentic"])
    if "preview" in model_id:
        stability = "preview"
    elif "experimental" in model_id or "exp" in model_id:
        stability = "experimental"
    else:
        stability = model_ref.stability

    card = CapabilityCard(
        model_ref=ModelRef(provider=provider, model=model_ref.model, stability=stability, source="discovered"),
        last_updated=date.today().isoformat(),
        model_kind=model_kind,
        modalities_input=modalities_input,
        modalities_output=modalities_output,
        supports_embeddings=supports_embeddings,
        embedding_dimensions=embedding_dimensions,
        embedding_input_modalities=["text"] if supports_embeddings else [],
        supports_tools=supports_tools,
        supports_structured_output=supports_structured_output,
        supports_streaming=supports_streaming,
        supports_web=supports_web,
        supports_file_input=supports_file_input,
        supports_code_execution=supports_code_execution,
        pricing={"confidence": "unknown", "source": "provider_discovery"},
        data_retention="provider_policy",
        known_edge_cases=["Discovered from provider API; capability details are heuristic until verified by smoke/evals."],
        benchmarks={"source": "provider_discovery"},
        strengths=sorted(set(strengths)),
        cost_tier=cost_tier,
        latency_tier=latency_tier,
        quality_tier=quality_tier,
    )
    return apply_decision_profile(_apply_embedding_kind(card), provider_metadata=provider_model.metadata)


def _apply_embedding_kind(card: CapabilityCard) -> CapabilityCard:
    model_id = card.model_ref.model.lower()
    if _is_embedding_model(card.model_ref.provider, model_id):
        card.model_kind = "embedding"
        card.supports_embeddings = True
        card.embedding_dimensions = _embedding_dimensions(card.model_ref.provider, model_id)
        card.embedding_input_modalities = ["text"]
        card.modalities_input = ["text"]
        card.modalities_output = ["embedding"]
        card.supports_tools = False
        card.supports_structured_output = False
        card.supports_file_input = False
        card.supports_streaming = False
        card.quality_tier = "embedding"
        card.strengths = sorted(set(card.strengths + ["embeddings", "rag", "semantic_search"]))
    return card


def _is_embedding_model(provider: str, model_id: str) -> bool:
    if provider == "openai":
        return "embedding" in model_id or model_id.startswith("text-embedding")
    if provider == "anthropic":
        return "embedding" in model_id or "embed" in model_id
    if provider == "ollama":
        embedding_markers = [
            "embed",
            "embedding",
            "all-minilm",
            "bge-",
            "e5-",
            "gte-",
            "jina-embeddings",
            "snowflake-arctic-embed",
        ]
        return any(marker in model_id for marker in embedding_markers)
    return "embedding" in model_id or "embed" in model_id


def _embedding_dimensions(provider: str, model_id: str) -> int | None:
    if provider == "openai":
        if "text-embedding-3-large" in model_id:
            return 3072
        if "text-embedding-3-small" in model_id or "text-embedding-ada-002" in model_id:
            return 1536
    if provider == "ollama":
        if "all-minilm" in model_id:
            return 384
        if "nomic-embed-text" in model_id:
            return 768
        if "mxbai-embed-large" in model_id:
            return 1024
    if provider == "nan" and model_id == "qwen3-embedding":
        return 4096
    return None


def _nan_builtin_card(model_id: str) -> CapabilityCard | None:
    for data in BUILTIN_CAPABILITY_CARDS:
        ref = data.get("model_ref", {})
        if ref.get("provider") == "nan" and str(ref.get("model", "")).lower() == model_id:
            return CapabilityCard.from_dict(data)
    return None


class ModelRegistry:
    """Loads built-in and project-local capability cards."""

    def __init__(self, config: CrupierConfig):
        self.config = config
        self._cards: dict[str, CapabilityCard] | None = None

    @classmethod
    def builtin_cards(cls) -> dict[str, CapabilityCard]:
        cards = [apply_decision_profile(CapabilityCard.from_dict(data)) for data in BUILTIN_CAPABILITY_CARDS]
        return {card.model_ref.key: card for card in cards}

    def load(self) -> dict[str, CapabilityCard]:
        if self._cards is not None:
            return self._cards
        cards = self.builtin_cards()
        cards.update(self._load_local_cards())
        for model_key in self.config.models.allow:
            normalized = ModelRef.parse(model_key).key
            cards.setdefault(normalized, _default_card_for(model_key))
        self._cards = cards
        return cards

    def _load_local_cards(self) -> dict[str, CapabilityCard]:
        directory = self.config.capability_cards_dir
        if not directory.exists():
            return {}
        cards: dict[str, CapabilityCard] = {}
        for path in sorted(directory.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            card = apply_decision_profile(CapabilityCard.from_dict(data))
            cards[card.model_ref.key] = card
        return cards

    def list(self, *, allowed_only: bool = False) -> list[CapabilityCard]:
        cards = self.load()
        if not allowed_only:
            return sorted(cards.values(), key=lambda card: card.model_ref.key)
        allowed = {ModelRef.parse(model_key).key for model_key in self.config.models.allow}
        return sorted((card for key, card in cards.items() if key in allowed), key=lambda card: card.model_ref.key)

    def get(self, model_key: str) -> CapabilityCard:
        model_key = ModelRef.parse(model_key).key
        cards = self.load()
        if model_key not in cards:
            raise CrupierModelUnsupportedError(f"No capability card found for {model_key!r}.")
        return cards[model_key]

    def allowed_cards(self) -> builtins.list[CapabilityCard]:
        if self.config.models.allow:
            return [self.get(model_key) for model_key in self.config.models.allow]
        return self.list()

    def save_card(self, card: CapabilityCard, *, dry_run: bool = False) -> str | None:
        """Write one project-local capability card."""

        self.config.ensure_project_dirs()
        path = self.config.capability_cards_dir / _card_filename(card.model_ref.key)
        if not dry_run:
            path.write_text(_json_dumps(card.to_dict()), encoding="utf-8")
            self._cards = None
            return str(path)
        return None

    def model_states(
        self,
        *,
        models: Iterable[str] | None = None,
        discovered_keys: Iterable[str] | None = None,
        locked_keys: Iterable[str] | None = None,
    ) -> builtins.list[dict[str, Any]]:
        """Return reader-facing registry state labels for models."""

        cards = self.load()
        builtin_keys = set(self.builtin_cards())
        allowed = {ModelRef.parse(model_key).key for model_key in self.config.models.allow}
        index = self._read_registry_index()
        index_source = index.get("source")
        indexed_models = {ModelRef.parse(model_key).key for model_key in index.get("models", [])}

        if discovered_keys is not None:
            discovered = {ModelRef.parse(model_key).key for model_key in discovered_keys}
        elif index_source == "provider_discovery":
            discovered = set(indexed_models)
        else:
            discovered = {
                key
                for key, card in cards.items()
                if card.model_ref.source == "discovered" or card.pricing.get("source") == "provider_discovery"
            }

        if locked_keys is not None:
            locked = {ModelRef.parse(model_key).key for model_key in locked_keys}
        elif index_source == "registry_snapshot":
            locked = set(indexed_models)
        else:
            locked = set()

        if models is None:
            keys = set(cards) | allowed | discovered | locked
        else:
            keys = {ModelRef.parse(model_key).key for model_key in models}

        states: list[dict[str, Any]] = []
        for model_key in sorted(keys):
            card = cards.get(model_key)
            ref = card.model_ref if card else ModelRef.parse(model_key)
            labels: list[str] = []
            if model_key in discovered:
                labels.append("discovered")
            if model_key in allowed:
                labels.append("allowed")
            if model_key in locked:
                labels.append("locked")
            if (
                index_source == "provider_discovery"
                and model_key not in discovered
                and card
                and (card.model_ref.source == "discovered" or card.pricing.get("source") == "provider_discovery")
            ):
                labels.append("stale")
            if not labels:
                labels.append("builtin" if model_key in builtin_keys else "local")
            states.append(
                {
                    "model": model_key,
                    "provider": ref.provider,
                    "stability": ref.stability,
                    "source": ref.source,
                    "routing_status": (card.routing_hints.get("routing_status") if card else "unknown"),
                    "lifecycle": (card.routing_hints.get("lifecycle") if card else ref.stability),
                    "production_default": bool(card.routing_hints.get("production_default", False)) if card else False,
                    "states": labels,
                }
            )
        return states

    def _read_registry_index(self) -> dict[str, Any]:
        path = self.config.registry_dir / "models.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}

    def _previous_discovered_keys(self, providers: Iterable[str | None]) -> set[str]:
        provider_set = {provider for provider in providers if provider}
        if not provider_set:
            return set()

        index = self._read_registry_index()
        if index.get("source") == "provider_discovery":
            return {
                ModelRef.parse(model_key).key
                for model_key in index.get("models", [])
                if ModelRef.parse(model_key).provider in provider_set
            }

        discovered: set[str] = set()
        for model_key, card in self._load_local_cards().items():
            if card.model_ref.provider not in provider_set:
                continue
            if card.model_ref.source == "discovered" or card.pricing.get("source") == "provider_discovery":
                discovered.add(model_key)
        return discovered

    def update(self, *, dry_run: bool = False, models: Iterable[str] | None = None) -> UpdateReport:
        """Write project-local capability cards for allowed models.

        This initial release uses local seed metadata only. The method is shaped so
        provider/API refresh and online source checks can be added without
        changing the public contract.
        """

        self.config.ensure_project_dirs()
        model_keys = list(models or self.config.models.allow)
        report = UpdateReport(dry_run=dry_run)
        changed: list[dict[str, Any]] = []

        for model_key in model_keys:
            normalized = ModelRef.parse(model_key).key
            card = self.load().get(normalized) or _default_card_for(model_key)
            path = self.config.capability_cards_dir / _card_filename(normalized)
            new_data = card.to_dict()
            old_data = None
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    old_data = json.load(handle)
            if old_data != new_data:
                if old_data is None:
                    report.added_models.append(normalized)
                else:
                    report.modified_models.append(normalized)
                    changed.append({"model": normalized, "fields": _diff_fields(old_data, new_data)})
                    _record_card_change_details(report, model=normalized, old_data=old_data, new_data=new_data)
                report.changed_models.append(normalized)
                if not dry_run:
                    path.write_text(_json_dumps(new_data), encoding="utf-8")
                    report.written_files.append(str(path))
            else:
                report.unchanged_models.append(normalized)
            if card.deprecation:
                report.deprecated_models.append(normalized)

        index_path = self.config.registry_dir / "models.json"
        index_data = {"models": [ModelRef.parse(model_key).key for model_key in model_keys]}
        if not dry_run:
            index_path.write_text(_json_dumps(index_data), encoding="utf-8")
            report.written_files.append(str(index_path))

        report.diff = {
            "added": sorted(report.added_models),
            "removed": [],
            "changed": changed,
            "unchanged": len(report.unchanged_models),
        }
        report.warnings.append("Using local seed capability cards. Run `crupier update --online` to discover provider models.")
        self._cards = None
        report.model_states = self.model_states(models=model_keys)
        return report

    def update_from_provider_models(
        self,
        provider_models: Iterable[ProviderModel],
        *,
        dry_run: bool = False,
        provider: str | None = None,
        discovered_providers: Iterable[str] | None = None,
        warnings: Iterable[str] | None = None,
    ) -> UpdateReport:
        self.config.ensure_project_dirs()
        report = UpdateReport(dry_run=dry_run)
        provider_models = list(provider_models)
        target_cards: dict[str, CapabilityCard] = {}
        updated_providers = {provider} if provider else set(discovered_providers or [])
        report.warnings.extend(warnings or [])

        for provider_model in provider_models:
            card = _card_from_provider_model(provider_model)
            normalized = card.model_ref.key
            updated_providers.add(card.model_ref.provider)
            target_cards[normalized] = card

        model_keys = sorted(target_cards)
        index_model_keys = model_keys
        if provider is None:
            retained = _retained_discovery_index_keys(self._read_registry_index(), exclude_providers=updated_providers)
            index_model_keys = sorted(set(model_keys) | retained)
        previous_discovered = self._previous_discovered_keys(updated_providers)
        changed: list[dict[str, Any]] = []

        for normalized in model_keys:
            card = target_cards[normalized]
            path = self.config.capability_cards_dir / _card_filename(normalized)
            new_data = card.to_dict()
            old_data = None
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    old_data = json.load(handle)
            if old_data != new_data:
                if old_data is None:
                    report.added_models.append(normalized)
                else:
                    report.modified_models.append(normalized)
                    changed.append({"model": normalized, "fields": _diff_fields(old_data, new_data)})
                    _record_card_change_details(report, model=normalized, old_data=old_data, new_data=new_data)
                report.changed_models.append(normalized)
                if not dry_run:
                    path.write_text(_json_dumps(new_data), encoding="utf-8")
                    report.written_files.append(str(path))
            else:
                report.unchanged_models.append(normalized)
            if card.deprecation:
                report.deprecated_models.append(normalized)

        report.removed_models = sorted(previous_discovered - set(model_keys))
        report.changed_models = sorted(set(report.changed_models) | set(report.removed_models))
        index_path = self.config.registry_dir / "models.json"
        index_data = {"models": index_model_keys, "source": "provider_discovery", "updated_at": date.today().isoformat()}
        if not dry_run:
            index_path.write_text(_json_dumps(index_data), encoding="utf-8")
            report.written_files.append(str(index_path))

        if not model_keys:
            report.warnings.append("No provider models were discovered.")
        removed_allowed = sorted(set(report.removed_models) & {ModelRef.parse(model).key for model in self.config.models.allow})
        if removed_allowed:
            report.requires_confirmation = True
            report.warnings.append(
                "Allowed models no longer appeared in provider discovery: " + ", ".join(removed_allowed)
            )
        report.diff = {
            "added": sorted(report.added_models),
            "removed": report.removed_models,
            "changed": changed,
            "unchanged": len(report.unchanged_models),
        }
        self._cards = None
        report.model_states = self.model_states(
            models=set(model_keys) | set(report.removed_models),
            discovered_keys=model_keys,
        )
        return report

    def snapshot_create(self, name: str | None = None, *, allowed_only: bool = False) -> dict[str, Any]:
        """Persist a reproducible snapshot of the current registry state."""

        self.config.ensure_project_dirs()
        snapshot_name = self._normalize_snapshot_name(name or self._default_snapshot_name())
        path = self._snapshot_path(snapshot_name)
        if path.exists():
            raise CrupierConfigError(f"Registry snapshot {snapshot_name!r} already exists.")

        data = self._snapshot_data(snapshot_name, allowed_only=allowed_only)
        path.write_text(_json_dumps(data), encoding="utf-8")
        return {**self._snapshot_summary(data, path), "path": str(path)}

    def snapshot_list(self) -> builtins.list[dict[str, Any]]:
        """Return summaries for project registry snapshots."""

        self.config.ensure_project_dirs()
        snapshots: builtins.list[dict[str, Any]] = []
        for path in sorted(self.config.registry_snapshots_dir.glob("*.json")):
            data = self._read_snapshot_path(path)
            snapshots.append(self._snapshot_summary(data, path))
        return snapshots

    def snapshot_diff(self, left: str, right: str = "current") -> dict[str, Any]:
        """Compare two snapshots, or a snapshot against the current registry."""

        if left == "current" and right != "current":
            right_data = self._load_snapshot(right)
            left_data = self._snapshot_data("current", allowed_only=bool(right_data.get("allowed_only", False)))
        else:
            left_data = self._load_snapshot(left)
            right_data = (
                self._snapshot_data("current", allowed_only=bool(left_data.get("allowed_only", False)))
                if right == "current"
                else self._load_snapshot(right)
            )
        left_cards = self._snapshot_cards(left_data)
        right_cards = self._snapshot_cards(right_data)
        left_keys = set(left_cards)
        right_keys = set(right_cards)
        common = left_keys & right_keys
        changed: list[dict[str, Any]] = []
        for model_key in sorted(common):
            if left_cards[model_key] == right_cards[model_key]:
                continue
            fields = sorted(
                field
                for field in set(left_cards[model_key]) | set(right_cards[model_key])
                if left_cards[model_key].get(field) != right_cards[model_key].get(field)
            )
            changed.append({"model": model_key, "fields": fields})

        return {
            "left": self._snapshot_summary(left_data),
            "right": self._snapshot_summary(right_data),
            "added": sorted(right_keys - left_keys),
            "removed": sorted(left_keys - right_keys),
            "changed": changed,
            "unchanged": len(common) - len(changed),
        }

    def snapshot_use(self, name: str, *, restore_allowlist: bool = False) -> dict[str, Any]:
        """Restore local capability cards from a saved snapshot."""

        data = self._load_snapshot(name)
        snapshot_name = data["name"]
        cards = {
            ModelRef.parse(model_key).key: CapabilityCard.from_dict(card_data)
            for model_key, card_data in self._snapshot_cards(data).items()
        }
        self.config.ensure_project_dirs()

        written_files: list[str] = []
        expected_filenames: set[str] = set()
        for model_key, card in sorted(cards.items()):
            path = self.config.capability_cards_dir / _card_filename(model_key)
            expected_filenames.add(path.name)
            new_data = card.to_dict()
            old_data = None
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    old_data = json.load(handle)
            if old_data != new_data:
                path.write_text(_json_dumps(new_data), encoding="utf-8")
                written_files.append(str(path))

        removed_files: list[str] = []
        for path in sorted(self.config.capability_cards_dir.glob("*.json")):
            if path.name not in expected_filenames:
                path.unlink()
                removed_files.append(str(path))

        index_path = self.config.registry_dir / "models.json"
        index_data = {
            "models": sorted(cards),
            "source": "registry_snapshot",
            "snapshot": snapshot_name,
            "used_at": _now_iso(),
        }
        index_path.write_text(_json_dumps(index_data), encoding="utf-8")
        written_files.append(str(index_path))

        if restore_allowlist:
            allowlist = [ModelRef.parse(model_key).key for model_key in data.get("allowlist", [])]
            write_models_allow(self.config.root, allowlist, replace=True)
            self.config = CrupierConfig.from_toml(self.config.root)

        self._cards = None
        return {
            "snapshot": snapshot_name,
            "restored_models": sorted(cards),
            "written_files": written_files,
            "removed_files": removed_files,
            "allowlist_restored": restore_allowlist,
        }

    @staticmethod
    def _default_snapshot_name() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"reg_{stamp}"

    @staticmethod
    def _normalize_snapshot_name(name: str) -> str:
        normalized = name[:-5] if name.endswith(".json") else name
        if not normalized or normalized in {".", ".."}:
            raise CrupierConfigError("Registry snapshot name cannot be empty.")
        if "/" in normalized or "\\" in normalized or not SNAPSHOT_NAME_RE.fullmatch(normalized):
            raise CrupierConfigError(
                "Registry snapshot names may only contain letters, numbers, underscores, dots, and dashes."
            )
        return normalized

    def _snapshot_path(self, name: str) -> Path:
        return self.config.registry_snapshots_dir / f"{self._normalize_snapshot_name(name)}.json"

    def _snapshot_data(self, name: str, *, allowed_only: bool) -> dict[str, Any]:
        cards = self.list(allowed_only=allowed_only)
        allowlist = [ModelRef.parse(model_key).key for model_key in self.config.models.allow]
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "name": name,
            "created_at": _now_iso(),
            "project": self.config.project.name,
            "allowed_only": allowed_only,
            "allowlist": allowlist,
            "cards": {card.model_ref.key: card.to_dict() for card in cards},
        }

    def _load_snapshot(self, name: str) -> dict[str, Any]:
        if name == "current":
            return self._snapshot_data("current", allowed_only=False)
        path = self._snapshot_path(name)
        if not path.exists():
            raise CrupierConfigError(f"Registry snapshot {self._normalize_snapshot_name(name)!r} does not exist.")
        return self._read_snapshot_path(path)

    @staticmethod
    def _read_snapshot_path(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise CrupierConfigError(f"Unsupported registry snapshot schema in {path}.")
        if "name" not in data or "cards" not in data:
            raise CrupierConfigError(f"Invalid registry snapshot in {path}.")
        return data

    @staticmethod
    def _snapshot_cards(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        cards = data.get("cards", {})
        if not isinstance(cards, dict):
            raise CrupierConfigError("Invalid registry snapshot: cards must be an object.")
        return {ModelRef.parse(model_key).key: dict(card_data) for model_key, card_data in cards.items()}

    @staticmethod
    def _snapshot_summary(data: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
        summary = {
            "name": data.get("name"),
            "created_at": data.get("created_at"),
            "project": data.get("project"),
            "allowed_only": bool(data.get("allowed_only", False)),
            "card_count": len(data.get("cards", {})),
            "allowlist_count": len(data.get("allowlist", [])),
        }
        if path is not None:
            summary["path"] = str(path)
        return summary
