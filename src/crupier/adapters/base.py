"""Provider adapter contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol

from crupier.models import RequestEnvelope


@dataclass(slots=True)
class AdapterResponse:
    text: str
    raw: Any = None
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddingResponse:
    embeddings: list[list[float]]
    raw: Any = None
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderModel:
    id: str
    provider: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def model_ref(self) -> str:
        return f"{self.provider}:{self.id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "model_ref": self.model_ref,
            "name": self.name,
            "metadata": _jsonable(self.metadata),
        }


class ProviderAdapter(Protocol):
    provider: str

    def generate(self, *, model: str, prompt: str, request: RequestEnvelope) -> AdapterResponse:
        """Generate text for a normalized prompt."""

    def list_models(self) -> list[ProviderModel]:
        """List models available to the configured account/provider."""

    def probe_capability(self, *, model: str, probe: str, request: RequestEnvelope) -> AdapterResponse:
        """Run a provider-native capability probe when supported."""

    def embed(self, *, model: str, input: Any) -> EmbeddingResponse:
        """Create embeddings when the provider/model supports it."""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    return repr(value)
