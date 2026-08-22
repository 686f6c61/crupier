"""Provider adapters."""

from .base import (
    AdapterResponse,
    CapabilityProbeAdapter,
    EmbeddingProviderAdapter,
    EmbeddingResponse,
    ModelDiscoveryAdapter,
    OperationProviderAdapter,
    OperationResponse,
    ProviderAdapter,
    ProviderModel,
)
from .factory import build_default_adapters
from .nan import NaNAdapter
from .openai_compatible import OpenAICompatibleAdapter
from .openrouter import OpenRouterAdapter

__all__ = [
    "AdapterResponse",
    "CapabilityProbeAdapter",
    "EmbeddingProviderAdapter",
    "EmbeddingResponse",
    "ModelDiscoveryAdapter",
    "NaNAdapter",
    "OpenAICompatibleAdapter",
    "OpenRouterAdapter",
    "OperationProviderAdapter",
    "OperationResponse",
    "ProviderAdapter",
    "ProviderModel",
    "build_default_adapters",
]
