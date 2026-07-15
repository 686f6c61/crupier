"""Provider adapters."""

from .base import (
    AdapterResponse,
    EmbeddingProviderAdapter,
    EmbeddingResponse,
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
    "EmbeddingResponse",
    "EmbeddingProviderAdapter",
    "OperationProviderAdapter",
    "OperationResponse",
    "OpenRouterAdapter",
    "OpenAICompatibleAdapter",
    "NaNAdapter",
    "ProviderAdapter",
    "ProviderModel",
    "build_default_adapters",
]
