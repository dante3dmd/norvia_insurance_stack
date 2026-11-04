from __future__ import annotations

import os

from .base import BaseEmbeddingProvider, BaseLLMProvider
from .fake import FakeEmbeddingProvider, FakeLLMProvider
from .real import LiteLLMEmbeddingProvider, LiteLLMProvider


def get_llm_provider() -> BaseLLMProvider:
    """Instantiate the configured LLM adapter, defaulting to the deterministic fake provider for local work."""
    provider_name = os.getenv("LLM_PROVIDER", "fake").lower()
    if provider_name == "fake":
        return FakeLLMProvider()
    if provider_name in {"real", "litellm", "openai"}:
        return LiteLLMProvider()
    raise ValueError(f"Unsupported LLM provider: {provider_name}")


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Instantiate the configured embedding adapter so retrieval uses the right vector backend."""
    provider_name = os.getenv("EMBEDDING_PROVIDER", "fake").lower()
    if provider_name == "fake":
        return FakeEmbeddingProvider()
    if provider_name in {"real", "litellm", "openai"}:
        return LiteLLMEmbeddingProvider()
    raise ValueError(f"Unsupported embedding provider: {provider_name}")
