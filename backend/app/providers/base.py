from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Common contract for all language-model adapters used by the app."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Return a text answer for a prompt in the provider-specific format."""
        raise NotImplementedError


class BaseEmbeddingProvider(ABC):
    """Common contract for all embedding adapters used to build vector search indexes."""

    @abstractmethod
    def embed(self, text: str, **kwargs) -> list[float]:
        """Convert a text snippet into a fixed-length embedding vector."""
        raise NotImplementedError
