from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str, **kwargs) -> list[float]:
        raise NotImplementedError
