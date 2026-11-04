from __future__ import annotations

from .base import BaseEmbeddingProvider, BaseLLMProvider


class FakeLLMProvider(BaseLLMProvider):
    """Deterministic provider for offline development and tests."""

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        normalized = prompt.strip()
        return f"Respuesta falsa para la consulta: {normalized}. Contexto local y determinista, sin red ni claves API."


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """Fixed embedding for tests: deterministic and easy to validate."""

    def embed(self, text: str, **kwargs) -> list[float]:
        return [0.1 for _ in range(1536)]
