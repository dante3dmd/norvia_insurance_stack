import os

import pytest

from app.providers.factory import get_embedding_provider, get_llm_provider
from app.providers.fake import FakeEmbeddingProvider, FakeLLMProvider


def test_fake_embedding_provider_returns_fixed_vector():
    provider = FakeEmbeddingProvider()
    embedding = provider.embed("cobertura de auto")

    assert isinstance(embedding, list)
    assert len(embedding) == 1536
    assert all(value == 0.1 for value in embedding)


def test_fake_llm_provider_returns_deterministic_response():
    provider = FakeLLMProvider()
    response = provider.generate("¿Qué cubre la póliza?")

    assert isinstance(response, str)
    assert "respuesta falsa" in response.lower()
    assert "¿qué cubre la póliza?" in response.lower()


def test_factory_uses_fake_provider_without_network_or_api_keys(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    llm = get_llm_provider()
    embedding = get_embedding_provider()

    assert isinstance(llm, FakeLLMProvider)
    assert isinstance(embedding, FakeEmbeddingProvider)
    assert len(embedding.embed("texto de prueba")) == 1536
    assert "respuesta falsa" in llm.generate("texto de prueba").lower()
