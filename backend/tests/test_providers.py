import os

import pytest

from app.providers.factory import get_embedding_provider, get_llm_provider
from app.providers.fake import FakeEmbeddingProvider, FakeLLMProvider
from app.providers.real import LiteLLMProvider


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


def test_real_provider_exposes_token_usage_without_changing_text_contract(monkeypatch):
    """Verify provider token counters are available for telemetry while generate still returns plain text."""
    class FakeResponse:
        """Provide the minimal successful HTTP response expected by the provider adapter."""

        def raise_for_status(self):
            """Match the HTTPX response method when the request succeeds."""

        def json(self):
            """Return a response body with text and aggregate token usage."""
            return {
                "choices": [{"message": {"content": "safe answer"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }

    response = FakeResponse()
    monkeypatch.setattr("app.providers.real.httpx.post", lambda *args, **kwargs: response)
    provider = LiteLLMProvider(api_key="test-key")

    result = provider.generate("private prompt")

    assert result == "safe answer"
    assert provider.last_usage == {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
