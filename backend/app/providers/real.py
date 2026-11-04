from __future__ import annotations

import os

import httpx

from .base import BaseEmbeddingProvider, BaseLLMProvider


class LiteLLMProvider(BaseLLMProvider):
    """Adapter for a real chat-completions backend such as LiteLLM or OpenAI."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Keep the provider configuration in environment variables so switching models is a config change only."""
        self.model = model or os.getenv("REAL_LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("REAL_LLM_BASE_URL", "https://api.openai.com/v1")
        self.timeout = timeout
        self.last_usage: dict[str, int] = {}

    def generate(self, prompt: str, **kwargs) -> str:
        """Send a prompt to the configured chat-completions API and return the model text."""
        self.last_usage = {}
        if not self.api_key:
            raise RuntimeError("No API key configured for the real LLM provider.")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        self.last_usage = {
            key: usage[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if isinstance(usage.get(key), int) and usage[key] >= 0
        }
        return data["choices"][0]["message"]["content"]


class LiteLLMEmbeddingProvider(BaseEmbeddingProvider):
    """Adapter for a real embeddings endpoint such as LiteLLM or OpenAI."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Store the model and credentials needed to create vector embeddings for retrieval."""
        self.model = model or os.getenv("REAL_EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("REAL_EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        self.timeout = timeout

    def embed(self, text: str, **kwargs) -> list[float]:
        """Call the configured embeddings API and return the vector representation for the text."""
        if not self.api_key:
            raise RuntimeError("No API key configured for the real embedding provider.")

        payload = {"model": self.model, "input": text, **kwargs}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
