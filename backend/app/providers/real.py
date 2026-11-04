from __future__ import annotations

import os

import httpx

from .base import BaseEmbeddingProvider, BaseLLMProvider


class LiteLLMProvider(BaseLLMProvider):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model or os.getenv("REAL_LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("REAL_LLM_BASE_URL", "https://api.openai.com/v1")
        self.timeout = timeout

    def generate(self, prompt: str, **kwargs) -> str:
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
        return data["choices"][0]["message"]["content"]


class LiteLLMEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model or os.getenv("REAL_EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("REAL_EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        self.timeout = timeout

    def embed(self, text: str, **kwargs) -> list[float]:
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
