from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config import settings
from app.models.contracts import ProviderType


@dataclass
class LLMClient:
    provider: ProviderType
    model: str
    endpoint: str
    api_key: str = ""

    def generate(self, prompt: str, system_prompt: str = "你是一个严谨的评估助手。") -> str:
        try:
            if self.provider == ProviderType.api:
                url = f"{self.endpoint.rstrip('/')}/chat/completions"
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": self.model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                }
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                return data["choices"][0]["message"]["content"]

            if self.provider == ProviderType.ollama:
                url = f"{self.endpoint.rstrip('/')}/api/generate"
                payload = {
                    "model": self.model,
                    "prompt": f"{system_prompt}\n\n{prompt}",
                    "stream": False,
                    "options": {"temperature": 0},
                }
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                return data.get("response", "")

            # Local provider 先保留可运行占位，后续可接 transformers pipeline
            return f"[local:{self.model}] {prompt[:500]}"
        except Exception as exc:
            return f"LLM provider call failed: {exc}"


@dataclass
class EmbeddingClient:
    provider: ProviderType
    model: str
    endpoint: str
    api_key: str = ""

    def embed(self, text: str) -> list[float]:
        if self.provider == ProviderType.api:
            try:
                url = f"{self.endpoint.rstrip('/')}/embeddings"
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                payload = {"model": self.model, "input": text}
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                return data["data"][0]["embedding"]
            except Exception:
                pass

        if self.provider == ProviderType.ollama:
            try:
                url = f"{self.endpoint.rstrip('/')}/api/embeddings"
                payload = {"model": self.model, "prompt": text}
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                return data.get("embedding", [])
            except Exception:
                pass

        # 本地回退向量：保证在 provider 不可用时流程仍可联调
        base = float(len(text) % 10)
        return [base + 0.1 * i for i in range(32)]


class ProviderFactory:
    @staticmethod
    def build_llm(provider: ProviderType) -> LLMClient:
        if provider == ProviderType.api:
            return LLMClient(
                provider=provider,
                model=settings.openai_chat_model,
                endpoint=settings.openai_base_url,
                api_key=settings.openai_api_key,
            )
        if provider == ProviderType.ollama:
            return LLMClient(provider=provider, model=settings.ollama_llm_model, endpoint=settings.ollama_base_url)
        return LLMClient(provider=provider, model=settings.local_llm_model_path or "local-llm", endpoint="local")

    @staticmethod
    def build_embedding(provider: ProviderType) -> EmbeddingClient:
        if provider == ProviderType.api:
            return EmbeddingClient(
                provider=provider,
                model=settings.openai_embed_model,
                endpoint=settings.openai_base_url,
                api_key=settings.openai_api_key,
            )
        if provider == ProviderType.ollama:
            # 默认重点支持 Qwen/Qwen3-Embedding-0.6B
            return EmbeddingClient(provider=provider, model=settings.ollama_embed_model, endpoint=settings.ollama_base_url)
        return EmbeddingClient(provider=provider, model=settings.local_embed_model_path or "local-embedding", endpoint="local")
