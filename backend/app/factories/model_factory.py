from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import httpx

from app.config import settings
from app.models.contracts import ProviderType
from app.services.request_audit import write_audit_event


@dataclass
class LLMClient:
    provider: ProviderType
    model: str
    endpoint: str
    api_key: str = ""

    def generate(
        self,
        prompt: str,
        system_prompt: str = "你是一个严谨的评估助手。",
        purpose: str = "general",
    ) -> str:
        started = perf_counter()
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
                write_audit_event(
                    "llm_generate",
                    {
                        "provider": self.provider,
                        "model": self.model,
                        "endpoint": self.endpoint,
                        "purpose": purpose,
                        "ok": True,
                        "duration_ms": round((perf_counter() - started) * 1000, 2),
                        "prompt_preview": prompt[:200],
                    },
                )
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
                write_audit_event(
                    "llm_generate",
                    {
                        "provider": self.provider,
                        "model": self.model,
                        "endpoint": self.endpoint,
                        "purpose": purpose,
                        "ok": True,
                        "duration_ms": round((perf_counter() - started) * 1000, 2),
                        "prompt_preview": prompt[:200],
                    },
                )
                return data.get("response", "")

            # Local provider 先保留可运行占位，后续可接 transformers pipeline
            write_audit_event(
                "llm_generate",
                {
                    "provider": self.provider,
                    "model": self.model,
                    "endpoint": self.endpoint,
                    "purpose": purpose,
                    "ok": True,
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                    "prompt_preview": prompt[:200],
                    "note": "local placeholder",
                },
            )
            return f"[local:{self.model}] {prompt[:500]}"
        except Exception as exc:
            write_audit_event(
                "llm_generate",
                {
                    "provider": self.provider,
                    "model": self.model,
                    "endpoint": self.endpoint,
                    "purpose": purpose,
                    "ok": False,
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                    "prompt_preview": prompt[:200],
                    "error": str(exc),
                },
            )
            return f"LLM provider call failed: {exc}"


@dataclass
class EmbeddingClient:
    provider: ProviderType
    model: str
    endpoint: str
    api_key: str = ""

    def embed(self, text: str) -> list[float]:
        started = perf_counter()
        if self.provider == ProviderType.api:
            try:
                url = f"{self.endpoint.rstrip('/')}/embeddings"
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                payload = {"model": self.model, "input": text}
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                write_audit_event(
                    "embedding",
                    {
                        "provider": self.provider,
                        "model": self.model,
                        "endpoint": self.endpoint,
                        "ok": True,
                        "duration_ms": round((perf_counter() - started) * 1000, 2),
                        "text_preview": text[:200],
                    },
                )
                return data["data"][0]["embedding"]
            except Exception as exc:
                write_audit_event(
                    "embedding",
                    {
                        "provider": self.provider,
                        "model": self.model,
                        "endpoint": self.endpoint,
                        "ok": False,
                        "duration_ms": round((perf_counter() - started) * 1000, 2),
                        "text_preview": text[:200],
                        "error": str(exc),
                    },
                )
                pass

        if self.provider == ProviderType.ollama:
            try:
                url = f"{self.endpoint.rstrip('/')}/api/embeddings"
                payload = {"model": self.model, "prompt": text}
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                write_audit_event(
                    "embedding",
                    {
                        "provider": self.provider,
                        "model": self.model,
                        "endpoint": self.endpoint,
                        "ok": True,
                        "duration_ms": round((perf_counter() - started) * 1000, 2),
                        "text_preview": text[:200],
                    },
                )
                return data.get("embedding", [])
            except Exception as exc:
                write_audit_event(
                    "embedding",
                    {
                        "provider": self.provider,
                        "model": self.model,
                        "endpoint": self.endpoint,
                        "ok": False,
                        "duration_ms": round((perf_counter() - started) * 1000, 2),
                        "text_preview": text[:200],
                        "error": str(exc),
                    },
                )
                pass

        # 本地回退向量：保证在 provider 不可用时流程仍可联调
        base = float(len(text) % 10)
        write_audit_event(
            "embedding",
            {
                "provider": self.provider,
                "model": self.model,
                "endpoint": self.endpoint,
                "ok": True,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
                "text_preview": text[:200],
                "note": "local fallback embedding",
            },
        )
        return [base + 0.1 * i for i in range(32)]


class ProviderFactory:
    @staticmethod
    def _normalize_provider(raw_provider: ProviderType | str | None) -> ProviderType:
        provider = raw_provider or settings.default_llm_provider
        return provider if isinstance(provider, ProviderType) else ProviderType(str(provider))

    @staticmethod
    def _build_function_llm(function: str, provider_override: ProviderType | None = None) -> LLMClient:
        provider_key = f"{function}_llm_provider"
        provider = ProviderFactory._normalize_provider(provider_override or getattr(settings, provider_key, ""))

        if provider == ProviderType.api:
            return LLMClient(
                provider=provider,
                model=getattr(settings, f"{function}_api_model") or settings.openai_chat_model,
                endpoint=getattr(settings, f"{function}_api_base_url") or settings.openai_base_url,
                api_key=getattr(settings, f"{function}_api_key") or settings.openai_api_key,
            )
        if provider == ProviderType.ollama:
            return LLMClient(
                provider=provider,
                model=getattr(settings, f"{function}_ollama_model") or settings.ollama_llm_model,
                endpoint=getattr(settings, f"{function}_ollama_base_url") or settings.ollama_base_url,
            )
        return LLMClient(
            provider=provider,
            model=getattr(settings, f"{function}_local_model_path") or settings.local_llm_model_path or "local-llm",
            endpoint="local",
        )

    @staticmethod
    def build_chat_llm(provider_override: ProviderType | None = None) -> LLMClient:
        return ProviderFactory._build_function_llm("chat", provider_override)

    @staticmethod
    def build_judge_llm(provider_override: ProviderType | None = None) -> LLMClient:
        return ProviderFactory._build_function_llm("judge", provider_override)

    @staticmethod
    def build_llm(provider: ProviderType) -> LLMClient:
        # 兼容旧调用，默认等价于按 chat 功能构建。
        return ProviderFactory.build_chat_llm(provider_override=provider)

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
