from __future__ import annotations

from collections import defaultdict
import math
import re
import threading
import time
from typing import Any

import chromadb

from app.config import settings
from app.core.interfaces import MemoryEngine
from app.models.contracts import EngineSaveRequest, EngineSaveResult, EngineSearchRequest, EngineSearchResult, EngineType, MemoryHit


class _BaseInMemoryEngine(MemoryEngine):
    def __init__(self, engine_type: EngineType) -> None:
        self.engine_type = engine_type
        self._store: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)

    def save(self, request: EngineSaveRequest) -> EngineSaveResult:
        for chunk in request.chunks:
            self._store[chunk.session_id].append((chunk.chunk_id, chunk.content, chunk.metadata))
        return EngineSaveResult(
            engine=self.engine_type,
            saved_count=len(request.chunks),
            message=f"Saved {len(request.chunks)} chunks into {self.engine_type}",
        )

    def search(self, request: EngineSearchRequest) -> EngineSearchResult:
        candidates = self._store.get(request.session_id, [])
        query_terms = set(request.query.lower().split())

        # 演示检索分数：按 query 词命中数排序
        scored: list[MemoryHit] = []
        for chunk_id, content, metadata in candidates:
            terms = set(content.lower().split())
            overlap = len(query_terms.intersection(terms))
            relevance = overlap / max(len(query_terms), 1)
            scored.append(MemoryHit(chunk_id=chunk_id, content=content, relevance=relevance, metadata=metadata))

        scored.sort(key=lambda x: x.relevance, reverse=True)
        return EngineSearchResult(engine=self.engine_type, hits=scored[: request.top_k])


class VectorEngine(MemoryEngine):
    _collection_lock = threading.Lock()

    def __init__(self, embedding_client: Any, collection_name: str | None = None) -> None:
        self.embedding_client = embedding_client
        self.engine_type = EngineType.vector_engine
        self.collection_name = collection_name or settings.chroma_collection_name
        self.client = self._new_client()
        # Warm up tenant/collection state to reduce first-run race failures.
        self._get_collection(self.collection_name)

    def _new_client(self):
        return chromadb.PersistentClient(path=settings.chroma_persist_dir)

    def _is_transient_tenant_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return "could not connect to tenant" in msg or "default_tenant" in msg

    def _run_chroma_with_retry(self, fn):
        attempts = 3
        for idx in range(attempts):
            try:
                return fn()
            except Exception as exc:
                is_last = idx == attempts - 1
                if not self._is_transient_tenant_error(exc) or is_last:
                    raise
                time.sleep(0.25 * (idx + 1))
                self.client = self._new_client()

    def _get_collection(self, name: str):
        def create_or_get():
            return self.client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

        with self._collection_lock:
            return self._run_chroma_with_retry(create_or_get)

    def _collection(self):
        return self._get_collection(self.collection_name)

    def _tokenize(self, text: str) -> set[str]:
        compact = text.strip().lower()
        if not compact:
            return set()
        if " " in compact:
            return set([part for part in re.split(r"\s+", compact) if part])
        # 中文等无空格文本时，按字符 token 粗粒度切分
        return set([ch for ch in compact if ch.strip()])

    def _distance_to_relevance(self, distance: float, strategy: str) -> float:
        if strategy == "exp_decay":
            return max(0.0, min(1.0, math.exp(-distance)))
        if strategy == "linear":
            # cosine distance 常见范围 [0,2]
            return max(0.0, min(1.0, 1.0 - distance / 2.0))
        # 默认 inverse_distance
        return max(0.0, min(1.0, 1.0 / (1.0 + distance)))

    def save(self, request: EngineSaveRequest) -> EngineSaveResult:
        collection = self._collection()
        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in request.chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            embeddings.append(self.embedding_client.embed(chunk.content))
            metadatas.append(
                {
                    "session_id": chunk.session_id,
                    "tags": ",".join(chunk.tags),
                    "score_hint": float(chunk.score_hint),
                }
            )

        try:
            self._run_chroma_with_retry(
                lambda: collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
            )
        except Exception as exc:
            err = str(exc)
            if "dimension" in err.lower():
                raise ValueError(
                    "Chroma collection embedding dimension mismatch. "
                    f"collection={self.collection_name}. "
                    "Please switch retrieval.collection_name to a new value, or clear the old collection data."
                ) from exc
            raise
        return EngineSaveResult(
            engine=self.engine_type,
            saved_count=len(request.chunks),
            message=f"Saved {len(request.chunks)} chunks into Chroma collection={self.collection_name}",
        )

    def search(self, request: EngineSearchRequest) -> EngineSearchResult:
        collection_name = request.filters.get("collection_name", self.collection_name)
        min_relevance = float(request.filters.get("min_relevance", 0.0))
        similarity_strategy = str(request.filters.get("similarity_strategy", "inverse_distance"))
        keyword_rerank = bool(request.filters.get("keyword_rerank", False))
        collection = self._get_collection(collection_name)

        query_embedding = self.embedding_client.embed(request.query)
        result = self._run_chroma_with_retry(
            lambda: collection.query(
                query_embeddings=[query_embedding],
                n_results=request.top_k,
                where={"session_id": request.session_id},
            )
        )

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        dists = result.get("distances", [[]])[0]
        metas = result.get("metadatas", [[]])[0]

        hits: list[MemoryHit] = []
        query_tokens = self._tokenize(request.query)
        for item_id, content, dist, meta in zip(ids, docs, dists, metas):
            distance = float(dist) if dist is not None else 1.0
            relevance = self._distance_to_relevance(distance, similarity_strategy)

            if keyword_rerank:
                doc_tokens = self._tokenize(content)
                overlap = len(query_tokens.intersection(doc_tokens))
                keyword_score = overlap / max(len(query_tokens), 1)
                relevance = 0.7 * relevance + 0.3 * keyword_score

            if relevance < min_relevance:
                continue
            hits.append(MemoryHit(chunk_id=item_id, content=content, relevance=relevance, metadata=meta or {}))

        hits.sort(key=lambda x: x.relevance, reverse=True)

        return EngineSearchResult(engine=self.engine_type, hits=hits)


class GraphEngine(_BaseInMemoryEngine):
    def __init__(self) -> None:
        super().__init__(EngineType.graph_engine)


class RelationalEngine(_BaseInMemoryEngine):
    def __init__(self) -> None:
        super().__init__(EngineType.relational_engine)
