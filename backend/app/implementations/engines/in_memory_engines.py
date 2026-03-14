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

    def _tokenize(self, text: str) -> set[str]:
        compact = text.strip().lower()
        if not compact:
            return set()
        if " " in compact:
            return set([part for part in re.split(r"\s+", compact) if part])
        # 中文等无空格文本时，按字符 token 粗粒度切分
        return set([ch for ch in compact if ch.strip()])

    def _extract_query_entities(self, query: str) -> set[str]:
        # 抽取英文实体样式与中文词块，作为结构化对齐的 query 线索。
        caps = set(token.strip(" ,.?!:;()[]{}\"'") for token in query.split() if token[:1].isupper())
        cjk_chunks = set(re.findall(r"[\u4e00-\u9fff]{2,}", query))
        return set([x.lower() for x in [*caps, *cjk_chunks] if x])

    def _score_lexical(self, query: str, content: str) -> float:
        query_terms = self._tokenize(query)
        if not query_terms:
            return 0.0
        terms = self._tokenize(content)
        overlap = len(query_terms.intersection(terms))
        return overlap / max(len(query_terms), 1)

    def _score_triples(self, query: str, metadata: dict[str, Any]) -> float:
        triples = metadata.get("triples", [])
        if not isinstance(triples, list) or not triples:
            return 0.0

        valid = []
        for item in triples:
            if not isinstance(item, dict):
                continue
            s = str(item.get("subject", "")).strip()
            p = str(item.get("predicate", "")).strip()
            o = str(item.get("object", "")).strip()
            if s and p and o:
                valid.append((s, p, o))

        if not valid:
            return 0.0

        query_terms = self._tokenize(query)
        query_entities = self._extract_query_entities(query)

        triple_term_tokens = set()
        triple_entities = set()
        for s, p, o in valid:
            triple_term_tokens.update(self._tokenize(s))
            triple_term_tokens.update(self._tokenize(p))
            triple_term_tokens.update(self._tokenize(o))
            triple_entities.add(s.lower())
            triple_entities.add(o.lower())

        lexical = len(query_terms.intersection(triple_term_tokens)) / max(len(query_terms), 1)
        entity_alignment = 0.0
        if query_entities:
            entity_alignment = len(query_entities.intersection(triple_entities)) / len(query_entities)

        # 结构完整性：有效三元组占比
        completeness = len(valid) / max(len(triples), 1)
        return self._combine_structural_components(lexical, entity_alignment, completeness)

    def _score_attributes(self, query: str, metadata: dict[str, Any]) -> float:
        attrs = metadata.get("attributes", [])
        if not isinstance(attrs, list) or not attrs:
            return 0.0

        valid = []
        for item in attrs:
            if not isinstance(item, dict):
                continue
            e = str(item.get("entity", "")).strip()
            a = str(item.get("attribute", "")).strip()
            v = str(item.get("value", "")).strip()
            if e and a and v:
                valid.append((e, a, v))

        if not valid:
            return 0.0

        query_terms = self._tokenize(query)
        query_entities = self._extract_query_entities(query)

        attr_tokens = set()
        attr_entities = set()
        for e, a, v in valid:
            attr_tokens.update(self._tokenize(e))
            attr_tokens.update(self._tokenize(a))
            attr_tokens.update(self._tokenize(v))
            attr_entities.add(e.lower())

        lexical = len(query_terms.intersection(attr_tokens)) / max(len(query_terms), 1)
        entity_alignment = 0.0
        if query_entities:
            entity_alignment = len(query_entities.intersection(attr_entities)) / len(query_entities)

        completeness = len(valid) / max(len(attrs), 1)
        return self._combine_structural_components(lexical, entity_alignment, completeness)

    def _normalize_weights(self, *weights: float) -> list[float]:
        clipped = [max(0.0, float(w)) for w in weights]
        total = sum(clipped)
        if total <= 0:
            return [1.0 / len(clipped)] * len(clipped)
        return [w / total for w in clipped]

    def _combine_structural_components(self, lexical: float, entity_alignment: float, completeness: float) -> float:
        if self.engine_type == EngineType.graph_engine:
            lw, ew, cw = self._normalize_weights(
                settings.graph_relevance_lexical_weight,
                settings.graph_relevance_entity_weight,
                settings.graph_relevance_completeness_weight,
            )
        elif self.engine_type == EngineType.relational_engine:
            lw, ew, cw = self._normalize_weights(
                settings.relational_relevance_lexical_weight,
                settings.relational_relevance_entity_weight,
                settings.relational_relevance_completeness_weight,
            )
        else:
            lw, ew, cw = self._normalize_weights(0.45, 0.35, 0.20)
        return lw * lexical + ew * entity_alignment + cw * completeness

    def _combine_final_relevance(self, lexical: float, structural: float, score_hint: float) -> float:
        if structural > 0:
            if self.engine_type == EngineType.graph_engine:
                sw, hw = self._normalize_weights(1.0, settings.graph_relevance_hint_weight)
            elif self.engine_type == EngineType.relational_engine:
                sw, hw = self._normalize_weights(1.0, settings.relational_relevance_hint_weight)
            else:
                sw, hw = self._normalize_weights(0.9, 0.1)
            return sw * structural + hw * score_hint

        if self.engine_type == EngineType.graph_engine:
            lw, hw = self._normalize_weights(
                settings.graph_relevance_fallback_lexical_weight,
                settings.graph_relevance_fallback_hint_weight,
            )
        elif self.engine_type == EngineType.relational_engine:
            lw, hw = self._normalize_weights(
                settings.relational_relevance_fallback_lexical_weight,
                settings.relational_relevance_fallback_hint_weight,
            )
        else:
            lw, hw = self._normalize_weights(0.9, 0.1)
        return lw * lexical + hw * score_hint

    def _score_structural(self, query: str, metadata: dict[str, Any]) -> float:
        if "triples" in metadata:
            return self._score_triples(query, metadata)
        if "attributes" in metadata:
            return self._score_attributes(query, metadata)
        return 0.0

    def save(self, request: EngineSaveRequest) -> EngineSaveResult:
        for chunk in request.chunks:
            enriched_meta = dict(chunk.metadata)
            enriched_meta["_score_hint"] = float(chunk.score_hint)
            self._store[chunk.session_id].append((chunk.chunk_id, chunk.content, enriched_meta))
        return EngineSaveResult(
            engine=self.engine_type,
            saved_count=len(request.chunks),
            message=f"Saved {len(request.chunks)} chunks into {self.engine_type}",
        )

    def search(self, request: EngineSearchRequest) -> EngineSearchResult:
        candidates = self._store.get(request.session_id, [])
        # 结构化评分：词匹配 + 三元组/属性质量 + score_hint 融合（权重可配置）
        scored: list[MemoryHit] = []
        for chunk_id, content, metadata in candidates:
            lexical = self._score_lexical(request.query, content)
            structural = self._score_structural(request.query, metadata)
            score_hint = max(0.0, min(1.0, float(metadata.get("_score_hint", 1.0))))

            relevance = self._combine_final_relevance(lexical, structural, score_hint)

            relevance = max(0.0, min(1.0, relevance))
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
