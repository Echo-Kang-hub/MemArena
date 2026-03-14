from __future__ import annotations

import json
import importlib
import re
from collections import Counter
from typing import Any
from uuid import uuid4

from app.core.interfaces import MemoryProcessor
from app.models.contracts import (
    EntityExtractorMethod,
    MemoryChunk,
    ProcessorOutput,
    ProcessorType,
    RawConversationInput,
    SummarizerMethod,
)

def _load_sklearn_components() -> tuple[Any | None, Any | None]:
    try:
        cluster_module = importlib.import_module("sklearn.cluster")
        text_module = importlib.import_module("sklearn.feature_extraction.text")
        return getattr(cluster_module, "KMeans", None), getattr(text_module, "TfidfVectorizer", None)
    except Exception:
        return None, None


def _load_spacy_module() -> Any | None:
    try:
        return importlib.import_module("spacy")
    except Exception:
        return None


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？；;])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _tokenize_for_salience(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def _heuristic_summary(sentences: list[str], max_sentences: int = 3) -> str:
    if not sentences:
        return ""
    token_freq = Counter(token for sentence in sentences for token in _tokenize_for_salience(sentence))
    scored: list[tuple[int, float]] = []
    for idx, sentence in enumerate(sentences):
        tokens = _tokenize_for_salience(sentence)
        if not tokens:
            scored.append((idx, 0.0))
            continue
        score = sum(token_freq[t] for t in tokens) / len(tokens)
        scored.append((idx, score))
    keep = sorted(sorted(scored, key=lambda x: x[1], reverse=True)[:max_sentences], key=lambda x: x[0])
    return " ".join(sentences[idx] for idx, _ in keep)


def _maybe_load_spacy_model() -> Any | None:
    spacy_module = _load_spacy_module()
    if spacy_module is None:
        return None
    for model_name in ("en_core_web_sm", "xx_ent_wiki_sm"):
        try:
            return spacy_module.load(model_name)
        except Exception:
            continue
    return None


def _extract_entities_heuristic(text: str) -> list[str]:
    uppercase_tokens = [token.strip(",.?!:;()[]{}\"'") for token in text.split() if token[:1].isupper()]
    cjk_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    merged = [*uppercase_tokens, *cjk_chunks]
    deduped: list[str] = []
    seen: set[str] = set()
    for token in merged:
        normalized = token.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped[:20]


def _extract_json_text(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    obj_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if obj_match:
        return obj_match.group(0).strip()
    arr_match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if arr_match:
        return arr_match.group(0).strip()
    return stripped


def _build_chunk_id(session_id: str, suffix: str) -> str:
    return f"{session_id}-{suffix}-{uuid4().hex[:8]}"


class RawLoggerProcessor(MemoryProcessor):
    # 全量记录：原始文本直接落入记忆块
    def process(self, payload: RawConversationInput) -> ProcessorOutput:
        chunk = MemoryChunk(
            chunk_id=_build_chunk_id(payload.session_id, "raw"),
            session_id=payload.session_id,
            content=payload.message,
            tags=["raw", "full"],
            metadata=payload.metadata,
        )
        return ProcessorOutput(source=ProcessorType.raw_logger, chunks=[chunk])


class SummarizerProcessor(MemoryProcessor):
    def __init__(self, method: SummarizerMethod = SummarizerMethod.llm, llm_client: Any | None = None) -> None:
        self.method = method
        self.llm_client = llm_client

    def _llm_summary(self, text: str) -> str:
        if self.llm_client is None:
            sentences = _split_sentences(text)
            return _heuristic_summary(sentences, max_sentences=3) or text[:220]

        prompt = (
            "请将以下用户输入总结为 3-5 条可检索记忆点。"
            "要求保留时间、地点、人物、任务和约束条件。"
            "输出使用短句，使用中文。\n\n"
            f"原文:\n{text}"
        )
        response = self.llm_client.generate(prompt, system_prompt="你是记忆压缩助手。", purpose="summarizer_llm")
        return response.strip() or text[:220]

    def _kmeans_summary(self, text: str) -> tuple[str, dict[str, Any]]:
        sentences = _split_sentences(text)
        if len(sentences) <= 2:
            return " ".join(sentences) if sentences else text[:220], {"clusters": len(sentences), "fallback": "short_input"}

        KMeans, TfidfVectorizer = _load_sklearn_components()
        if KMeans is None or TfidfVectorizer is None:
            return _heuristic_summary(sentences, max_sentences=3), {"clusters": min(3, len(sentences)), "fallback": "sklearn_missing"}

        cluster_count = min(3, len(sentences))
        try:
            tfidf = TfidfVectorizer(max_features=2048, ngram_range=(1, 2))
            matrix = tfidf.fit_transform(sentences)
            kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
            kmeans.fit(matrix)
            distances = kmeans.transform(matrix)
            labels = list(kmeans.labels_)

            selected_indices: list[int] = []
            for cluster_id in range(cluster_count):
                members = [idx for idx, lbl in enumerate(labels) if lbl == cluster_id]
                if not members:
                    continue
                representative = min(members, key=lambda idx: float(distances[idx][cluster_id]))
                selected_indices.append(representative)

            selected_indices = sorted(set(selected_indices))
            if not selected_indices:
                return _heuristic_summary(sentences, max_sentences=3), {"clusters": cluster_count, "fallback": "empty_cluster_selection"}
            return " ".join(sentences[idx] for idx in selected_indices), {"clusters": cluster_count}
        except Exception:
            return _heuristic_summary(sentences, max_sentences=3), {"clusters": cluster_count, "fallback": "kmeans_runtime_error"}

    # 成熟版摘要：支持 LLM 与 K-Means 聚类摘要
    def process(self, payload: RawConversationInput) -> ProcessorOutput:
        if self.method == SummarizerMethod.kmeans:
            summary, method_meta = self._kmeans_summary(payload.message)
        else:
            summary = self._llm_summary(payload.message)
            method_meta = {}

        chunk = MemoryChunk(
            chunk_id=_build_chunk_id(payload.session_id, "sum"),
            session_id=payload.session_id,
            content=f"摘要: {summary}",
            tags=["summary", f"method:{self.method.value}"],
            metadata={"method": self.method.value, **method_meta, **payload.metadata},
        )
        return ProcessorOutput(source=ProcessorType.summarizer, chunks=[chunk])


class EntityExtractorProcessor(MemoryProcessor):
    def __init__(self, method: EntityExtractorMethod = EntityExtractorMethod.llm_triple, llm_client: Any | None = None) -> None:
        self.method = method
        self.llm_client = llm_client
        self._spacy_model = _maybe_load_spacy_model() if method in {
            EntityExtractorMethod.spacy_llm_triple,
            EntityExtractorMethod.spacy_llm_attribute,
        } else None

    def _extract_candidates(self, text: str) -> list[str]:
        if self._spacy_model is not None:
            try:
                doc = self._spacy_model(text)
                entities: list[str] = []
                seen: set[str] = set()
                for ent in doc.ents:
                    value = ent.text.strip()
                    if not value:
                        continue
                    lowered = value.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    entities.append(value)
                if entities:
                    return entities[:20]
            except Exception:
                pass
        return _extract_entities_heuristic(text)

    def _generate_triples(self, text: str, entities: list[str]) -> list[dict[str, str]]:
        if self.llm_client is None:
            if len(entities) >= 2:
                return [
                    {"subject": entities[0], "predicate": "related_to", "object": entities[1]},
                ]
            if len(entities) == 1:
                return [{"subject": entities[0], "predicate": "mentioned_in", "object": "conversation"}]
            return []

        entity_hint = "、".join(entities) if entities else "(none)"
        prompt = (
            "从下面文本中抽取知识三元组，输出 JSON 对象，格式为"
            " {\"triples\":[{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\"}]}。"
            "不要输出解释。\n"
            f"候选实体: {entity_hint}\n"
            f"文本: {text}"
        )
        raw = self.llm_client.generate(prompt, system_prompt="你是信息抽取引擎。", purpose="entity_extract_triple")
        try:
            parsed = json.loads(_extract_json_text(raw))
            triples = parsed.get("triples", []) if isinstance(parsed, dict) else []
            normalized: list[dict[str, str]] = []
            for item in triples:
                if not isinstance(item, dict):
                    continue
                subject = str(item.get("subject", "")).strip()
                predicate = str(item.get("predicate", "")).strip()
                obj = str(item.get("object", "")).strip()
                if subject and predicate and obj:
                    normalized.append({"subject": subject, "predicate": predicate, "object": obj})
            return normalized
        except Exception:
            return []

    def _generate_attributes(self, text: str, entities: list[str]) -> list[dict[str, str]]:
        if self.llm_client is None:
            return [{"entity": ent, "attribute": "mentioned", "value": "true"} for ent in entities[:10]]

        entity_hint = "、".join(entities) if entities else "(none)"
        prompt = (
            "从下面文本中抽取实体属性，输出 JSON 对象，格式为"
            " {\"attributes\":[{\"entity\":\"...\",\"attribute\":\"...\",\"value\":\"...\"}]}。"
            "不要输出解释。\n"
            f"候选实体: {entity_hint}\n"
            f"文本: {text}"
        )
        raw = self.llm_client.generate(prompt, system_prompt="你是信息抽取引擎。", purpose="entity_extract_attribute")
        try:
            parsed = json.loads(_extract_json_text(raw))
            attrs = parsed.get("attributes", []) if isinstance(parsed, dict) else []
            normalized: list[dict[str, str]] = []
            for item in attrs:
                if not isinstance(item, dict):
                    continue
                entity = str(item.get("entity", "")).strip()
                attribute = str(item.get("attribute", "")).strip()
                value = str(item.get("value", "")).strip()
                if entity and attribute and value:
                    normalized.append({"entity": entity, "attribute": attribute, "value": value})
            return normalized
        except Exception:
            return []

    # 成熟版实体抽取：支持 LLM、spaCy+LLM，输出三元组或属性两类结构
    def process(self, payload: RawConversationInput) -> ProcessorOutput:
        entities = self._extract_candidates(payload.message)
        is_triple_mode = self.method in {
            EntityExtractorMethod.llm_triple,
            EntityExtractorMethod.spacy_llm_triple,
        }

        if is_triple_mode:
            triples = self._generate_triples(payload.message, entities)
            if triples:
                lines = [f"({t['subject']})-[{t['predicate']}]->({t['object']})" for t in triples]
                content = "三元组:\n" + "\n".join(lines)
            else:
                content = "三元组: 无"
            metadata = {"entities": entities, "triples": triples, "method": self.method.value, **payload.metadata}
            tags = ["entity", "triple", "graph", f"method:{self.method.value}"]
        else:
            attributes = self._generate_attributes(payload.message, entities)
            if attributes:
                lines = [f"{a['entity']} | {a['attribute']} = {a['value']}" for a in attributes]
                content = "属性:\n" + "\n".join(lines)
            else:
                content = "属性: 无"
            metadata = {"entities": entities, "attributes": attributes, "method": self.method.value, **payload.metadata}
            tags = ["entity", "attribute", "relational", f"method:{self.method.value}"]

        chunk = MemoryChunk(
            chunk_id=_build_chunk_id(payload.session_id, "ent"),
            session_id=payload.session_id,
            content=content,
            tags=tags,
            metadata=metadata,
        )
        return ProcessorOutput(source=ProcessorType.entity_extractor, chunks=[chunk])
