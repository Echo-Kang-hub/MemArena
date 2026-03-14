from __future__ import annotations

import pytest

from app.main import _merge_memory_hits, _validate_processor_engine_mapping
from app.models.contracts import (
    BenchmarkConfig,
    BenchmarkRunRequest,
    EngineType,
    EntityExtractorMethod,
    ProviderType,
    ProcessorType,
    ReflectorType,
    AssemblerType,
    MemoryHit,
)


def _build_request(method: EntityExtractorMethod, engine: EngineType) -> BenchmarkRunRequest:
    return BenchmarkRunRequest(
        config=BenchmarkConfig(
            processor=ProcessorType.entity_extractor,
            engine=engine,
            assembler=AssemblerType.reasoning_chain,
            reflector=ReflectorType.none,
            llm_provider=ProviderType.api,
            embedding_provider=ProviderType.api,
            entity_extractor_method=method,
        ),
        input_text="test",
    )


def test_merge_memory_hits_dedup_by_normalized_content_prefers_stm() -> None:
    stm_hit = MemoryHit(
        chunk_id="stm-1",
        content="  User likes sushi  ",
        relevance=0.51,
        metadata={"stm": True, "source": "stm"},
    )
    ltm_dup = MemoryHit(
        chunk_id="ltm-dup",
        content="user   likes   sushi",
        relevance=0.99,
        metadata={"stm": False, "source": "ltm"},
    )
    ltm_other = MemoryHit(
        chunk_id="ltm-2",
        content="User plans a business trip next week",
        relevance=0.88,
        metadata={"stm": False, "source": "ltm"},
    )

    merged = _merge_memory_hits(ltm_hits=[ltm_dup, ltm_other], stm_hits=[stm_hit], top_k=5)

    assert len(merged) == 2
    sushi_hits = [h for h in merged if "sushi" in h.content.lower()]
    assert len(sushi_hits) == 1
    assert sushi_hits[0].chunk_id == "stm-1"
    assert sushi_hits[0].metadata.get("stm") is True


def test_merge_memory_hits_top_k_has_minimum_one() -> None:
    stm_hit = MemoryHit(chunk_id="stm-1", content="a", relevance=0.1, metadata={"stm": True})
    ltm_hit = MemoryHit(chunk_id="ltm-1", content="b", relevance=0.2, metadata={"stm": False})

    merged = _merge_memory_hits(ltm_hits=[ltm_hit], stm_hits=[stm_hit], top_k=0)

    assert len(merged) == 1


def test_validate_processor_engine_mapping_mem0_requires_relational() -> None:
    payload = _build_request(EntityExtractorMethod.mem0_dual_facts, EngineType.graph_engine)

    with pytest.raises(ValueError, match="require engine=RelationalEngine"):
        _validate_processor_engine_mapping(payload)


def test_validate_processor_engine_mapping_mem0_relational_ok() -> None:
    payload = _build_request(EntityExtractorMethod.mem0_user_facts, EngineType.relational_engine)

    _validate_processor_engine_mapping(payload)
