from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import _merge_memory_hits, _validate_processor_engine_mapping
from app.models.contracts import (
    AssembleRequest,
    BenchmarkConfig,
    BenchmarkRunRequest,
    EngineType,
    EntityExtractorMethod,
    ProviderType,
    ProcessorType,
    ReflectorType,
    AssemblerType,
    MemoryHit,
    RawConversationInput,
)
from app.implementations.processors.basic_processors import RawLoggerProcessor
from app.implementations.assemblers.basic_assemblers import SystemInjectorAssembler
from app.main import app


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


def test_raw_logger_splits_user_and_assistant_chunks() -> None:
    processor = RawLoggerProcessor()
    payload = RawConversationInput(
        session_id="s1",
        user_id="u1",
        message="user says hello",
        metadata={"assistant_message": "assistant replies hi"},
    )

    output = processor.process(payload)
    assert len(output.chunks) == 2
    roles = sorted(str(c.metadata.get("role", "")) for c in output.chunks)
    assert roles == ["assistant", "user"]


def test_system_injector_partitions_memory_by_role() -> None:
    assembler = SystemInjectorAssembler()
    req = AssembleRequest(
        user_query="what's next?",
        memory_hits=[
            MemoryHit(chunk_id="1", content="u-fact", relevance=0.9, metadata={"role": "user"}),
            MemoryHit(chunk_id="2", content="a-fact", relevance=0.8, metadata={"role": "assistant"}),
        ],
    )

    out = assembler.assemble(req)
    assert "[MEMORY_USER]" in out.prompt
    assert "[MEMORY_ASSISTANT]" in out.prompt
    assert "u-fact" in out.prompt
    assert "a-fact" in out.prompt


def test_api_prompt_contains_memory_user_and_assistant_partitions() -> None:
    client = TestClient(app)
    payload = {
        "config": {
            "processor": "RawLogger",
            "engine": "RelationalEngine",
            "assembler": "SystemInjector",
            "reflector": "None",
            "llm_provider": "local",
            "chat_llm_provider": "local",
            "judge_llm_provider": "local",
            "summarizer_llm_provider": "local",
            "entity_llm_provider": "local",
            "embedding_provider": "local",
            "summarizer_method": "llm",
            "entity_extractor_method": "llm_triple",
            "compute_device": "cpu",
        },
        "session_id": "api-e2e-role-sections",
        "user_id": "tester",
        "input_text": "请记住我喜欢无糖拿铁",
        "assistant_message": "好的，我记住了。",
        "expected_facts": ["无糖拿铁"],
        "retrieval": {
            "top_k": 5,
            "min_relevance": 0.0,
            "collection_name": "memarena_memory",
            "similarity_strategy": "inverse_distance",
            "keyword_rerank": False,
            "short_term_mode": "None",
        },
    }

    resp = client.post("/api/benchmark/run", json=payload)
    assert resp.status_code == 200, resp.text
    prompt = resp.json()["assemble_result"]["prompt"]
    assert "[MEMORY_USER]" in prompt
    assert "[MEMORY_ASSISTANT]" in prompt


def test_api_conflict_consolidator_reflector_with_independent_provider() -> None:
    client = TestClient(app)
    payload = {
        "config": {
            "processor": "EntityExtractor",
            "engine": "RelationalEngine",
            "assembler": "SystemInjector",
            "reflector": "ConflictConsolidator",
            "llm_provider": "api",
            "chat_llm_provider": "local",
            "judge_llm_provider": "local",
            "summarizer_llm_provider": "local",
            "entity_llm_provider": "local",
            "reflector_llm_provider": "local",
            "embedding_provider": "local",
            "summarizer_method": "llm",
            "entity_extractor_method": "mem0_dual_facts",
            "compute_device": "cpu",
        },
        "session_id": "api-e2e-combined-reflector",
        "user_id": "tester",
        "input_text": "我以前在北京工作，现在在上海。更正：常驻城市是杭州。",
        "assistant_message": "已记录更正信息。",
        "expected_facts": ["杭州"],
        "retrieval": {
            "top_k": 5,
            "min_relevance": 0.0,
            "collection_name": "memarena_memory",
            "similarity_strategy": "inverse_distance",
            "keyword_rerank": False,
            "short_term_mode": "None",
            "reflector_llm_mode": "LLMWithFallback",
        },
    }

    resp = client.post("/api/benchmark/run", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    reflector = body["reflector_result"]
    assert reflector is not None
    assert reflector["reflector"] == "ConflictConsolidator"
    stats = reflector.get("stats", {})
    assert stats.get("composition") == ["Consolidator", "ConflictResolver"]
    assert "memory_decision" in stats
    assert "proposed_resolutions" in stats
