from __future__ import annotations

from typing import Any

from app.core.interfaces import ContextAssembler, MemoryEngine, MemoryProcessor, MemoryReflector
from app.implementations.assemblers.basic_assemblers import (
    RankedPruningAssembler,
    ReasoningChainAssembler,
    ReverseTimelineAssembler,
    SystemInjectorAssembler,
    TimelineRolloverAssembler,
    XMLTaggingAssembler,
)
from app.implementations.engines.in_memory_engines import GraphEngine, RelationalEngine, VectorEngine
from app.implementations.processors.basic_processors import (
    EntityExtractorProcessor,
    RawLoggerProcessor,
    SummarizerProcessor,
)
from app.implementations.reflectors.basic_reflectors import (
    AbstractionReflector,
    ConflictResolverReflector,
    ConsolidatorReflector,
    DecayFilterReflector,
    GenerativeReflectionReflector,
    InsightLinkerReflector,
)
from app.models.contracts import AssemblerType, EngineType, ProcessorType, ReflectorLLMMode, ReflectorType
from app.models.contracts import EntityExtractorMethod, SummarizerMethod


def build_processor(
    kind: ProcessorType,
    summarizer_method: SummarizerMethod = SummarizerMethod.llm,
    entity_extractor_method: EntityExtractorMethod = EntityExtractorMethod.llm_triple,
    summarizer_llm_client: Any | None = None,
    entity_llm_client: Any | None = None,
) -> MemoryProcessor:
    mapping: dict[ProcessorType, MemoryProcessor] = {
        ProcessorType.raw_logger: RawLoggerProcessor(),
        ProcessorType.summarizer: SummarizerProcessor(method=summarizer_method, llm_client=summarizer_llm_client),
        ProcessorType.entity_extractor: EntityExtractorProcessor(method=entity_extractor_method, llm_client=entity_llm_client),
    }
    return mapping[kind]


def build_engine(kind: EngineType, embedding_client: Any | None = None, collection_name: str | None = None) -> MemoryEngine:
    if kind == EngineType.vector_engine:
        return VectorEngine(embedding_client=embedding_client, collection_name=collection_name)
    if kind == EngineType.graph_engine:
        return GraphEngine()
    return RelationalEngine()


def build_assembler(kind: AssemblerType) -> ContextAssembler:
    mapping: dict[AssemblerType, ContextAssembler] = {
        AssemblerType.system_injector: SystemInjectorAssembler(),
        AssemblerType.xml_tagging: XMLTaggingAssembler(),
        AssemblerType.timeline_rollover: TimelineRolloverAssembler(),
        AssemblerType.reverse_timeline: ReverseTimelineAssembler(),
        AssemblerType.ranked_pruning: RankedPruningAssembler(),
        AssemblerType.reasoning_chain: ReasoningChainAssembler(),
    }
    return mapping[kind]


def build_reflector(
    kind: ReflectorType,
    reflection_llm_client: Any | None = None,
    llm_mode: ReflectorLLMMode = ReflectorLLMMode.llm_with_fallback,
) -> MemoryReflector | None:
    if kind == ReflectorType.none:
        return None
    mapping: dict[ReflectorType, MemoryReflector] = {
        ReflectorType.generative_reflection: GenerativeReflectionReflector(llm_client=reflection_llm_client, llm_mode=llm_mode),
        ReflectorType.conflict_resolver: ConflictResolverReflector(llm_client=reflection_llm_client, llm_mode=llm_mode),
        ReflectorType.consolidator: ConsolidatorReflector(llm_client=reflection_llm_client, llm_mode=llm_mode),
        ReflectorType.decay_filter: DecayFilterReflector(),
        ReflectorType.insight_linker: InsightLinkerReflector(llm_client=reflection_llm_client, llm_mode=llm_mode),
        ReflectorType.abstraction_reflector: AbstractionReflector(llm_client=reflection_llm_client, llm_mode=llm_mode),
    }
    return mapping[kind]
