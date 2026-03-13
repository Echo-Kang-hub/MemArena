from __future__ import annotations

from typing import Any

from app.core.interfaces import ContextAssembler, MemoryEngine, MemoryProcessor, MemoryReflector
from app.implementations.assemblers.basic_assemblers import (
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
from app.implementations.reflectors.basic_reflectors import ConflictResolverReflector, GenerativeReflectionReflector
from app.models.contracts import AssemblerType, EngineType, ProcessorType, ReflectorType


def build_processor(kind: ProcessorType) -> MemoryProcessor:
    mapping: dict[ProcessorType, MemoryProcessor] = {
        ProcessorType.raw_logger: RawLoggerProcessor(),
        ProcessorType.summarizer: SummarizerProcessor(),
        ProcessorType.entity_extractor: EntityExtractorProcessor(),
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
    }
    return mapping[kind]


def build_reflector(kind: ReflectorType) -> MemoryReflector | None:
    if kind == ReflectorType.none:
        return None
    mapping: dict[ReflectorType, MemoryReflector] = {
        ReflectorType.generative_reflection: GenerativeReflectionReflector(),
        ReflectorType.conflict_resolver: ConflictResolverReflector(),
    }
    return mapping[kind]
