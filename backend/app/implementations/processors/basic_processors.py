from __future__ import annotations

from app.core.interfaces import MemoryProcessor
from app.models.contracts import MemoryChunk, ProcessorOutput, ProcessorType, RawConversationInput


class RawLoggerProcessor(MemoryProcessor):
    # 全量记录：原始文本直接落入记忆块
    def process(self, payload: RawConversationInput) -> ProcessorOutput:
        chunk = MemoryChunk(
            chunk_id=f"{payload.session_id}-raw-1",
            session_id=payload.session_id,
            content=payload.message,
            tags=["raw", "full"],
            metadata=payload.metadata,
        )
        return ProcessorOutput(source=ProcessorType.raw_logger, chunks=[chunk])


class SummarizerProcessor(MemoryProcessor):
    # 滑动摘要（演示版）：使用前 120 字符作为摘要窗口
    def process(self, payload: RawConversationInput) -> ProcessorOutput:
        summary = payload.message[:120]
        chunk = MemoryChunk(
            chunk_id=f"{payload.session_id}-sum-1",
            session_id=payload.session_id,
            content=f"摘要: {summary}",
            tags=["summary"],
            metadata=payload.metadata,
        )
        return ProcessorOutput(source=ProcessorType.summarizer, chunks=[chunk])


class EntityExtractorProcessor(MemoryProcessor):
    # 实体提取（演示版）：以简单规则抓取首字母大写 token
    def process(self, payload: RawConversationInput) -> ProcessorOutput:
        entities = [token.strip(",.?!") for token in payload.message.split() if token[:1].isupper()]
        content = "实体: " + ", ".join(entities) if entities else "实体: 无"
        chunk = MemoryChunk(
            chunk_id=f"{payload.session_id}-ent-1",
            session_id=payload.session_id,
            content=content,
            tags=["entity"],
            metadata={"entities": entities, **payload.metadata},
        )
        return ProcessorOutput(source=ProcessorType.entity_extractor, chunks=[chunk])
