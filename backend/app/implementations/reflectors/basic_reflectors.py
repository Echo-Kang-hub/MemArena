from __future__ import annotations

from app.core.interfaces import MemoryReflector
from app.models.contracts import ReflectRequest, ReflectResult, ReflectorType


class GenerativeReflectionReflector(MemoryReflector):
    async def reflect(self, request: ReflectRequest) -> ReflectResult:
        insights = [
            "用户近期查询主题较稳定，可尝试提高摘要压缩比。",
            f"最近问题与 {len(request.memory_hits)} 条记忆发生关联。",
        ]
        return ReflectResult(reflector=ReflectorType.generative_reflection, insights=insights)


class ConflictResolverReflector(MemoryReflector):
    async def reflect(self, request: ReflectRequest) -> ReflectResult:
        insights = ["未发现明显冲突条目。", "建议后续引入时间戳与来源置信度做冲突消解。"]
        return ReflectResult(reflector=ReflectorType.conflict_resolver, insights=insights)
