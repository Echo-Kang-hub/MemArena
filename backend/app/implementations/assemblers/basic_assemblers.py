from __future__ import annotations

from app.core.interfaces import ContextAssembler
from app.models.contracts import AssembleRequest, AssembleResult, AssemblerType


class SystemInjectorAssembler(ContextAssembler):
    def assemble(self, request: AssembleRequest) -> AssembleResult:
        memory_block = "\n".join([f"- {hit.content}" for hit in request.memory_hits])
        prompt = (
            f"[SYSTEM]\n{request.system_prompt}\n\n"
            f"[MEMORY]\n{memory_block}\n\n"
            f"[USER]\n{request.user_query}"
        )
        return AssembleResult(
            assembler=AssemblerType.system_injector,
            prompt=prompt,
            preview_blocks=[
                {"role": "system", "text": request.system_prompt},
                {"role": "memory", "text": memory_block},
                {"role": "user", "text": request.user_query},
            ],
        )


class XMLTaggingAssembler(ContextAssembler):
    def assemble(self, request: AssembleRequest) -> AssembleResult:
        memories = "\n".join([f"  <item score=\"{hit.relevance:.2f}\">{hit.content}</item>" for hit in request.memory_hits])
        prompt = (
            "<system>你是一个可靠的 AI 助手，必须优先参考 memory。</system>\n"
            f"<memory>\n{memories}\n</memory>\n"
            f"<user>{request.user_query}</user>"
        )
        return AssembleResult(
            assembler=AssemblerType.xml_tagging,
            prompt=prompt,
            preview_blocks=[
                {"role": "xml-system", "text": "你是一个可靠的 AI 助手，必须优先参考 memory。"},
                {"role": "xml-memory", "text": memories},
                {"role": "xml-user", "text": request.user_query},
            ],
        )


class TimelineRolloverAssembler(ContextAssembler):
    def assemble(self, request: AssembleRequest) -> AssembleResult:
        timeline = "\n".join(
            [f"T{i + 1}: {hit.content} (score={hit.relevance:.2f})" for i, hit in enumerate(request.memory_hits)]
        )
        prompt = f"时间线记忆:\n{timeline}\n\n当前问题:\n{request.user_query}"
        return AssembleResult(
            assembler=AssemblerType.timeline_rollover,
            prompt=prompt,
            preview_blocks=[{"role": "timeline", "text": timeline}, {"role": "user", "text": request.user_query}],
        )
