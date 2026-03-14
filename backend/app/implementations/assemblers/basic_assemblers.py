from __future__ import annotations

import math

from app.config import settings
from app.core.interfaces import ContextAssembler
from app.models.contracts import AssembleRequest, AssembleResult, AssemblerType, MemoryHit


def _is_pinned(hit: MemoryHit) -> bool:
    meta = hit.metadata or {}
    if bool(meta.get("pinned", False)):
        return True
    tags = meta.get("tags", [])
    if isinstance(tags, list):
        lowered = {str(tag).strip().lower() for tag in tags}
        if "pinned" in lowered or "core" in lowered:
            return True
    return False


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # 粗略 token 估计：中文按字符，英文按词，取较保守上界。
    by_chars = math.ceil(len(text) * 0.65)
    by_words = math.ceil(len(text.split()) * 1.3)
    return max(by_chars, by_words, 1)


def _partition_hits_by_role(hits: list[MemoryHit]) -> tuple[list[MemoryHit], list[MemoryHit], list[MemoryHit]]:
    user_hits: list[MemoryHit] = []
    assistant_hits: list[MemoryHit] = []
    other_hits: list[MemoryHit] = []
    for hit in hits:
        role = str((hit.metadata or {}).get("role", "")).strip().lower()
        if role == "user":
            user_hits.append(hit)
        elif role == "assistant":
            assistant_hits.append(hit)
        else:
            other_hits.append(hit)
    return user_hits, assistant_hits, other_hits


def _build_role_memory_block(hits: list[MemoryHit]) -> str:
    user_hits, assistant_hits, other_hits = _partition_hits_by_role(hits)
    user_block = "\n".join([f"- {hit.content}" for hit in user_hits]) if user_hits else "- (none)"
    assistant_block = "\n".join([f"- {hit.content}" for hit in assistant_hits]) if assistant_hits else "- (none)"
    other_block = "\n".join([f"- {hit.content}" for hit in other_hits]) if other_hits else "- (none)"
    return (
        "[MEMORY_USER]\n"
        f"{user_block}\n\n"
        "[MEMORY_ASSISTANT]\n"
        f"{assistant_block}\n\n"
        "[MEMORY_OTHER]\n"
        f"{other_block}"
    )


class SystemInjectorAssembler(ContextAssembler):
    def assemble(self, request: AssembleRequest) -> AssembleResult:
        ordered_hits = sorted(request.memory_hits, key=lambda x: (not _is_pinned(x), -x.relevance))
        memory_block = _build_role_memory_block(ordered_hits)
        prompt = (
            f"[SYSTEM]\n{request.system_prompt}\n\n"
            f"{memory_block}\n\n"
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
        user_hits, assistant_hits, other_hits = _partition_hits_by_role(request.memory_hits)
        memories_user = "\n".join([f"    <item score=\"{hit.relevance:.2f}\">{hit.content}</item>" for hit in user_hits]) or "    <item>(none)</item>"
        memories_assistant = "\n".join([f"    <item score=\"{hit.relevance:.2f}\">{hit.content}</item>" for hit in assistant_hits]) or "    <item>(none)</item>"
        memories_other = "\n".join([f"    <item score=\"{hit.relevance:.2f}\">{hit.content}</item>" for hit in other_hits]) or "    <item>(none)</item>"
        memories = (
            "  <user_memory>\n"
            f"{memories_user}\n"
            "  </user_memory>\n"
            "  <assistant_memory>\n"
            f"{memories_assistant}\n"
            "  </assistant_memory>\n"
            "  <other_memory>\n"
            f"{memories_other}\n"
            "  </other_memory>"
        )
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
        user_hits, assistant_hits, other_hits = _partition_hits_by_role(request.memory_hits)
        timeline = "\n".join(
            [f"U{i + 1}: {hit.content} (score={hit.relevance:.2f})" for i, hit in enumerate(user_hits)]
            + [f"A{i + 1}: {hit.content} (score={hit.relevance:.2f})" for i, hit in enumerate(assistant_hits)]
            + [f"O{i + 1}: {hit.content} (score={hit.relevance:.2f})" for i, hit in enumerate(other_hits)]
        ) or "(none)"
        prompt = f"时间线记忆:\n{timeline}\n\n当前问题:\n{request.user_query}"
        return AssembleResult(
            assembler=AssemblerType.timeline_rollover,
            prompt=prompt,
            preview_blocks=[{"role": "timeline", "text": timeline}, {"role": "user", "text": request.user_query}],
        )


class ReverseTimelineAssembler(ContextAssembler):
    def assemble(self, request: AssembleRequest) -> AssembleResult:
        reverse_hits = list(reversed(request.memory_hits))
        user_hits, assistant_hits, other_hits = _partition_hits_by_role(reverse_hits)
        timeline = "\n".join(
            [f"UR{i + 1}: {hit.content} (score={hit.relevance:.2f})" for i, hit in enumerate(user_hits)]
            + [f"AR{i + 1}: {hit.content} (score={hit.relevance:.2f})" for i, hit in enumerate(assistant_hits)]
            + [f"OR{i + 1}: {hit.content} (score={hit.relevance:.2f})" for i, hit in enumerate(other_hits)]
        ) or "(none)"
        prompt = f"倒序时间线记忆(最新优先):\n{timeline}\n\n当前问题:\n{request.user_query}"
        return AssembleResult(
            assembler=AssemblerType.reverse_timeline,
            prompt=prompt,
            preview_blocks=[{"role": "reverse-timeline", "text": timeline}, {"role": "user", "text": request.user_query}],
        )


class RankedPruningAssembler(ContextAssembler):
    def assemble(self, request: AssembleRequest) -> AssembleResult:
        budget = request.token_budget or settings.context_token_budget
        ranked = sorted(request.memory_hits, key=lambda x: x.relevance, reverse=True)

        selected: list[MemoryHit] = []
        used = 0
        for hit in ranked:
            line = f"- [{hit.relevance:.2f}] {hit.content}"
            line_cost = _estimate_tokens(line)
            if selected and used + line_cost > budget:
                continue
            selected.append(hit)
            used += line_cost

        if not selected and ranked:
            selected = [ranked[0]]
            used = _estimate_tokens(f"- [{ranked[0].relevance:.2f}] {ranked[0].content}")

        pruned = max(0, len(ranked) - len(selected))
        user_hits, assistant_hits, other_hits = _partition_hits_by_role(selected)
        memory_block = (
            "[MEMORY_USER]\n"
            + ("\n".join([f"- [{hit.relevance:.2f}] {hit.content}" for hit in user_hits]) or "- (none)")
            + "\n\n[MEMORY_ASSISTANT]\n"
            + ("\n".join([f"- [{hit.relevance:.2f}] {hit.content}" for hit in assistant_hits]) or "- (none)")
            + "\n\n[MEMORY_OTHER]\n"
            + ("\n".join([f"- [{hit.relevance:.2f}] {hit.content}" for hit in other_hits]) or "- (none)")
        )
        prompt = (
            f"[SYSTEM]\n{request.system_prompt}\n\n"
            f"[MEMORY-RANKED budget={budget} used~{used} pruned={pruned}]\n{memory_block}\n\n"
            f"[USER]\n{request.user_query}"
        )
        return AssembleResult(
            assembler=AssemblerType.ranked_pruning,
            prompt=prompt,
            preview_blocks=[
                {"role": "budget", "text": f"budget={budget}, used~{used}, pruned={pruned}"},
                {"role": "memory-ranked", "text": memory_block},
                {"role": "user", "text": request.user_query},
            ],
        )


class ReasoningChainAssembler(ContextAssembler):
    def assemble(self, request: AssembleRequest) -> AssembleResult:
        chains: list[str] = []
        for hit in request.memory_hits:
            chain_items = (hit.metadata or {}).get("reasoning_chains", [])
            if not isinstance(chain_items, list):
                continue
            for c in chain_items:
                line = str(c).strip()
                if line and line not in chains:
                    chains.append(line)

        chain_block = "\n".join([f"- {c}" for c in chains]) if chains else "- (no explicit chain found)"
        memory_block = _build_role_memory_block(request.memory_hits)
        prompt = (
            f"[SYSTEM]\n{request.system_prompt}\n\n"
            f"[REASONING_CHAINS]\n{chain_block}\n\n"
            f"[REFERENCE_MEMORY]\n{memory_block}\n\n"
            f"[USER]\n{request.user_query}"
        )
        return AssembleResult(
            assembler=AssemblerType.reasoning_chain,
            prompt=prompt,
            preview_blocks=[
                {"role": "reasoning-chains", "text": chain_block},
                {"role": "reference-memory", "text": memory_block},
                {"role": "user", "text": request.user_query},
            ],
        )
