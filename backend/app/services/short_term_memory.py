from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.models.contracts import MemoryHit, ShortTermMemoryMode


@dataclass
class _SessionSTMState:
    turns: list[dict[str, str]] = field(default_factory=list)
    rolling_summary: str = ""
    blackboard: dict[str, str] = field(default_factory=dict)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    by_chars = int(len(text) * 0.65) + 1
    by_words = int(len(text.split()) * 1.3) + 1
    return max(by_chars, by_words, 1)


def _tokenize(text: str) -> set[str]:
    compact = (text or "").strip().lower()
    if not compact:
        return set()
    if " " in compact:
        return {part for part in re.split(r"\s+", compact) if part}
    return {ch for ch in compact if ch.strip()}


def _lexical_score(query: str, content: str) -> float:
    q = _tokenize(query)
    if not q:
        return 0.0
    c = _tokenize(content)
    return len(q.intersection(c)) / max(len(q), 1)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？；;])\s+|\n+", (text or "").strip())
    return [part.strip() for part in parts if part.strip()]


def _heuristic_summarize(text: str, max_sentences: int = 3) -> str:
    sents = _split_sentences(text)
    if not sents:
        return ""
    if len(sents) <= max_sentences:
        return " ".join(sents)

    scored: list[tuple[int, float]] = []
    all_tokens = [_tokenize(s) for s in sents]
    for idx, tokens in enumerate(all_tokens):
        if not tokens:
            scored.append((idx, 0.0))
            continue
        overlap = 0.0
        for jdx, other in enumerate(all_tokens):
            if jdx == idx or not other:
                continue
            overlap += len(tokens.intersection(other)) / max(len(tokens.union(other)), 1)
        scored.append((idx, overlap))

    keep_idx = sorted(sorted(scored, key=lambda x: x[1], reverse=True)[:max_sentences], key=lambda x: x[0])
    return " ".join(sents[i] for i, _ in keep_idx)


def _looks_like_invalid_llm_summary(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return True
    if lowered.startswith("llm provider call failed:"):
        return True
    if lowered.startswith("[local:"):
        return True
    if "for more information check:" in lowered and "status/401" in lowered:
        return True
    return False


class ShortTermMemoryManager:
    def __init__(self) -> None:
        self._states: dict[str, _SessionSTMState] = {}

    def _state(self, session_id: str) -> _SessionSTMState:
        state = self._states.get(session_id)
        if state is None:
            state = _SessionSTMState()
            self._states[session_id] = state
        return state

    def _update_blackboard(self, state: _SessionSTMState, text: str) -> None:
        lowered = (text or "").lower()
        if any(k in lowered for k in ["任务完成", "完成了", "先不聊", "换个话题", "结束这个任务"]):
            state.blackboard.clear()
            return

        task_map = {
            "订机票": "订机票",
            "机票": "订机票",
            "会议": "安排会议",
            "行程": "规划行程",
            "项目": "推进项目",
        }
        for key, value in task_map.items():
            if key in text:
                state.blackboard["当前任务"] = value
                break

        city_match = re.search(r"去([\u4e00-\u9fff]{2,10})", text)
        if city_match:
            state.blackboard["目的地"] = city_match.group(1)

        from_match = re.search(r"从([\u4e00-\u9fff]{2,10})出发", text)
        if from_match:
            state.blackboard["出发地"] = from_match.group(1)

        if "护照" in text:
            state.blackboard["证件"] = "护照"
        if "预算" in text:
            state.blackboard["预算约束"] = "已提及预算"

    def _compress_rolling_summary(
        self,
        old_summary: str,
        overflow_turns: list[dict[str, str]],
        llm_client: Any | None,
    ) -> str:
        overflow_text = "\n".join(
            f"- [{str(x.get('role', 'unknown'))}] {str(x.get('text', '')).strip()}"
            for x in overflow_turns
            if str(x.get("text", "")).strip()
        )
        if not overflow_text.strip():
            return old_summary

        if llm_client is not None:
            prompt = (
                "你是短期记忆压缩助手。请把现有摘要与旧对话融合为新的简洁摘要，保留任务目标、约束、重要实体、用户偏好。"
                "输出 4-8 条短句，中文。\n\n"
                f"现有摘要:\n{old_summary or '(无)'}\n\n"
                f"被滚出的旧对话:\n{overflow_text}"
            )
            try:
                merged = llm_client.generate(prompt, system_prompt="你是记忆压缩助手。", purpose="stm_summary_update")
                merged = (merged or "").strip()
                if merged and not _looks_like_invalid_llm_summary(merged):
                    return merged
            except Exception:
                pass

        combined = (old_summary + "\n" + overflow_text).strip()
        return _heuristic_summarize(combined, max_sentences=4)

    def ingest(
        self,
        session_id: str,
        text: str,
        mode: ShortTermMemoryMode,
        summary_keep_recent_turns: int,
        role: str = "user",
        llm_client: Any | None = None,
    ) -> None:
        if mode == ShortTermMemoryMode.none:
            return

        state = self._state(session_id)
        role_norm = str(role or "unknown").strip().lower()
        if role_norm not in {"user", "assistant"}:
            role_norm = "unknown"
        state.turns.append({"role": role_norm, "text": str(text or "")})
        if len(state.turns) > 120:
            state.turns = state.turns[-120:]

        if mode == ShortTermMemoryMode.working_memory_blackboard and role_norm == "user":
            self._update_blackboard(state, text)
            return

        if mode == ShortTermMemoryMode.rolling_summary:
            keep = max(1, int(summary_keep_recent_turns))
            if len(state.turns) > keep:
                overflow = state.turns[:-keep]
                state.turns = state.turns[-keep:]
                state.rolling_summary = self._compress_rolling_summary(state.rolling_summary, overflow, llm_client)

    def _select_by_token_budget(self, turns: list[dict[str, str]], token_budget: int) -> list[dict[str, str]]:
        selected: list[dict[str, str]] = []
        used = 0
        for turn in reversed(turns):
            text = str(turn.get("text", ""))
            cost = _estimate_tokens(text)
            if selected and used + cost > token_budget:
                break
            selected.append(turn)
            used += cost
        selected.reverse()
        return selected

    def _build_turn_hits(self, query: str, turns: list[dict[str, str]], source: str, top_k: int) -> list[MemoryHit]:
        if not turns:
            return []

        weighted: list[tuple[float, int, str, str]] = []
        total = len(turns)
        for idx, turn in enumerate(turns):
            text = str(turn.get("text", ""))
            role = str(turn.get("role", "unknown")).strip().lower() or "unknown"
            recency = (idx + 1) / total
            lexical = _lexical_score(query, text)
            score = min(1.0, 0.6 * recency + 0.4 * lexical)
            weighted.append((score, idx, role, text))

        weighted.sort(key=lambda x: x[0], reverse=True)
        picked = weighted[: max(1, top_k)]
        return [
            MemoryHit(
                chunk_id=f"stm-{source}-{i}",
                content=item[3],
                relevance=float(item[0]),
                metadata={"stm_source": source, "stm_index": item[1], "stm": True, "role": item[2]},
            )
            for i, item in enumerate(picked, start=1)
        ]

    def retrieve(
        self,
        session_id: str,
        query: str,
        mode: ShortTermMemoryMode,
        top_k: int,
        window_turns: int,
        token_budget: int,
        summary_keep_recent_turns: int,
    ) -> list[MemoryHit]:
        if mode == ShortTermMemoryMode.none:
            return []

        state = self._state(session_id)
        if mode == ShortTermMemoryMode.sliding_window:
            turns = state.turns[-max(1, int(window_turns)) :]
            return self._build_turn_hits(query, turns, "sliding_window", top_k)

        if mode == ShortTermMemoryMode.token_buffer:
            turns = self._select_by_token_budget(state.turns, max(128, int(token_budget)))
            return self._build_turn_hits(query, turns, "token_buffer", top_k)

        if mode == ShortTermMemoryMode.rolling_summary:
            keep = max(1, int(summary_keep_recent_turns))
            turns = state.turns[-keep:]
            hits = self._build_turn_hits(query, turns, "rolling_recent", max(1, top_k - 1))
            if state.rolling_summary:
                summary_score = min(1.0, 0.45 + 0.55 * _lexical_score(query, state.rolling_summary))
                hits.append(
                    MemoryHit(
                        chunk_id="stm-rolling-summary",
                        content=f"历史摘要: {state.rolling_summary}",
                        relevance=summary_score,
                        metadata={"stm_source": "rolling_summary", "stm": True},
                    )
                )
            hits.sort(key=lambda x: x.relevance, reverse=True)
            return hits[: max(1, top_k)]

        if mode == ShortTermMemoryMode.working_memory_blackboard:
            if not state.blackboard:
                return self._build_turn_hits(query, state.turns[-2:], "blackboard_fallback", top_k)
            board_lines = [f"{k}: {v}" for k, v in state.blackboard.items()]
            content = "工作记忆黑板:\n" + "\n".join(board_lines)
            score = min(1.0, 0.55 + 0.45 * _lexical_score(query, content))
            return [
                MemoryHit(
                    chunk_id="stm-blackboard",
                    content=content,
                    relevance=score,
                    metadata={"stm_source": "working_memory_blackboard", "stm": True, "blackboard": state.blackboard},
                )
            ]

        return []
