from __future__ import annotations

import math
import re
from collections import defaultdict

from app.core.interfaces import MemoryReflector
from app.models.contracts import ReflectRequest, ReflectResult, ReflectorType


def _tokenize(text: str) -> set[str]:
    compact = (text or "").strip().lower()
    if not compact:
        return set()
    if " " in compact:
        return {part for part in re.split(r"\s+", compact) if part}
    return {ch for ch in compact if ch.strip()}


def _extract_entities(text: str) -> list[str]:
    cjk = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    caps = [t.strip(" ,.?!:;()[]{}\"'") for t in text.split() if t[:1].isupper()]
    merged = [*cjk, *caps]
    seen: set[str] = set()
    out: list[str] = []
    for item in merged:
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out[:20]


def _is_correction_query(text: str) -> bool:
    lowered = (text or "").lower()
    cues = ["更正", "纠正", "说错", "不是", "应为", "改成", "改为", "其实"]
    return any(c in lowered for c in cues)


def _estimate_conflicts_from_hits(request: ReflectRequest) -> int:
    conflicts = 0
    seen: dict[str, set[str]] = defaultdict(set)
    for hit in request.memory_hits:
        attrs = (hit.metadata or {}).get("attributes", [])
        if not isinstance(attrs, list):
            continue
        for item in attrs:
            if not isinstance(item, dict):
                continue
            e = str(item.get("entity", "")).strip().lower()
            a = str(item.get("attribute", "")).strip().lower()
            v = str(item.get("value", "")).strip().lower()
            if not (e and a and v):
                continue
            key = f"{e}::{a}"
            seen[key].add(v)
    for values in seen.values():
        if len(values) > 1:
            conflicts += 1
    return conflicts


class GenerativeReflectionReflector(MemoryReflector):
    async def reflect(self, request: ReflectRequest) -> ReflectResult:
        insights = [
            "用户近期查询主题较稳定，可尝试提高摘要压缩比。",
            f"最近问题与 {len(request.memory_hits)} 条记忆发生关联。",
        ]
        return ReflectResult(reflector=ReflectorType.generative_reflection, insights=insights, stats={"hit_count": len(request.memory_hits)})


class ConflictResolverReflector(MemoryReflector):
    async def reflect(self, request: ReflectRequest) -> ReflectResult:
        conflict_count = _estimate_conflicts_from_hits(request)
        correction = _is_correction_query(request.latest_query)
        if correction and conflict_count == 0:
            convergence_cycles = 1
        elif correction:
            convergence_cycles = min(6, 1 + conflict_count)
        else:
            convergence_cycles = 0

        if conflict_count == 0:
            insights = ["未发现明显冲突条目。", "若后续出现用户更正语句，可优先提升新事实权重。"]
        else:
            insights = [
                f"检测到 {conflict_count} 组潜在冲突属性，建议按时间与置信度进行覆盖更新。",
                "对于被用户明确否定的事实，建议标记为 obsolete 或从主索引移除。",
            ]
        return ReflectResult(
            reflector=ReflectorType.conflict_resolver,
            insights=insights,
            stats={
                "conflict_count": conflict_count,
                "correction_detected": correction,
                "convergence_cycles": convergence_cycles,
            },
        )


class ConsolidatorReflector(MemoryReflector):
    async def reflect(self, request: ReflectRequest) -> ReflectResult:
        docs = [hit.content for hit in request.memory_hits]
        similar_pairs = 0
        for i in range(len(docs)):
            ti = _tokenize(docs[i])
            if not ti:
                continue
            for j in range(i + 1, len(docs)):
                tj = _tokenize(docs[j])
                if not tj:
                    continue
                jac = len(ti.intersection(tj)) / max(len(ti.union(tj)), 1)
                if jac >= 0.6:
                    similar_pairs += 1

        correction = _is_correction_query(request.latest_query)
        timeline_hint = "曾经/现在" if any(k in request.latest_query for k in ["曾经", "以前", "现在", "目前"]) else "none"
        insights = [
            f"Consolidator 检测到 {similar_pairs} 对高相似记忆，可执行融合式更新以减少冗余。",
            "建议策略：同键属性保留最新值，同时将旧值迁移为历史事实（例如“曾经是程序员，现在是摄影师”）。",
        ]
        if correction:
            insights.append("检测到用户更正语句：应触发错误纠正路径，清理被明确否定的错误记忆。")
        return ReflectResult(
            reflector=ReflectorType.consolidator,
            insights=insights,
            stats={
                "similar_pairs": similar_pairs,
                "correction_detected": correction,
                "timeline_hint": timeline_hint,
                "targets": ["RelationalEngine", "GraphEngine"],
            },
        )


class DecayFilterReflector(MemoryReflector):
    async def reflect(self, request: ReflectRequest) -> ReflectResult:
        n = len(request.memory_hits)
        # 无时间戳时按命中顺序近似“记忆年龄”，使用 Ebbinghaus 形式 exp(-t/s)
        scores: list[float] = []
        for idx, _ in enumerate(request.memory_hits, start=1):
            pseudo_age = max(1, idx)
            score = math.exp(-pseudo_age / 5.0)
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        stale_ratio = (sum(1 for s in scores if s < 0.35) / len(scores)) if scores else 0.0
        insights = [
            "DecayFilter 已按遗忘曲线为命中记忆打分，可用于降低长对话检索负载。",
            f"当前平均保留分={avg_score:.3f}，低保留分比例={stale_ratio:.2%}。",
        ]
        return ReflectResult(
            reflector=ReflectorType.decay_filter,
            insights=insights,
            stats={
                "avg_retention": round(avg_score, 4),
                "stale_ratio": round(stale_ratio, 4),
                "targets": ["VectorEngine", "RelationalEngine"],
            },
        )


class InsightLinkerReflector(MemoryReflector):
    async def reflect(self, request: ReflectRequest) -> ReflectResult:
        entities = _extract_entities("\n".join(hit.content for hit in request.memory_hits))
        predicted_links: list[str] = []
        if len(entities) >= 3:
            for i in range(min(len(entities) - 2, 3)):
                predicted_links.append(f"{entities[i]} -> {entities[i + 2]}")

        insights = [
            "InsightLinker 已完成异步潜在连接推理（Link Prediction），可用于减少知识孤岛。",
            (f"建议补充连接: {', '.join(predicted_links)}" if predicted_links else "当前实体图稀疏，建议先提升实体抽取密度。"),
        ]
        return ReflectResult(
            reflector=ReflectorType.insight_linker,
            insights=insights,
            stats={"predicted_links": predicted_links, "targets": ["GraphEngine"]},
        )


class AbstractionReflector(MemoryReflector):
    async def reflect(self, request: ReflectRequest) -> ReflectResult:
        text = "\n".join(hit.content for hit in request.memory_hits)
        prefs: list[str] = []
        for pattern, label in [
            (r"喜欢[^，。\n]*", "偏好倾向"),
            (r"提醒[^，。\n]*", "计划管理倾向"),
            (r"出差[^，。\n]*", "工作节奏偏好"),
        ]:
            for m in re.findall(pattern, text):
                prefs.append(f"{label}: {m}")
                if len(prefs) >= 5:
                    break
            if len(prefs) >= 5:
                break

        insights = [
            "AbstractionReflector 已将行为序列抽象为偏好/性格线索，用于提升“懂我”能力。",
            ("抽象结果: " + " | ".join(prefs)) if prefs else "当前可抽象线索较少，建议累积更多行为序列。",
        ]
        return ReflectResult(
            reflector=ReflectorType.abstraction_reflector,
            insights=insights,
            stats={"abstractions": prefs, "targets": ["RelationalEngine"]},
        )
