from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any

from app.core.interfaces import MemoryReflector
from app.models.contracts import ReflectRequest, ReflectResult, ReflectorLLMMode, ReflectorType


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


def _extract_json_text(raw: str) -> str:
    stripped = (raw or "").strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    obj_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if obj_match:
        return obj_match.group(0).strip()
    return stripped


def _collect_attribute_groups(request: ReflectRequest) -> dict[str, dict[str, set[str]]]:
    grouped: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for hit in request.memory_hits:
        attrs = (hit.metadata or {}).get("attributes", [])
        if not isinstance(attrs, list):
            continue
        for item in attrs:
            if not isinstance(item, dict):
                continue
            ent = str(item.get("entity", "")).strip().lower()
            attr = str(item.get("attribute", "")).strip().lower()
            val = str(item.get("value", "")).strip()
            if ent and attr and val:
                grouped[ent][attr].add(val)
    return grouped


def _find_conflict_items(request: ReflectRequest) -> list[dict[str, Any]]:
    grouped = _collect_attribute_groups(request)
    conflicts: list[dict[str, Any]] = []
    for ent, attrs in grouped.items():
        for attr, values in attrs.items():
            if len(values) > 1:
                conflicts.append({"entity": ent, "attribute": attr, "values": sorted(values)})
    return conflicts


def _is_multi_valued_predicate(predicate: str) -> bool:
    lowered = (predicate or "").strip().lower()
    multi = {
        "related_to",
        "associate_with",
        "associated_with",
        "includes",
        "contains",
        "member_of",
        "标签",
        "包含",
        "包括",
        "关联",
    }
    return lowered in multi


def _find_triple_conflicts(request: ReflectRequest) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for hit in request.memory_hits:
        triples = (hit.metadata or {}).get("triples", [])
        if not isinstance(triples, list):
            continue
        for item in triples:
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject", "")).strip()
            predicate = str(item.get("predicate", "")).strip()
            obj = str(item.get("object", "")).strip()
            if not (subject and predicate and obj):
                continue
            grouped[(subject.lower(), predicate.lower())].add(obj)

    conflicts: list[dict[str, Any]] = []
    for (subject, predicate), objects in grouped.items():
        if len(objects) <= 1:
            continue
        if _is_multi_valued_predicate(predicate):
            continue
        conflicts.append({
            "subject": subject,
            "predicate": predicate,
            "objects": sorted(objects),
        })
    return conflicts


def _pick_value_from_query(options: list[str], query: str) -> tuple[str | None, float]:
    q = (query or "").lower()
    best: str | None = None
    for item in options:
        token = (item or "").strip().lower()
        if token and token in q:
            best = item
            break
    if best is not None:
        return best, 0.9
    if not options:
        return None, 0.0
    return options[-1], 0.58


def _build_proposed_resolutions(request: ReflectRequest) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for item in _find_conflict_items(request):
        chosen, confidence = _pick_value_from_query(item.get("values", []), request.latest_query)
        if not chosen:
            continue
        proposals.append(
            {
                "kind": "attribute",
                "entity": item["entity"],
                "attribute": item["attribute"],
                "resolved_value": chosen,
                "candidate_values": item.get("values", []),
                "confidence": confidence,
            }
        )

    for item in _find_triple_conflicts(request):
        chosen, confidence = _pick_value_from_query(item.get("objects", []), request.latest_query)
        if not chosen:
            continue
        proposals.append(
            {
                "kind": "triple",
                "subject": item["subject"],
                "predicate": item["predicate"],
                "resolved_object": chosen,
                "candidate_objects": item.get("objects", []),
                "confidence": confidence,
            }
        )
    return proposals


def _estimate_redundant_pairs(hits: list) -> int:
    docs = [hit.content for hit in hits]
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
            if jac >= 0.65:
                similar_pairs += 1
    return similar_pairs


def _extract_themes(request: ReflectRequest) -> list[str]:
    text = "\n".join(hit.content for hit in request.memory_hits)
    candidates = re.findall(r"[\u4e00-\u9fff]{2,8}", text)
    freq: dict[str, int] = defaultdict(int)
    for item in candidates:
        if len(item) < 2:
            continue
        freq[item] += 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranked[:6]]


def _partition_hits_by_role(request: ReflectRequest) -> tuple[list[str], list[str], list[str]]:
    user_hits: list[str] = []
    assistant_hits: list[str] = []
    other_hits: list[str] = []
    for hit in request.memory_hits:
        role = str((hit.metadata or {}).get("role", "")).strip().lower()
        if role == "user":
            user_hits.append(hit.content)
        elif role == "assistant":
            assistant_hits.append(hit.content)
        else:
            other_hits.append(hit.content)
    return user_hits, assistant_hits, other_hits


def _should_use_llm(mode: ReflectorLLMMode) -> bool:
    return mode in {ReflectorLLMMode.llm, ReflectorLLMMode.llm_with_fallback}


def _llm_json_or_none(llm_client: Any | None, prompt: str, purpose: str) -> dict[str, Any] | None:
    if llm_client is None:
        return None
    try:
        raw = llm_client.generate(prompt, system_prompt="你是严谨的记忆反思分析师。", purpose=purpose)
        parsed = json.loads(_extract_json_text(raw))
        if isinstance(parsed, dict):
            return parsed
        return None
    except Exception:
        return None


class GenerativeReflectionReflector(MemoryReflector):
    def __init__(
        self,
        llm_client: Any | None = None,
        llm_mode: ReflectorLLMMode = ReflectorLLMMode.llm_with_fallback,
    ) -> None:
        self.llm_client = llm_client
        self.llm_mode = llm_mode

    def _heuristic_reflect(self, request: ReflectRequest) -> tuple[list[str], dict[str, Any]]:
        themes = _extract_themes(request)
        correction = _is_correction_query(request.latest_query)
        conflict_items = _find_conflict_items(request)
        redundant_pairs = _estimate_redundant_pairs(request.memory_hits)

        insights: list[str] = []
        if themes:
            insights.append(f"高频主题: {', '.join(themes[:4])}，建议按主题建立记忆分区并提高主题内去重阈值。")
        else:
            insights.append("当前主题分布稀疏，建议先提升实体抽取覆盖率后再做主题聚合。")

        if conflict_items:
            first = conflict_items[0]
            insights.append(
                "发现事实冲突苗头："
                f"{first['entity']} / {first['attribute']} 出现多值 {first['values']}，"
                "建议触发冲突消解并保留时序版本。"
            )
        else:
            insights.append("未见明显属性冲突，可将资源优先投入冗余压缩与索引质量优化。")

        if redundant_pairs > 0:
            insights.append(f"检测到 {redundant_pairs} 对高相似记忆，建议执行 Consolidator 合并并保留来源追踪。")

        if correction:
            insights.append("最新查询包含纠错信号，建议提高近期记忆权重并将旧事实降级为历史版本。")

        return insights[:5], {
            "source": "heuristic",
            "theme_count": len(themes),
            "conflict_count": len(conflict_items),
            "redundant_pairs": redundant_pairs,
            "correction_detected": correction,
        }

    async def reflect(self, request: ReflectRequest) -> ReflectResult:
        if not _should_use_llm(self.llm_mode) or self.llm_client is None:
            insights, stats = self._heuristic_reflect(request)
            stats["llm_mode"] = self.llm_mode.value
            return ReflectResult(reflector=ReflectorType.generative_reflection, insights=insights, stats=stats)

        memory_lines = "\n".join([f"- {hit.content}" for hit in request.memory_hits[:12]])
        prompt = (
            "你是记忆系统反思器。请基于最新问题与命中记忆，输出高阶洞察和可执行建议。"
            "必须返回 JSON："
            '{"insights":["..."],"actions":["..."],"risks":["..."]}。\n\n'
            f"latest_query:\n{request.latest_query}\n\n"
            f"memory_hits:\n{memory_lines or '- (none)'}"
        )

        try:
            raw = self.llm_client.generate(prompt, system_prompt="你是严谨的记忆反思分析师。", purpose="reflector_generative")
            parsed = json.loads(_extract_json_text(raw))
            insights = []
            for key in ("insights", "actions", "risks"):
                values = parsed.get(key, []) if isinstance(parsed, dict) else []
                if isinstance(values, list):
                    insights.extend([str(x).strip() for x in values if str(x).strip()])

            if not insights:
                insights, stats = self._heuristic_reflect(request)
                stats["source"] = "llm_fallback_heuristic"
                return ReflectResult(reflector=ReflectorType.generative_reflection, insights=insights, stats=stats)

            return ReflectResult(
                reflector=ReflectorType.generative_reflection,
                insights=insights[:8],
                stats={"source": "llm", "hit_count": len(request.memory_hits)},
            )
        except Exception:
            insights, stats = self._heuristic_reflect(request)
            stats["source"] = "llm_parse_fallback"
            return ReflectResult(reflector=ReflectorType.generative_reflection, insights=insights, stats=stats)


class ConflictResolverReflector(MemoryReflector):
    def __init__(
        self,
        llm_client: Any | None = None,
        llm_mode: ReflectorLLMMode = ReflectorLLMMode.llm_with_fallback,
    ) -> None:
        self.llm_client = llm_client
        self.llm_mode = llm_mode

    async def reflect(self, request: ReflectRequest) -> ReflectResult:
        conflict_items = _find_conflict_items(request)
        triple_conflicts = _find_triple_conflicts(request)
        proposed_resolutions = _build_proposed_resolutions(request)
        conflict_count = len(conflict_items) + len(triple_conflicts)
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
            samples = "; ".join(
                [f"{x['entity']}/{x['attribute']}={x['values']}" for x in conflict_items[:3]]
            )
            triple_samples = "; ".join(
                [f"{x['subject']}/{x['predicate']}={x['objects']}" for x in triple_conflicts[:2]]
            )
            insights = [
                f"检测到 {conflict_count} 组潜在冲突（属性+三元组），属性样例: {samples or 'N/A'}",
                (f"三元组冲突样例: {triple_samples}" if triple_samples else "当前未发现明显三元组冲突。"),
                "建议按时间戳、用户纠正信号、来源置信度做冲突裁决并写回版本状态。",
                "被用户明确否定的值应降级为 obsolete，同时保留审计轨迹避免静默覆盖。",
            ]

        if _should_use_llm(self.llm_mode) and self.llm_client is not None and conflict_count > 0:
            prompt = (
                "你是冲突消解反思器。基于冲突数据生成可执行洞察。输出 JSON: "
                '{"insights":["..."],"proposed_resolutions":[{"kind":"attribute|triple","confidence":0.0}]}\n\n'
                f"latest_query: {request.latest_query}\n"
                f"attribute_conflicts: {json.dumps(conflict_items[:10], ensure_ascii=False)}\n"
                f"triple_conflicts: {json.dumps(triple_conflicts[:10], ensure_ascii=False)}\n"
                f"heuristic_proposals: {json.dumps(proposed_resolutions[:20], ensure_ascii=False)}"
            )
            parsed = _llm_json_or_none(self.llm_client, prompt, "reflector_conflict_resolver")
            if isinstance(parsed, dict):
                llm_insights = parsed.get("insights", [])
                if isinstance(llm_insights, list):
                    cleaned = [str(x).strip() for x in llm_insights if str(x).strip()]
                    if cleaned:
                        insights = cleaned[:8]

                llm_resolutions = parsed.get("proposed_resolutions", [])
                if isinstance(llm_resolutions, list):
                    normalized_updates: list[dict[str, Any]] = []
                    for item in llm_resolutions:
                        if isinstance(item, dict):
                            normalized_updates.append(item)
                    if normalized_updates:
                        proposed_resolutions = normalized_updates[:20]

        return ReflectResult(
            reflector=ReflectorType.conflict_resolver,
            insights=insights,
            stats={
                "llm_mode": self.llm_mode.value,
                "conflict_count": conflict_count,
                "conflict_items": conflict_items[:10],
                "triple_conflicts": triple_conflicts[:10],
                "proposed_resolutions": proposed_resolutions[:20],
                "correction_detected": correction,
                "convergence_cycles": convergence_cycles,
            },
        )


class ConsolidatorReflector(MemoryReflector):
    def __init__(
        self,
        llm_client: Any | None = None,
        llm_mode: ReflectorLLMMode = ReflectorLLMMode.llm_with_fallback,
    ) -> None:
        self.llm_client = llm_client
        self.llm_mode = llm_mode

    async def reflect(self, request: ReflectRequest) -> ReflectResult:
        similar_pairs = _estimate_redundant_pairs(request.memory_hits)
        timeline_hint = "present_vs_past" if any(k in request.latest_query for k in ["曾经", "以前", "现在", "目前"]) else "none"
        user_hits, assistant_hits, other_hits = _partition_hits_by_role(request)
        memory_decision: dict[str, Any] = {
            "source": "heuristic",
            "keep": [*user_hits[:3], *assistant_hits[:2]],
            "update": [],
            "drop": [],
            "rationale": "优先保留用户事实与高相关助手结论；其余在冲突或重复时再进入裁决。",
        }
        insights = [
            f"Consolidator 检测到 {similar_pairs} 对高相似记忆，可执行融合式更新以减少冗余。",
            "建议策略：做近重复内容合并、摘要压缩与来源聚合，并输出候选 Memory Decision。",
        ]

        if _should_use_llm(self.llm_mode) and self.llm_client is not None:
            prompt = (
                "你是去重合并反思器。请输出 JSON: {\"insights\":[\"...\"],\"merge_rules\":[\"...\"],"
                "\"memory_decision\":{\"keep\":[\"...\"],\"update\":[\"...\"],\"drop\":[\"...\"],\"rationale\":\"...\"}}。\n\n"
                f"latest_query: {request.latest_query}\n"
                f"similar_pairs: {similar_pairs}\n"
                f"user_hits: {json.dumps(user_hits[:8], ensure_ascii=False)}\n"
                f"assistant_hits: {json.dumps(assistant_hits[:8], ensure_ascii=False)}\n"
                f"other_hits: {json.dumps(other_hits[:6], ensure_ascii=False)}"
            )
            parsed = _llm_json_or_none(self.llm_client, prompt, "reflector_consolidator")
            if isinstance(parsed, dict):
                llm_insights = parsed.get("insights", [])
                if isinstance(llm_insights, list):
                    cleaned = [str(x).strip() for x in llm_insights if str(x).strip()]
                    if cleaned:
                        insights = cleaned[:6]
                llm_memory_decision = parsed.get("memory_decision", {})
                if isinstance(llm_memory_decision, dict):
                    memory_decision = {
                        "source": "llm",
                        "keep": [str(x).strip() for x in llm_memory_decision.get("keep", []) if str(x).strip()][:10],
                        "update": [str(x).strip() for x in llm_memory_decision.get("update", []) if str(x).strip()][:10],
                        "drop": [str(x).strip() for x in llm_memory_decision.get("drop", []) if str(x).strip()][:10],
                        "rationale": str(llm_memory_decision.get("rationale", "")).strip(),
                    }

        return ReflectResult(
            reflector=ReflectorType.consolidator,
            insights=insights,
            stats={
                "llm_mode": self.llm_mode.value,
                "similar_pairs": similar_pairs,
                "timeline_hint": timeline_hint,
                "role_boundary": "dedup_only",
                "role_hit_counts": {
                    "user": len(user_hits),
                    "assistant": len(assistant_hits),
                    "other": len(other_hits),
                },
                "memory_decision": memory_decision,
                "targets": ["RelationalEngine", "GraphEngine"],
            },
        )


class ConflictConsolidatorReflector(MemoryReflector):
    def __init__(
        self,
        llm_client: Any | None = None,
        llm_mode: ReflectorLLMMode = ReflectorLLMMode.llm_with_fallback,
    ) -> None:
        self._consolidator = ConsolidatorReflector(llm_client=llm_client, llm_mode=llm_mode)
        self._resolver = ConflictResolverReflector(llm_client=llm_client, llm_mode=llm_mode)
        self.llm_mode = llm_mode

    async def reflect(self, request: ReflectRequest) -> ReflectResult:
        consolidator_result = await self._consolidator.reflect(request)
        resolver_result = await self._resolver.reflect(request)

        insights: list[str] = []
        seen: set[str] = set()
        for line in [*consolidator_result.insights, *resolver_result.insights]:
            normalized = str(line).strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            insights.append(normalized)

        stats = {
            "llm_mode": self.llm_mode.value,
            "composition": ["Consolidator", "ConflictResolver"],
            "memory_decision": (consolidator_result.stats or {}).get("memory_decision", {}),
            "proposed_resolutions": (resolver_result.stats or {}).get("proposed_resolutions", []),
            "conflict_count": int((resolver_result.stats or {}).get("conflict_count", 0) or 0),
            "similar_pairs": int((consolidator_result.stats or {}).get("similar_pairs", 0) or 0),
            "convergence_cycles": int((resolver_result.stats or {}).get("convergence_cycles", 0) or 0),
            "targets": ["VectorEngine", "RelationalEngine", "GraphEngine"],
        }

        return ReflectResult(
            reflector=ReflectorType.conflict_consolidator,
            insights=insights[:10],
            stats=stats,
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
    def __init__(
        self,
        llm_client: Any | None = None,
        llm_mode: ReflectorLLMMode = ReflectorLLMMode.llm_with_fallback,
    ) -> None:
        self.llm_client = llm_client
        self.llm_mode = llm_mode

    async def reflect(self, request: ReflectRequest) -> ReflectResult:
        entities = _extract_entities("\n".join(hit.content for hit in request.memory_hits))
        predicted_links: list[str] = []
        if len(entities) >= 3:
            for i in range(min(len(entities) - 2, 3)):
                predicted_links.append(f"{entities[i]} -> {entities[i + 2]}")

        if _should_use_llm(self.llm_mode) and self.llm_client is not None and entities:
            prompt = (
                "你是知识连接反思器。请从给定实体中推断潜在连接，输出 JSON: {\"predicted_links\":[\"A -> B\"],\"insights\":[\"...\"]}。\n\n"
                f"latest_query: {request.latest_query}\n"
                f"entities: {json.dumps(entities[:20], ensure_ascii=False)}"
            )
            parsed = _llm_json_or_none(self.llm_client, prompt, "reflector_insight_linker")
            if isinstance(parsed, dict):
                links = parsed.get("predicted_links", [])
                if isinstance(links, list):
                    normalized = [str(x).strip() for x in links if str(x).strip()]
                    if normalized:
                        predicted_links = normalized[:8]

        insights = [
            "InsightLinker 已完成异步潜在连接推理（Link Prediction），可用于减少知识孤岛。",
            (f"建议补充连接: {', '.join(predicted_links)}" if predicted_links else "当前实体图稀疏，建议先提升实体抽取密度。"),
        ]
        return ReflectResult(
            reflector=ReflectorType.insight_linker,
            insights=insights,
            stats={"llm_mode": self.llm_mode.value, "predicted_links": predicted_links, "targets": ["GraphEngine"]},
        )


class AbstractionReflector(MemoryReflector):
    def __init__(
        self,
        llm_client: Any | None = None,
        llm_mode: ReflectorLLMMode = ReflectorLLMMode.llm_with_fallback,
    ) -> None:
        self.llm_client = llm_client
        self.llm_mode = llm_mode

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

        if _should_use_llm(self.llm_mode) and self.llm_client is not None and text.strip():
            prompt = (
                "你是用户画像抽象反思器。请提炼偏好与行为倾向，输出 JSON: {\"abstractions\":[\"...\"],\"insights\":[\"...\"]}。\n\n"
                f"latest_query: {request.latest_query}\n"
                f"memory_text: {text[:4000]}"
            )
            parsed = _llm_json_or_none(self.llm_client, prompt, "reflector_abstraction")
            if isinstance(parsed, dict):
                abstractions = parsed.get("abstractions", [])
                if isinstance(abstractions, list):
                    normalized = [str(x).strip() for x in abstractions if str(x).strip()]
                    if normalized:
                        prefs = normalized[:8]

        insights = [
            "AbstractionReflector 已将行为序列抽象为偏好/性格线索，用于提升“懂我”能力。",
            ("抽象结果: " + " | ".join(prefs)) if prefs else "当前可抽象线索较少，建议累积更多行为序列。",
        ]
        return ReflectResult(
            reflector=ReflectorType.abstraction_reflector,
            insights=insights,
            stats={"llm_mode": self.llm_mode.value, "abstractions": prefs, "targets": ["RelationalEngine"]},
        )
