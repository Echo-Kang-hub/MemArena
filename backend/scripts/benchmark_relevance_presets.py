from __future__ import annotations

import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.implementations.engines.in_memory_engines import GraphEngine, RelationalEngine  # noqa: E402
from app.main import app  # noqa: E402
from app.models.contracts import EngineSaveRequest, EngineSearchRequest, MemoryChunk, ProcessorType  # noqa: E402


REPORT_DIR = ROOT / "backend" / "data" / "reports"


def _apply_graph_weights(weights: dict[str, float]) -> None:
    settings.graph_relevance_lexical_weight = weights["lexical"]
    settings.graph_relevance_entity_weight = weights["entity"]
    settings.graph_relevance_completeness_weight = weights["completeness"]
    settings.graph_relevance_hint_weight = weights["hint"]
    settings.graph_relevance_fallback_lexical_weight = weights["fallback_lexical"]
    settings.graph_relevance_fallback_hint_weight = weights["fallback_hint"]


def _apply_relational_weights(weights: dict[str, float]) -> None:
    settings.relational_relevance_lexical_weight = weights["lexical"]
    settings.relational_relevance_entity_weight = weights["entity"]
    settings.relational_relevance_completeness_weight = weights["completeness"]
    settings.relational_relevance_hint_weight = weights["hint"]
    settings.relational_relevance_fallback_lexical_weight = weights["fallback_lexical"]
    settings.relational_relevance_fallback_hint_weight = weights["fallback_hint"]


def _safe_mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _probe_structured_scoring(preset_name: str) -> list[dict[str, Any]]:
    graph_engine = GraphEngine()
    relational_engine = RelationalEngine()

    graph_session = f"probe-graph-{preset_name}"
    relational_session = f"probe-rel-{preset_name}"

    graph_engine.save(
        EngineSaveRequest(
            source=ProcessorType.entity_extractor,
            chunks=[
                MemoryChunk(
                    chunk_id="g-related",
                    session_id=graph_session,
                    content="三元组: (王磊)-[参加]->(复盘)",
                    metadata={
                        "triples": [
                            {"subject": "王磊", "predicate": "参加", "object": "复盘"},
                            {"subject": "周五", "predicate": "关联", "object": "预算表"},
                        ]
                    },
                    score_hint=0.9,
                ),
                MemoryChunk(
                    chunk_id="g-noisy",
                    session_id=graph_session,
                    content="三元组: (深圳)-[位于]->(华南)",
                    metadata={
                        "triples": [
                            {"subject": "深圳", "predicate": "位于", "object": "华南"},
                        ]
                    },
                    score_hint=1.0,
                ),
                MemoryChunk(
                    chunk_id="g-incomplete",
                    session_id=graph_session,
                    content="三元组: (王磊)-[参加]->(?)",
                    metadata={
                        "triples": [
                            {"subject": "王磊", "predicate": "参加", "object": ""},
                        ]
                    },
                    score_hint=1.0,
                ),
            ],
        )
    )

    relational_engine.save(
        EngineSaveRequest(
            source=ProcessorType.entity_extractor,
            chunks=[
                MemoryChunk(
                    chunk_id="r-related",
                    session_id=relational_session,
                    content="属性: 王磊 | 任务 = 复盘",
                    metadata={
                        "attributes": [
                            {"entity": "王磊", "attribute": "任务", "value": "复盘"},
                            {"entity": "周五", "attribute": "安排", "value": "提交预算表"},
                        ]
                    },
                    score_hint=0.9,
                ),
                MemoryChunk(
                    chunk_id="r-noisy",
                    session_id=relational_session,
                    content="属性: 深圳 | 地区 = 华南",
                    metadata={
                        "attributes": [
                            {"entity": "深圳", "attribute": "地区", "value": "华南"},
                        ]
                    },
                    score_hint=1.0,
                ),
                MemoryChunk(
                    chunk_id="r-incomplete",
                    session_id=relational_session,
                    content="属性: 王磊 | 任务 = ",
                    metadata={
                        "attributes": [
                            {"entity": "王磊", "attribute": "任务", "value": ""},
                        ]
                    },
                    score_hint=1.0,
                ),
            ],
        )
    )

    graph_query = "周五 王磊 复盘"
    rel_query = "周五 王磊 复盘"

    g_hits = graph_engine.search(EngineSearchRequest(session_id=graph_session, query=graph_query, top_k=3)).hits
    r_hits = relational_engine.search(EngineSearchRequest(session_id=relational_session, query=rel_query, top_k=3)).hits

    def summarize_probe(mode: str, hits: list[Any]) -> dict[str, Any]:
        top = hits[0] if hits else None
        return {
            "preset": preset_name,
            "mode": mode,
            "probe_top1_chunk": top.chunk_id if top else "",
            "probe_top1_relevance": float(top.relevance) if top else 0.0,
            "probe_rank_order": [h.chunk_id for h in hits],
        }

    return [summarize_probe("triple_probe", g_hits), summarize_probe("attribute_probe", r_hits)]


def _run_once(client: TestClient, *, preset_name: str, mode: str, engine: str, method: str, graph: dict[str, float], relational: dict[str, float]) -> dict[str, Any]:
    _apply_graph_weights(graph)
    _apply_relational_weights(relational)

    payload = {
        "dataset_name": "builtin_memory_smoke",
        "sample_size": 5,
        "start_index": 0,
        "max_concurrency": 1,
        "isolate_sessions": True,
        "user_id": "preset-eval",
        "retrieval": {
            "top_k": 5,
            "min_relevance": 0.0,
            "collection_name": "memarena_relevance_eval",
            "similarity_strategy": "inverse_distance",
            "keyword_rerank": False,
        },
        "config": {
            "processor": "EntityExtractor",
            "engine": engine,
            "assembler": "SystemInjector",
            "reflector": "None",
            "llm_provider": "local",
            "chat_llm_provider": "local",
            "judge_llm_provider": "local",
            "summarizer_llm_provider": "local",
            "entity_llm_provider": "local",
            "embedding_provider": "local",
            "summarizer_method": "llm",
            "entity_extractor_method": method,
            "compute_device": "cpu",
        },
    }

    resp = client.post("/api/benchmark/run-dataset", json=payload)
    resp.raise_for_status()
    data = resp.json()

    top1_scores: list[float] = []
    top3_means: list[float] = []
    nonzero_top1 = 0

    for case in data.get("case_results", []):
        hits = case.get("search_result", {}).get("hits", [])
        if not hits:
            continue
        top_scores = [float(hit.get("relevance", 0.0)) for hit in hits]
        top1 = top_scores[0]
        top1_scores.append(top1)
        top3_means.append(_safe_mean(top_scores[:3]))
        if top1 > 0:
            nonzero_top1 += 1

    total_cases = len(data.get("case_results", []))
    avg_metrics = data.get("avg_metrics", {})
    return {
        "preset": preset_name,
        "mode": mode,
        "dataset": payload["dataset_name"],
        "cases": total_cases,
        "avg_recall_at_k": float(avg_metrics.get("recall_at_k") or 0.0),
        "avg_precision": float(avg_metrics.get("precision") or 0.0),
        "avg_top1_relevance": _safe_mean(top1_scores),
        "avg_top3_relevance": _safe_mean(top3_means),
        "top1_nonzero_ratio": (nonzero_top1 / total_cases) if total_cases else 0.0,
    }


def _write_reports(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORT_DIR / f"relevance_presets_{stamp}.json"
    csv_path = REPORT_DIR / f"relevance_presets_{stamp}.csv"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = dict(row)
            if isinstance(normalized.get("probe_rank_order"), list):
                normalized["probe_rank_order"] = " > ".join(str(v) for v in normalized["probe_rank_order"])
            writer.writerow(normalized)

    return json_path, csv_path


def main() -> None:
    presets = {
        "balanced": {
            "graph": {
                "lexical": 0.45,
                "entity": 0.35,
                "completeness": 0.20,
                "hint": 0.10,
                "fallback_lexical": 0.90,
                "fallback_hint": 0.10,
            },
            "relational": {
                "lexical": 0.45,
                "entity": 0.35,
                "completeness": 0.20,
                "hint": 0.10,
                "fallback_lexical": 0.90,
                "fallback_hint": 0.10,
            },
        },
        "precision": {
            "graph": {
                "lexical": 0.20,
                "entity": 0.50,
                "completeness": 0.30,
                "hint": 0.05,
                "fallback_lexical": 0.70,
                "fallback_hint": 0.30,
            },
            "relational": {
                "lexical": 0.20,
                "entity": 0.50,
                "completeness": 0.30,
                "hint": 0.05,
                "fallback_lexical": 0.70,
                "fallback_hint": 0.30,
            },
        },
        "recall": {
            "graph": {
                "lexical": 0.60,
                "entity": 0.25,
                "completeness": 0.15,
                "hint": 0.10,
                "fallback_lexical": 0.95,
                "fallback_hint": 0.05,
            },
            "relational": {
                "lexical": 0.60,
                "entity": 0.25,
                "completeness": 0.15,
                "hint": 0.10,
                "fallback_lexical": 0.95,
                "fallback_hint": 0.05,
            },
        },
    }

    modes = [
        ("triple", "GraphEngine", "llm_triple"),
        ("attribute", "RelationalEngine", "llm_attribute"),
    ]

    rows: list[dict[str, Any]] = []
    with TestClient(app) as client:
        for preset_name, cfg in presets.items():
            _apply_graph_weights(cfg["graph"])
            _apply_relational_weights(cfg["relational"])
            rows.extend(_probe_structured_scoring(preset_name))
            for mode, engine, method in modes:
                rows.append(
                    _run_once(
                        client,
                        preset_name=preset_name,
                        mode=mode,
                        engine=engine,
                        method=method,
                        graph=cfg["graph"],
                        relational=cfg["relational"],
                    )
                )

    json_path, csv_path = _write_reports(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\\nREPORT_JSON={json_path}")
    print(f"REPORT_CSV={csv_path}")


if __name__ == "__main__":
    main()
