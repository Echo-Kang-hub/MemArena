from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import sys

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app  # noqa: E402


REPORT_DIR = ROOT / "backend" / "data" / "reports"


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _run_mode(client: TestClient, method: str, engine: str) -> dict[str, Any]:
    payload = {
        "dataset_name": "builtin_memory_smoke",
        "sample_size": 5,
        "start_index": 0,
        "max_concurrency": 1,
        "isolate_sessions": True,
        "user_id": "entity-live-eval",
        "retrieval": {
            "top_k": 5,
            "min_relevance": 0.0,
            "collection_name": "memarena_entity_live_eval",
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
            "entity_llm_provider": "api",
            "embedding_provider": "local",
            "summarizer_method": "llm",
            "entity_extractor_method": method,
            "compute_device": "cpu",
        },
    }

    resp = client.post("/api/benchmark/run-dataset", json=payload)
    resp.raise_for_status()
    data = resp.json()

    cases = data.get("case_results", [])
    total = len(cases)

    non_empty = 0
    struct_items: list[float] = []
    top1_scores: list[float] = []

    for case in cases:
        hits = case.get("search_result", {}).get("hits", [])
        if not hits:
            continue

        first = hits[0]
        top1_scores.append(float(first.get("relevance", 0.0)))

        meta = first.get("metadata", {}) or {}
        triples = meta.get("triples", []) if isinstance(meta, dict) else []
        attrs = meta.get("attributes", []) if isinstance(meta, dict) else []

        if method.endswith("triple"):
            valid = [t for t in triples if isinstance(t, dict) and t.get("subject") and t.get("predicate") and t.get("object")]
            struct_items.append(float(len(valid)))
            if valid:
                non_empty += 1
        else:
            valid = [a for a in attrs if isinstance(a, dict) and a.get("entity") and a.get("attribute") and a.get("value")]
            struct_items.append(float(len(valid)))
            if valid:
                non_empty += 1

    return {
        "method": method,
        "engine": engine,
        "cases": total,
        "non_empty_struct_ratio": (non_empty / total) if total else 0.0,
        "avg_struct_items": _safe_mean(struct_items),
        "avg_top1_relevance": _safe_mean(top1_scores),
        "avg_recall_at_k": float((data.get("avg_metrics") or {}).get("recall_at_k") or 0.0),
        "avg_precision": float((data.get("avg_metrics") or {}).get("precision") or 0.0),
    }


def _write_reports(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORT_DIR / f"entity_live_eval_{stamp}.json"
    csv_path = REPORT_DIR / f"entity_live_eval_{stamp}.csv"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "method",
        "engine",
        "cases",
        "non_empty_struct_ratio",
        "avg_struct_items",
        "avg_top1_relevance",
        "avg_recall_at_k",
        "avg_precision",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return json_path, csv_path


def main() -> None:
    configs = [
        ("llm_triple", "GraphEngine"),
        ("spacy_llm_triple", "GraphEngine"),
        ("llm_attribute", "RelationalEngine"),
        ("spacy_llm_attribute", "RelationalEngine"),
    ]

    rows: list[dict[str, Any]] = []
    with TestClient(app) as client:
        for method, engine in configs:
            try:
                rows.append(_run_mode(client, method, engine))
            except Exception as exc:
                rows.append(
                    {
                        "method": method,
                        "engine": engine,
                        "cases": 0,
                        "non_empty_struct_ratio": 0.0,
                        "avg_struct_items": 0.0,
                        "avg_top1_relevance": 0.0,
                        "avg_recall_at_k": 0.0,
                        "avg_precision": 0.0,
                        "error": str(exc),
                    }
                )

    json_path, csv_path = _write_reports(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\nENTITY_LIVE_JSON={json_path}")
    print(f"ENTITY_LIVE_CSV={csv_path}")


if __name__ == "__main__":
    main()
