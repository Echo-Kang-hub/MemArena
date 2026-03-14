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

ASSEMBLERS = [
    "SystemInjector",
    "XMLTagging",
    "TimelineRollover",
    "ReverseTimeline",
    "RankedPruning",
    "ReasoningChain",
]


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _safe_mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _extract_reasoning_stats(case_results: list[dict[str, Any]]) -> dict[str, float | None]:
    chain_counts: list[float] = []
    avg_priorities: list[float] = []
    avg_hops: list[float] = []
    seed_ratios: list[float] = []

    for case in case_results:
        hits = ((case.get("search_result") or {}).get("hits") or [])
        if not hits:
            continue
        metadata = (hits[0].get("metadata") or {})
        details = metadata.get("reasoning_chain_details")
        if not isinstance(details, list) or not details:
            chains = metadata.get("reasoning_chains")
            if isinstance(chains, list) and chains:
                chain_counts.append(float(len(chains)))
            continue

        chain_counts.append(float(len(details)))
        priorities = [float(d.get("priority")) for d in details if isinstance(d, dict) and d.get("priority") is not None]
        hops = [float(d.get("hop")) for d in details if isinstance(d, dict) and d.get("hop") is not None]
        seeds = [1.0 if bool(d.get("seed_touch", False)) else 0.0 for d in details if isinstance(d, dict)]

        if priorities:
            avg_priorities.append(sum(priorities) / len(priorities))
        if hops:
            avg_hops.append(sum(hops) / len(hops))
        if seeds:
            seed_ratios.append(sum(seeds) / len(seeds))

    return {
        "reasoning_chain_count": _safe_mean(chain_counts),
        "reasoning_avg_priority": _safe_mean(avg_priorities),
        "reasoning_avg_hop": _safe_mean(avg_hops),
        "reasoning_seed_touch_ratio": _safe_mean(seed_ratios),
    }


def _composite_score(row: dict[str, Any]) -> float:
    precision = float(row.get("precision") or 0.0)
    faithfulness = float(row.get("faithfulness") or 0.0)
    info_loss = float(row.get("info_loss") or 1.0)
    context_distraction = row.get("context_distraction")
    seed_ratio = row.get("reasoning_seed_touch_ratio")
    avg_priority = row.get("reasoning_avg_priority")

    context_quality = 0.5
    if context_distraction is not None:
        context_quality = 1.0 - float(context_distraction)

    seed_quality = 0.5 if seed_ratio is None else float(seed_ratio)
    # priority 通常在 [0,4] 左右，映射到 [0,1]
    priority_quality = 0.5 if avg_priority is None else _clamp01(float(avg_priority) / 4.0)

    return _clamp01(
        0.35 * precision
        + 0.25 * faithfulness
        + 0.15 * (1.0 - info_loss)
        + 0.10 * context_quality
        + 0.10 * seed_quality
        + 0.05 * priority_quality
    )


def _run_one(client: TestClient, assembler: str) -> dict[str, Any]:
    payload = {
        "dataset_name": "builtin_memory_smoke",
        "sample_size": 8,
        "start_index": 0,
        "max_concurrency": 1,
        "isolate_sessions": True,
        "user_id": "assembler-compare",
        "retrieval": {
            "top_k": 5,
            "min_relevance": 0.0,
            "collection_name": "memarena_assembler_compare",
            "similarity_strategy": "inverse_distance",
            "keyword_rerank": False,
            "max_context_tokens": 1200,
            "reasoning_hops": 2,
        },
        "config": {
            "processor": "EntityExtractor",
            "engine": "GraphEngine",
            "assembler": assembler,
            "reflector": "None",
            "llm_provider": "local",
            "chat_llm_provider": "local",
            "judge_llm_provider": "local",
            "summarizer_llm_provider": "local",
            "entity_llm_provider": "local",
            "embedding_provider": "local",
            "summarizer_method": "llm",
            "entity_extractor_method": "llm_triple",
            "compute_device": "cpu",
        },
    }

    resp = client.post("/api/benchmark/run-dataset", json=payload)
    resp.raise_for_status()
    data = resp.json()

    avg = data.get("avg_metrics") or {}
    case_results = data.get("case_results") or []
    reasoning = _extract_reasoning_stats(case_results)

    row = {
        "assembler": assembler,
        "cases": len(case_results),
        "precision": float(avg.get("precision") or 0.0),
        "faithfulness": float(avg.get("faithfulness") or 0.0),
        "info_loss": float(avg.get("info_loss") or 0.0),
        "recall_at_k": avg.get("recall_at_k"),
        "qa_accuracy": avg.get("qa_accuracy"),
        "qa_f1": avg.get("qa_f1"),
        "consistency_score": avg.get("consistency_score"),
        "rejection_rate": avg.get("rejection_rate"),
        "rejection_correctness_unknown": avg.get("rejection_correctness_unknown"),
        "convergence_speed": avg.get("convergence_speed"),
        "context_distraction": avg.get("context_distraction"),
        **reasoning,
    }
    row["composite_score"] = _composite_score(row)
    return row


def _write_reports(rows: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORT_DIR / f"assembler_compare_{stamp}.json"
    csv_path = REPORT_DIR / f"assembler_compare_{stamp}.csv"
    md_path = REPORT_DIR / f"assembler_compare_{stamp}.md"

    ranked = sorted(rows, key=lambda x: float(x.get("composite_score") or 0.0), reverse=True)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "ranked": ranked,
        "recommended": ranked[0]["assembler"] if ranked else None,
    }

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "assembler",
        "cases",
        "precision",
        "faithfulness",
        "info_loss",
        "recall_at_k",
        "qa_accuracy",
        "qa_f1",
        "consistency_score",
        "rejection_rate",
        "rejection_correctness_unknown",
        "convergence_speed",
        "context_distraction",
        "reasoning_chain_count",
        "reasoning_avg_priority",
        "reasoning_avg_hop",
        "reasoning_seed_touch_ratio",
        "composite_score",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked:
            writer.writerow(row)

    table = [
        "| assembler | precision | faithfulness | info_loss | context_distraction | reasoning_chain_count | reasoning_avg_priority | reasoning_avg_hop | seed_touch_ratio | composite_score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ranked:
        table.append(
            "| {assembler} | {precision:.4f} | {faithfulness:.4f} | {info_loss:.4f} | {context_distraction} | {reasoning_chain_count} | {reasoning_avg_priority} | {reasoning_avg_hop} | {reasoning_seed_touch_ratio} | {composite_score:.4f} |".format(
                assembler=r.get("assembler", ""),
                precision=float(r.get("precision") or 0.0),
                faithfulness=float(r.get("faithfulness") or 0.0),
                info_loss=float(r.get("info_loss") or 0.0),
                context_distraction=("N/A" if r.get("context_distraction") is None else f"{float(r['context_distraction']):.4f}"),
                reasoning_chain_count=("N/A" if r.get("reasoning_chain_count") is None else f"{float(r['reasoning_chain_count']):.2f}"),
                reasoning_avg_priority=("N/A" if r.get("reasoning_avg_priority") is None else f"{float(r['reasoning_avg_priority']):.4f}"),
                reasoning_avg_hop=("N/A" if r.get("reasoning_avg_hop") is None else f"{float(r['reasoning_avg_hop']):.2f}"),
                reasoning_seed_touch_ratio=("N/A" if r.get("reasoning_seed_touch_ratio") is None else f"{float(r['reasoning_seed_touch_ratio']):.4f}"),
                composite_score=float(r.get("composite_score") or 0.0),
            )
        )

    md_lines = [
        "# Assembler Comparison Report",
        "",
        f"- Recommended Assembler: **{summary.get('recommended') or 'N/A'}**",
        f"- Sample Size: {ranked[0].get('cases', 0) if ranked else 0}",
        "",
        "## Ranking",
        "",
        *table,
        "",
        "## Notes",
        "- Composite score favors precision/faithfulness and penalizes info_loss/context_distraction.",
        "- Reasoning quality signals are only available when reasoning_chain_details exists.",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return json_path, csv_path, md_path


def main() -> None:
    rows: list[dict[str, Any]] = []
    with TestClient(app) as client:
        for assembler in ASSEMBLERS:
            try:
                rows.append(_run_one(client, assembler))
            except Exception as exc:
                rows.append(
                    {
                        "assembler": assembler,
                        "cases": 0,
                        "precision": 0.0,
                        "faithfulness": 0.0,
                        "info_loss": 1.0,
                        "context_distraction": None,
                        "reasoning_chain_count": None,
                        "reasoning_avg_priority": None,
                        "reasoning_avg_hop": None,
                        "reasoning_seed_touch_ratio": None,
                        "composite_score": 0.0,
                        "error": str(exc),
                    }
                )

    json_path, csv_path, md_path = _write_reports(rows)
    print(json.dumps({"rows": rows}, ensure_ascii=False, indent=2))
    print(f"\nASSEMBLER_COMPARE_JSON={json_path}")
    print(f"ASSEMBLER_COMPARE_CSV={csv_path}")
    print(f"ASSEMBLER_COMPARE_MD={md_path}")


if __name__ == "__main__":
    main()
