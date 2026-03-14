from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "backend" / "data" / "reports"

PRESET_WEIGHTS: dict[str, dict[str, dict[str, float]]] = {
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


def _latest_csv() -> Path:
    candidates = sorted(REPORT_DIR.glob("relevance_presets_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No report csv found in: {REPORT_DIR}")
    return candidates[-1]


def _to_float(v: str | None) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _append(d: dict[str, list[float]], key: str, value: float | None) -> None:
    if value is not None:
        d.setdefault(key, []).append(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize(csv_path: Path) -> dict[str, Any]:
    per_preset: dict[str, dict[str, list[float]]] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            preset = (row.get("preset") or "").strip()
            mode = (row.get("mode") or "").strip()
            if not preset:
                continue
            bucket = per_preset.setdefault(preset, {})

            if mode.endswith("_probe"):
                _append(bucket, "probe_top1", _to_float(row.get("probe_top1_relevance")))
            else:
                _append(bucket, "dataset_top1", _to_float(row.get("avg_top1_relevance")))
                _append(bucket, "dataset_top3", _to_float(row.get("avg_top3_relevance")))
                _append(bucket, "dataset_nonzero", _to_float(row.get("top1_nonzero_ratio")))
                _append(bucket, "dataset_recall", _to_float(row.get("avg_recall_at_k")))

    scored: list[dict[str, Any]] = []
    for preset, v in per_preset.items():
        probe_top1 = _mean(v.get("probe_top1", []))
        dataset_top1 = _mean(v.get("dataset_top1", []))
        dataset_top3 = _mean(v.get("dataset_top3", []))
        dataset_nonzero = _mean(v.get("dataset_nonzero", []))
        dataset_recall = _mean(v.get("dataset_recall", []))

        # 结构化可解释信号为主，数据集信号为辅（local 占位模型下仍可保留趋势信息）
        composite = 0.7 * probe_top1 + 0.2 * dataset_top1 + 0.1 * dataset_top3

        scored.append(
            {
                "preset": preset,
                "probe_top1_mean": probe_top1,
                "dataset_top1_mean": dataset_top1,
                "dataset_top3_mean": dataset_top3,
                "dataset_nonzero_mean": dataset_nonzero,
                "dataset_recall_mean": dataset_recall,
                "composite_score": composite,
            }
        )

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    best = scored[0] if scored else None
    return {
        "source_csv": str(csv_path),
        "ranked_presets": scored,
        "recommended": best,
        "notes": [
            "dataset recall/precision may stay low when local placeholder LLM cannot produce structured JSON",
            "probe metrics are the primary signal for ranking under local mode",
        ],
    }


def _env_snippet(preset_name: str) -> str:
    cfg = PRESET_WEIGHTS[preset_name]
    g = cfg["graph"]
    r = cfg["relational"]
    lines = [
        "# Recommended relevance preset",
        f"GRAPH_RELEVANCE_LEXICAL_WEIGHT={g['lexical']}",
        f"GRAPH_RELEVANCE_ENTITY_WEIGHT={g['entity']}",
        f"GRAPH_RELEVANCE_COMPLETENESS_WEIGHT={g['completeness']}",
        f"GRAPH_RELEVANCE_HINT_WEIGHT={g['hint']}",
        f"GRAPH_RELEVANCE_FALLBACK_LEXICAL_WEIGHT={g['fallback_lexical']}",
        f"GRAPH_RELEVANCE_FALLBACK_HINT_WEIGHT={g['fallback_hint']}",
        "",
        f"RELATIONAL_RELEVANCE_LEXICAL_WEIGHT={r['lexical']}",
        f"RELATIONAL_RELEVANCE_ENTITY_WEIGHT={r['entity']}",
        f"RELATIONAL_RELEVANCE_COMPLETENESS_WEIGHT={r['completeness']}",
        f"RELATIONAL_RELEVANCE_HINT_WEIGHT={r['hint']}",
        f"RELATIONAL_RELEVANCE_FALLBACK_LEXICAL_WEIGHT={r['fallback_lexical']}",
        f"RELATIONAL_RELEVANCE_FALLBACK_HINT_WEIGHT={r['fallback_hint']}",
    ]
    return "\n".join(lines)


def _write_markdown(summary: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"relevance_recommendation_{ts}.md"

    ranked = summary.get("ranked_presets", [])
    best = summary.get("recommended") or {}
    best_name = best.get("preset", "balanced")

    rows = [
        "| preset | probe_top1_mean | dataset_top1_mean | dataset_top3_mean | composite_score |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in ranked:
        rows.append(
            f"| {item['preset']} | {item['probe_top1_mean']:.4f} | {item['dataset_top1_mean']:.4f} | {item['dataset_top3_mean']:.4f} | {item['composite_score']:.4f} |"
        )

    md = [
        "# Relevance Preset Recommendation",
        "",
        f"- Source CSV: {summary.get('source_csv', '')}",
        f"- Recommended preset: **{best_name}**",
        "",
        "## Ranking",
        "",
        *rows,
        "",
        "## Suggested .env Settings",
        "",
        "```dotenv",
        _env_snippet(best_name),
        "```",
        "",
        "## Notes",
        "",
    ]
    for note in summary.get("notes", []):
        md.append(f"- {note}")

    path.write_text("\n".join(md), encoding="utf-8")
    return path


def main() -> None:
    csv_path = _latest_csv()
    summary = _summarize(csv_path)
    md_path = _write_markdown(summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nRECOMMENDATION_MD={md_path}")


if __name__ == "__main__":
    main()
