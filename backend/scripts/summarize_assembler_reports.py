from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "backend" / "data" / "reports"
DEFAULT_TARGET_ASSEMBLERS = ["SystemInjector", "XMLTagging", "RankedPruning", "ReasoningChain"]


def _parse_weights(raw: str) -> dict[str, float]:
    defaults = {
        "precision": 0.35,
        "faithfulness": 0.25,
        "info_loss": 0.20,
        "context_distraction": 0.10,
        "seed_touch": 0.10,
    }
    text = (raw or "").strip()
    if not text:
        return defaults

    out = dict(defaults)
    for pair in text.split(","):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        key = k.strip()
        if key not in out:
            continue
        try:
            out[key] = float(v.strip())
        except ValueError:
            continue
    return out


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


def _append(bucket: dict[str, list[float]], key: str, value: float | None) -> None:
    if value is not None:
        bucket.setdefault(key, []).append(value)


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _collect_rows(
    report_dir: Path,
    csv_glob: str,
    target_assemblers: list[str],
) -> tuple[list[Path], dict[str, dict[str, list[float]]]]:
    csv_files = sorted(report_dir.glob(csv_glob))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {report_dir} with glob={csv_glob}")

    per_assembler: dict[str, dict[str, list[float]]] = {}
    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assembler = (row.get("assembler") or "").strip()
                if not assembler:
                    continue
                if assembler not in target_assemblers:
                    continue

                bucket = per_assembler.setdefault(assembler, {})
                _append(bucket, "precision", _to_float(row.get("precision")))
                _append(bucket, "faithfulness", _to_float(row.get("faithfulness")))
                _append(bucket, "info_loss", _to_float(row.get("info_loss")))
                _append(bucket, "context_distraction", _to_float(row.get("context_distraction")))
                _append(bucket, "reasoning_chain_count", _to_float(row.get("reasoning_chain_count")))
                _append(bucket, "reasoning_avg_priority", _to_float(row.get("reasoning_avg_priority")))
                _append(bucket, "reasoning_avg_hop", _to_float(row.get("reasoning_avg_hop")))
                _append(bucket, "reasoning_seed_touch_ratio", _to_float(row.get("reasoning_seed_touch_ratio")))
                _append(bucket, "composite_score", _to_float(row.get("composite_score")))

    return csv_files, per_assembler


def _build_ranked(
    per_assembler: dict[str, dict[str, list[float]]],
    target_assemblers: list[str],
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for assembler in target_assemblers:
        bucket = per_assembler.get(assembler, {})

        precision = _mean(bucket.get("precision", []))
        faithfulness = _mean(bucket.get("faithfulness", []))
        info_loss = _mean(bucket.get("info_loss", []))
        context_distraction = _mean(bucket.get("context_distraction", []))
        chain_count = _mean(bucket.get("reasoning_chain_count", []))
        avg_priority = _mean(bucket.get("reasoning_avg_priority", []))
        avg_hop = _mean(bucket.get("reasoning_avg_hop", []))
        seed_touch = _mean(bucket.get("reasoning_seed_touch_ratio", []))
        composite = _mean(bucket.get("composite_score", []))

        if composite is None:
            base = 0.0
            if precision is not None:
                base += float(weights["precision"]) * precision
            if faithfulness is not None:
                base += float(weights["faithfulness"]) * faithfulness
            if info_loss is not None:
                base += float(weights["info_loss"]) * (1.0 - info_loss)
            if context_distraction is not None:
                base += float(weights["context_distraction"]) * (1.0 - context_distraction)
            if seed_touch is not None:
                base += float(weights["seed_touch"]) * seed_touch
            composite = base

        ranked.append(
            {
                "assembler": assembler,
                "runs": len(bucket.get("precision", [])),
                "precision": precision,
                "faithfulness": faithfulness,
                "info_loss": info_loss,
                "context_distraction": context_distraction,
                "reasoning_chain_count": chain_count,
                "reasoning_avg_priority": avg_priority,
                "reasoning_avg_hop": avg_hop,
                "reasoning_seed_touch_ratio": seed_touch,
                "composite_score": composite or 0.0,
            }
        )

    ranked.sort(key=lambda x: float(x.get("composite_score") or 0.0), reverse=True)
    return ranked


def _fmt(v: float | None, ndigits: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.{ndigits}f}"


def _write_reports(report_dir: Path, csv_files: list[Path], ranked: list[dict[str, Any]], target_assemblers: list[str]) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"assembler_summary_{stamp}.json"
    md_path = report_dir / f"assembler_summary_{stamp}.md"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "source_files": [str(p) for p in csv_files],
        "ranked": ranked,
        "recommended": ranked[0]["assembler"] if ranked else None,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    table = [
        "| rank | assembler | runs | precision | faithfulness | info_loss | context_distraction | chain_count | avg_priority | avg_hop | seed_touch_ratio | composite_score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(ranked, start=1):
        table.append(
            "| {rank} | {assembler} | {runs} | {precision} | {faithfulness} | {info_loss} | {context_distraction} | {chain_count} | {avg_priority} | {avg_hop} | {seed_touch} | {composite} |".format(
                rank=idx,
                assembler=row["assembler"],
                runs=row["runs"],
                precision=_fmt(row.get("precision")),
                faithfulness=_fmt(row.get("faithfulness")),
                info_loss=_fmt(row.get("info_loss")),
                context_distraction=_fmt(row.get("context_distraction")),
                chain_count=_fmt(row.get("reasoning_chain_count"), 2),
                avg_priority=_fmt(row.get("reasoning_avg_priority")),
                avg_hop=_fmt(row.get("reasoning_avg_hop"), 2),
                seed_touch=_fmt(row.get("reasoning_seed_touch_ratio")),
                composite=_fmt(row.get("composite_score")),
            )
        )

    md_lines = [
        "# Assembler Ranking Summary",
        "",
        f"- Included reports: {len(csv_files)}",
        f"- Recommended: **{payload['recommended'] or 'N/A'}**",
        "",
        "## Target Assemblers",
        *[f"- {name}" for name in target_assemblers],
        "",
        "## Ranking",
        "",
        *table,
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize assembler comparison CSV reports")
    parser.add_argument("--report-dir", default=str(REPORT_DIR), help="Directory containing assembler CSV reports")
    parser.add_argument("--csv-glob", default="assembler_compare_*.csv", help="Glob for CSV files inside report-dir")
    parser.add_argument(
        "--assemblers",
        default=",".join(DEFAULT_TARGET_ASSEMBLERS),
        help="Comma-separated assembler names to include",
    )
    parser.add_argument(
        "--weights",
        default="precision=0.35,faithfulness=0.25,info_loss=0.20,context_distraction=0.10,seed_touch=0.10",
        help="Comma-separated weights, e.g. precision=0.4,faithfulness=0.3,info_loss=0.2,context_distraction=0.05,seed_touch=0.05",
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    target_assemblers = [x.strip() for x in str(args.assemblers).split(",") if x.strip()]
    if not target_assemblers:
        target_assemblers = list(DEFAULT_TARGET_ASSEMBLERS)
    weights = _parse_weights(str(args.weights))

    try:
        csv_files, per_assembler = _collect_rows(
            report_dir=report_dir,
            csv_glob=str(args.csv_glob),
            target_assemblers=target_assemblers,
        )
    except FileNotFoundError as exc:
        print(str(exc))
        print("Hint: run `python backend/scripts/benchmark_assembler_comparison.py` first.")
        return

    ranked = _build_ranked(per_assembler, target_assemblers=target_assemblers, weights=weights)
    json_path, md_path = _write_reports(report_dir, csv_files, ranked, target_assemblers=target_assemblers)

    print(json.dumps({"ranked": ranked}, ensure_ascii=False, indent=2))
    print(f"\nASSEMBLER_SUMMARY_JSON={json_path}")
    print(f"ASSEMBLER_SUMMARY_MD={md_path}")


if __name__ == "__main__":
    main()
