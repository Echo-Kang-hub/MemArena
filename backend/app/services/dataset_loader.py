from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.models.contracts import BatchCase


def dataset_root() -> Path:
    # 数据集目录固定为 backend/datasets
    return Path(__file__).resolve().parents[2] / "datasets"


# Avoid loading very large files fully when only listing dataset options.
LIST_COUNT_MAX_BYTES = 8 * 1024 * 1024


def _safe_dataset_count(path: Path) -> int:
    try:
        if path.stat().st_size > LIST_COUNT_MAX_BYTES:
            return -1
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
        return 0
    except Exception:
        return -1


def list_datasets() -> list[dict[str, Any]]:
    root = dataset_root()
    root.mkdir(parents=True, exist_ok=True)

    datasets: list[dict[str, Any]] = []
    json_files = sorted(root.glob("*.json"))
    jsonl_files = sorted(root.glob("*.jsonl"))
    for p in [*json_files, *jsonl_files]:
        count = _safe_dataset_count(p) if p.suffix.lower() == ".json" else -1
        datasets.append({"name": p.stem, "file": p.name, "count": count})
    return datasets


def load_dataset_cases(dataset_name: str) -> list[BatchCase]:
    path = dataset_root() / f"{dataset_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Dataset file must be a JSON array")

    cases: list[BatchCase] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("case_id") or f"case-{idx}")
        input_text = str(item.get("input_text") or "")
        expected_facts = item.get("expected_facts") or []
        if not isinstance(expected_facts, list):
            expected_facts = []
        cases.append(
            BatchCase(
                case_id=case_id,
                input_text=input_text,
                expected_facts=[str(v) for v in expected_facts],
                session_id=str(item.get("session_id") or "dataset-session"),
            )
        )
    return cases
