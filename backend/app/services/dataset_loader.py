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
_DATASET_COUNT_CACHE: dict[str, tuple[int, float, int]] = {}


def _count_by_record_markers(path: Path) -> int:
    """Fast count for large JSON files by counting common per-record keys."""
    markers = {
        b'"case_id"': 0,
        b'"question_id"': 0,
        b'"in_dataset"': 0,
        b'"id"': 0,
    }
    with path.open("rb") as f:
        while True:
            chunk = f.read(2 * 1024 * 1024)
            if not chunk:
                break
            for marker in markers:
                markers[marker] += chunk.count(marker)

    for marker in (b'"case_id"', b'"question_id"', b'"in_dataset"', b'"id"'):
        if markers[marker] > 0:
            return int(markers[marker])
    return -1


def _adaptive_count_from_data(data: Any) -> int:
    if isinstance(data, list):
        return len(data)
    if not isinstance(data, dict):
        return -1

    split_keys = ["train", "validation", "valid", "dev", "test"]
    split_total = sum(len(data[k]) for k in split_keys if isinstance(data.get(k), list))
    if split_total > 0:
        return split_total

    top_level_lists = [v for v in data.values() if isinstance(v, list)]
    if len(top_level_lists) == 1:
        return len(top_level_lists[0])
    if len(top_level_lists) > 1:
        return sum(len(v) for v in top_level_lists)

    # Fallback for dict-of-records style JSON.
    if data and all(isinstance(v, dict) for v in data.values()):
        return len(data)

    return -1


def _count_json_array_items_stream(path: Path) -> int:
    """Count top-level JSON array items without loading the whole file into memory."""
    with path.open("r", encoding="utf-8") as f:
        # Move to first non-whitespace character.
        while True:
            ch = f.read(1)
            if not ch:
                return 0
            if not ch.isspace():
                break

        if ch != "[":
            return -1

        count = 0
        depth = 1
        in_string = False
        escape = False
        expecting_value = True

        while True:
            ch = f.read(1)
            if not ch:
                break

            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = False
                continue

            if ch == '"':
                if depth == 1 and expecting_value:
                    count += 1
                    expecting_value = False
                in_string = True
                continue

            if ch in " \t\r\n":
                continue

            if ch == "[":
                if depth == 1 and expecting_value:
                    count += 1
                    expecting_value = False
                depth += 1
                continue

            if ch == "{":
                if depth == 1 and expecting_value:
                    count += 1
                    expecting_value = False
                depth += 1
                continue

            if ch in "]}":
                depth -= 1
                if depth < 0:
                    return -1
                if depth == 0:
                    return count
                continue

            if depth == 1:
                if expecting_value:
                    # number / true / false / null
                    if ch == ",":
                        return -1
                    count += 1
                    expecting_value = False
                else:
                    if ch == ",":
                        expecting_value = True
                    elif ch == "]":
                        return count

        return count if depth == 0 else -1


def _count_jsonl_items(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _safe_dataset_count(path: Path) -> int:
    stat = path.stat()
    cache_key = str(path.resolve())
    cached = _DATASET_COUNT_CACHE.get(cache_key)
    if cached and cached[0] == stat.st_size and cached[1] == stat.st_mtime and cached[2] >= 0:
        return cached[2]

    try:
        if path.suffix.lower() == ".jsonl":
            count = _count_jsonl_items(path)
        elif stat.st_size > LIST_COUNT_MAX_BYTES:
            # Large files: prefer fast marker scan, fallback to exact stream count.
            count = _count_by_record_markers(path)
            if count < 0:
                count = _count_json_array_items_stream(path)
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            count = _adaptive_count_from_data(data)

        _DATASET_COUNT_CACHE[cache_key] = (stat.st_size, stat.st_mtime, count)
        return count
    except Exception:
        return -1


def list_datasets() -> list[dict[str, Any]]:
    root = dataset_root()
    root.mkdir(parents=True, exist_ok=True)

    datasets: list[dict[str, Any]] = []
    json_files = sorted(root.glob("*.json"))
    jsonl_files = sorted(root.glob("*.jsonl"))
    for p in [*json_files, *jsonl_files]:
        count = _safe_dataset_count(p)
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
