from __future__ import annotations

import json
from contextvars import ContextVar, Token
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from app.config import settings

_LOCK = Lock()
_RUN_ID_CTX: ContextVar[str | None] = ContextVar("audit_run_id", default=None)


def _log_file() -> Path:
    p = Path(settings.request_log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def set_audit_run_id(run_id: str | None) -> Token:
    return _RUN_ID_CTX.set(run_id)


def reset_audit_run_id(token: Token) -> None:
    _RUN_ID_CTX.reset(token)


def write_audit_event(event_type: str, payload: dict[str, Any]) -> None:
    # 采用 jsonl 便于后续按行检索、流式分析与导入日志系统
    run_id = _RUN_ID_CTX.get()
    merged_payload = dict(payload)
    if run_id and "run_id" not in merged_payload:
        merged_payload["run_id"] = run_id

    record = {
        "ts": datetime.utcnow().isoformat(),
        "event_type": event_type,
        **merged_payload,
    }
    with _LOCK:
        with _log_file().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def query_audit_events(run_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    path = _log_file()
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                continue

            if run_id and str(item.get("run_id", "")) != run_id:
                continue
            records.append(item)

    if limit <= 0:
        return records
    return records[-limit:]
