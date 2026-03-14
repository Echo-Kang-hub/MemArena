from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.implementations.evaluation.llm_judge_bench import LLMJudgeBench  # noqa: E402
from app.main import app  # noqa: E402
from app.models.contracts import EvalRequest, MemoryHit  # noqa: E402


REPORT_DIR = ROOT / "backend" / "data" / "reports"


def _build_correction_payload(reflector: str, session_id: str) -> dict[str, Any]:
    # 使用同一 session 连续写入“旧事实 -> 更正事实”，触发收敛速度估计。
    return {
        "config": {
            "processor": "EntityExtractor",
            "engine": "RelationalEngine",
            "assembler": "SystemInjector",
            "reflector": reflector,
            "llm_provider": "local",
            "chat_llm_provider": "local",
            "judge_llm_provider": "local",
            "summarizer_llm_provider": "local",
            "entity_llm_provider": "local",
            "embedding_provider": "local",
            "summarizer_method": "llm",
            "entity_extractor_method": "llm_attribute",
            "compute_device": "cpu",
        },
        "retrieval": {
            "top_k": 5,
            "min_relevance": 0.0,
            "collection_name": "memarena_convergence_smoke",
            "similarity_strategy": "inverse_distance",
            "keyword_rerank": False,
        },
        "user_id": "convergence-smoke-user",
        "cases": [
            {
                "case_id": "c1",
                "input_text": "我住在北京。",
                "expected_facts": ["北京"],
                "session_id": session_id,
            },
            {
                "case_id": "c2",
                "input_text": "更正：我不住在北京，我现在住在上海。",
                "expected_facts": ["上海"],
                "session_id": session_id,
            },
        ],
        "isolate_sessions": False,
        "max_concurrency": 1,
    }


def _run_api_smoke(client: TestClient, reflector: str, session_id: str) -> dict[str, Any]:
    payload = _build_correction_payload(reflector=reflector, session_id=session_id)
    resp = client.post("/api/benchmark/run-batch", json=payload)
    resp.raise_for_status()
    data = resp.json()

    case_results = data.get("case_results", [])
    case_convergence = [
        ((case.get("eval_result") or {}).get("metrics") or {}).get("convergence_speed")
        for case in case_results
    ]

    return {
        "reflector": reflector,
        "run_id": data.get("run_id"),
        "avg_convergence_speed": ((data.get("avg_metrics") or {}).get("convergence_speed")),
        "case_convergence_speed": case_convergence,
        "csv_header": (data.get("csv_report", "").splitlines() or [""])[0],
    }


def _deterministic_probe() -> dict[str, Any]:
    # 构造两个冲突组：
    # - user::city: beijing vs shanghai
    # - user::drink: coffee vs tea
    # 公式为 min(6, 1 + conflict_groups) => 3.0
    bench = LLMJudgeBench(llm_client=None)
    req = EvalRequest(
        run_id="deterministic-probe",
        assembled_prompt="更正：我不是北京人，住在上海；另外我不喝咖啡，改喝茶。",
        generated_response="我已更新：住在上海，喝茶。",
        retrieved=[
            MemoryHit(
                chunk_id="h1",
                content="旧记忆",
                relevance=0.9,
                metadata={
                    "attributes": [
                        {"entity": "user", "attribute": "city", "value": "beijing"},
                        {"entity": "user", "attribute": "drink", "value": "coffee"},
                    ]
                },
            ),
            MemoryHit(
                chunk_id="h2",
                content="新记忆",
                relevance=0.95,
                metadata={
                    "attributes": [
                        {"entity": "user", "attribute": "city", "value": "shanghai"},
                        {"entity": "user", "attribute": "drink", "value": "tea"},
                    ]
                },
            ),
        ],
        expected_facts=["上海", "茶"],
    )

    score = bench._compute_convergence_speed(req)
    return {
        "probe_name": "two_conflict_groups",
        "expected": 3.0,
        "actual": score,
        "note": "conflict_groups=2 -> convergence_speed=1+2=3",
    }


def _write_reports(results: dict[str, Any]) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORT_DIR / f"convergence_smoke_{stamp}.json"
    csv_path = REPORT_DIR / f"convergence_smoke_{stamp}.csv"

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "kind",
        "reflector",
        "run_id",
        "avg_convergence_speed",
        "case1_convergence_speed",
        "case2_convergence_speed",
        "csv_header",
        "probe_name",
        "expected",
        "actual",
        "note",
    ]
    rows: list[dict[str, Any]] = []

    for item in results.get("api_smoke", []):
        case_values = item.get("case_convergence_speed") or []
        rows.append(
            {
                "kind": "api_smoke",
                "reflector": item.get("reflector", ""),
                "run_id": item.get("run_id", ""),
                "avg_convergence_speed": item.get("avg_convergence_speed", ""),
                "case1_convergence_speed": case_values[0] if len(case_values) > 0 else "",
                "case2_convergence_speed": case_values[1] if len(case_values) > 1 else "",
                "csv_header": item.get("csv_header", ""),
                "probe_name": "",
                "expected": "",
                "actual": "",
                "note": "",
            }
        )

    probe = results.get("deterministic_probe", {})
    rows.append(
        {
            "kind": "deterministic_probe",
            "reflector": "",
            "run_id": "",
            "avg_convergence_speed": "",
            "case1_convergence_speed": "",
            "case2_convergence_speed": "",
            "csv_header": "",
            "probe_name": probe.get("probe_name", ""),
            "expected": probe.get("expected", ""),
            "actual": probe.get("actual", ""),
            "note": probe.get("note", ""),
        }
    )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return json_path, csv_path


def main() -> None:
    results: dict[str, Any] = {"api_smoke": [], "deterministic_probe": {}}

    with TestClient(app) as client:
        results["api_smoke"].append(_run_api_smoke(client, reflector="ConflictResolver", session_id="conv-smoke-a"))
        results["api_smoke"].append(_run_api_smoke(client, reflector="Consolidator", session_id="conv-smoke-b"))

    results["deterministic_probe"] = _deterministic_probe()
    json_path, csv_path = _write_reports(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nCONVERGENCE_SMOKE_JSON={json_path}")
    print(f"CONVERGENCE_SMOKE_CSV={csv_path}")


if __name__ == "__main__":
    main()
