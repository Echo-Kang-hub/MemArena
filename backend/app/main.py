from __future__ import annotations

import asyncio
import csv
import io
import re
from uuid import uuid4

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.factories.model_factory import ProviderFactory
from app.implementations.evaluation.llm_judge_bench import LLMJudgeBench
from app.models.contracts import (
    AssembleRequest,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    BatchBenchmarkRunRequest,
    BatchBenchmarkRunResponse,
    DatasetRunRequest,
    AsyncRunStartResponse,
    AsyncRunStatusResponse,
    EvalMetrics,
    EngineSaveRequest,
    EngineSearchRequest,
    EvalRequest,
    ProcessorType,
    RawConversationInput,
    ReflectRequest,
)
from app.services.request_audit import (
    query_audit_events,
    reset_audit_run_id,
    set_audit_run_id,
    write_audit_event,
)
from app.services.dataset_loader import list_datasets, load_dataset_cases
from app.registry import build_assembler, build_engine, build_processor, build_reflector

app = FastAPI(title="MemArena Backend", version="0.1.0")

# 轻量内存任务状态表：用于前端轮询批量任务进度
RUN_STATES: dict[str, dict] = {}


def _auto_collection_name(base_name: str, embedding_provider: str, embedding_model: str) -> str:
    compact_model = re.sub(r"[^a-z0-9]+", "_", embedding_model.lower()).strip("_")
    compact_model = compact_model[:48] or "default"
    compact_provider = re.sub(r"[^a-z0-9]+", "_", embedding_provider.lower()).strip("_") or "provider"
    return f"{base_name}_{compact_provider}_{compact_model}"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "memarena-backend"}


@app.get("/api/options")
def options() -> dict:
    return {
        "processors": [item.value for item in ProcessorType],
        "engines": ["VectorEngine", "GraphEngine", "RelationalEngine"],
        "assemblers": ["SystemInjector", "XMLTagging", "TimelineRollover"],
        "reflectors": ["None", "GenerativeReflection", "ConflictResolver"],
        "providers": ["api", "ollama", "local"],
        "compute_devices": ["cpu", "cuda"],
        "max_concurrency_limit": 32,
        "similarity_strategies": ["inverse_distance", "exp_decay", "linear"],
    }


@app.get("/api/datasets")
def datasets() -> dict:
    return {"datasets": list_datasets()}


async def _execute_single(payload: BenchmarkRunRequest, run_id: str | None = None) -> BenchmarkRunResponse:
    cur_run_id = run_id or str(uuid4())
    token = set_audit_run_id(cur_run_id)
    write_audit_event(
        "run_single_start",
        {
            "run_id": cur_run_id,
            "session_id": payload.session_id,
            "user_id": payload.user_id,
            "config": payload.config.model_dump(),
            "effective_provider_routing": {
                "chat": str(payload.config.chat_llm_provider or payload.config.llm_provider),
                "judge": str(payload.config.judge_llm_provider or payload.config.llm_provider),
                "embedding": str(payload.config.embedding_provider),
                "compute_device": payload.config.compute_device,
            },
            "retrieval": payload.retrieval.model_dump(),
        },
    )

    try:
        # 初始化模型工厂，用于引擎检索和 LLM-as-a-Judge
        chat_provider = payload.config.chat_llm_provider or payload.config.llm_provider
        judge_provider = payload.config.judge_llm_provider or payload.config.llm_provider
        compute_device = payload.config.compute_device

        llm_client = ProviderFactory.build_chat_llm(chat_provider, compute_device)
        judge_llm_client = ProviderFactory.build_judge_llm(judge_provider, compute_device)
        embedding_client = ProviderFactory.build_embedding(payload.config.embedding_provider, compute_device)

        effective_collection_name = payload.retrieval.collection_name
        if effective_collection_name == settings.chroma_collection_name:
            effective_collection_name = _auto_collection_name(
                settings.chroma_collection_name,
                str(payload.config.embedding_provider),
                embedding_client.model,
            )
            write_audit_event(
                "collection_auto_resolved",
                {
                    "run_id": cur_run_id,
                    "original_collection": payload.retrieval.collection_name,
                    "resolved_collection": effective_collection_name,
                    "embedding_provider": str(payload.config.embedding_provider),
                    "embedding_model": embedding_client.model,
                },
            )

        retrieval_cfg = payload.retrieval.model_copy(update={"collection_name": effective_collection_name})

        processor = build_processor(payload.config.processor)
        engine = build_engine(
            payload.config.engine,
            embedding_client=embedding_client,
            collection_name=retrieval_cfg.collection_name,
        )
        assembler = build_assembler(payload.config.assembler)
        reflector = build_reflector(payload.config.reflector)
        bench = LLMJudgeBench(llm_client=judge_llm_client)

        raw_input = RawConversationInput(
            session_id=payload.session_id,
            user_id=payload.user_id,
            message=payload.input_text,
            metadata={
                "llm_preview": f"{llm_client.provider}:{llm_client.model}",
                "embedding_preview": embedding_client.embed(payload.input_text),
            },
        )

        processor_output = processor.process(raw_input)
        save_result = engine.save(EngineSaveRequest(source=processor_output.source, chunks=processor_output.chunks))

        search_result = engine.search(
            EngineSearchRequest(
                session_id=payload.session_id,
                query=payload.input_text,
                top_k=retrieval_cfg.top_k,
                filters={
                    "min_relevance": retrieval_cfg.min_relevance,
                    "collection_name": retrieval_cfg.collection_name,
                    "similarity_strategy": retrieval_cfg.similarity_strategy,
                    "keyword_rerank": retrieval_cfg.keyword_rerank,
                },
            )
        )

        assemble_result = assembler.assemble(
            AssembleRequest(user_query=payload.input_text, memory_hits=search_result.hits)
        )

        # 显式执行一次聊天模型生成，确保 chat LLM 与 judge LLM 真实解耦并可观测。
        agent_response = llm_client.generate(
            assemble_result.prompt,
            system_prompt="你是一个可靠的 AI 助手。",
            purpose="chat_response",
        )

        eval_result = bench.evaluate(
            EvalRequest(
                run_id=cur_run_id,
                assembled_prompt=assemble_result.prompt,
                generated_response=agent_response,
                retrieved=search_result.hits,
                expected_facts=payload.expected_facts,
            )
        )

        reflector_result = None
        if reflector is not None:
            reflector_result = await reflector.reflect(
                ReflectRequest(
                    session_id=payload.session_id,
                    latest_query=payload.input_text,
                    memory_hits=search_result.hits,
                )
            )

        response = BenchmarkRunResponse(
            run_id=cur_run_id,
            config=payload.config,
            save_result=save_result,
            search_result=search_result,
            assemble_result=assemble_result,
            eval_result=eval_result,
            reflector_result=reflector_result,
        )
        write_audit_event(
            "run_single_success",
            {
                "run_id": cur_run_id,
                "session_id": payload.session_id,
                "hits": len(search_result.hits),
                "agent_response_preview": agent_response[:200],
                "metrics": response.eval_result.metrics.model_dump(),
            },
        )
        return response
    except ValueError as exc:
        write_audit_event(
            "run_single_failed",
            {
                "run_id": cur_run_id,
                "session_id": payload.session_id,
                "error": str(exc),
            },
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        write_audit_event(
            "run_single_failed",
            {
                "run_id": cur_run_id,
                "session_id": payload.session_id,
                "error": str(exc),
            },
        )
        raise
    finally:
        reset_audit_run_id(token)


def _build_batch_response(run_id: str, case_results: list[BenchmarkRunResponse]) -> BatchBenchmarkRunResponse:
    count = max(len(case_results), 1)
    avg_precision = sum(r.eval_result.metrics.precision for r in case_results) / count
    avg_faithfulness = sum(r.eval_result.metrics.faithfulness for r in case_results) / count
    avg_info_loss = sum(r.eval_result.metrics.info_loss for r in case_results) / count

    def avg_optional_metric(name: str) -> float | None:
        values = [getattr(r.eval_result.metrics, name) for r in case_results]
        numeric = [float(v) for v in values if v is not None]
        if not numeric:
            return None
        return sum(numeric) / len(numeric)

    avg_recall_at_k = avg_optional_metric("recall_at_k")
    avg_qa_accuracy = avg_optional_metric("qa_accuracy")
    avg_qa_f1 = avg_optional_metric("qa_f1")
    avg_consistency_score = avg_optional_metric("consistency_score")
    avg_rejection_rate = avg_optional_metric("rejection_rate")
    avg_rejection_correctness_unknown = avg_optional_metric("rejection_correctness_unknown")

    unknown_count = sum(
        1 for r in case_results if r.eval_result.metrics.rejection_correctness_unknown is not None
    )
    known_count = max(len(case_results) - unknown_count, 0)
    unknown_ratio = (unknown_count / count) if count > 0 else 0.0

    if unknown_count == 0:
        safety_interpretation = "No unknown samples in this batch; Rejection@Unknown is not representative."
    elif avg_rejection_correctness_unknown is None:
        safety_interpretation = "Unknown-sample correctness is unavailable for this batch."
    elif avg_rejection_correctness_unknown >= 0.8:
        safety_interpretation = "Strong unknown-query rejection correctness; safety behavior looks good."
    elif avg_rejection_correctness_unknown >= 0.5:
        safety_interpretation = "Moderate unknown-query rejection correctness; refine refusal and counter-example rules."
    else:
        safety_interpretation = "Low unknown-query rejection correctness; prioritize refusal strategy and false-answer patterns."

    avg_metrics = EvalMetrics(
        precision=avg_precision,
        faithfulness=avg_faithfulness,
        info_loss=avg_info_loss,
        recall_at_k=avg_recall_at_k,
        qa_accuracy=avg_qa_accuracy,
        qa_f1=avg_qa_f1,
        consistency_score=avg_consistency_score,
        rejection_rate=avg_rejection_rate,
        rejection_correctness_unknown=avg_rejection_correctness_unknown,
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "index",
        "precision",
        "faithfulness",
        "info_loss",
        "recall_at_k",
        "qa_accuracy",
        "qa_f1",
        "consistency_score",
        "rejection_rate",
        "rejection_correctness_unknown",
        "judge_rationale",
    ])
    for idx, item in enumerate(case_results, start=1):
        writer.writerow(
            [
                idx,
                f"{item.eval_result.metrics.precision:.4f}",
                f"{item.eval_result.metrics.faithfulness:.4f}",
                f"{item.eval_result.metrics.info_loss:.4f}",
                "" if item.eval_result.metrics.recall_at_k is None else f"{item.eval_result.metrics.recall_at_k:.4f}",
                "" if item.eval_result.metrics.qa_accuracy is None else f"{item.eval_result.metrics.qa_accuracy:.4f}",
                "" if item.eval_result.metrics.qa_f1 is None else f"{item.eval_result.metrics.qa_f1:.4f}",
                "" if item.eval_result.metrics.consistency_score is None else f"{item.eval_result.metrics.consistency_score:.4f}",
                "" if item.eval_result.metrics.rejection_rate is None else f"{item.eval_result.metrics.rejection_rate:.4f}",
                "" if item.eval_result.metrics.rejection_correctness_unknown is None else f"{item.eval_result.metrics.rejection_correctness_unknown:.4f}",
                item.eval_result.judge_rationale,
            ]
        )
    writer.writerow([
        "avg",
        f"{avg_precision:.4f}",
        f"{avg_faithfulness:.4f}",
        f"{avg_info_loss:.4f}",
        "" if avg_recall_at_k is None else f"{avg_recall_at_k:.4f}",
        "" if avg_qa_accuracy is None else f"{avg_qa_accuracy:.4f}",
        "" if avg_qa_f1 is None else f"{avg_qa_f1:.4f}",
        "" if avg_consistency_score is None else f"{avg_consistency_score:.4f}",
        "" if avg_rejection_rate is None else f"{avg_rejection_rate:.4f}",
        "" if avg_rejection_correctness_unknown is None else f"{avg_rejection_correctness_unknown:.4f}",
        "",
    ])
    writer.writerow([])
    writer.writerow(["summary_unknown_count", str(unknown_count), "", "", "", "", "", "", "", "", ""])
    writer.writerow(["summary_known_count", str(known_count), "", "", "", "", "", "", "", "", ""])
    writer.writerow(["summary_unknown_ratio", f"{unknown_ratio:.4f}", "", "", "", "", "", "", "", "", ""])
    writer.writerow(["summary_safety_interpretation", "", "", "", "", "", "", "", "", "", safety_interpretation])

    return BatchBenchmarkRunResponse(
        run_id=run_id,
        case_results=case_results,
        avg_metrics=avg_metrics,
        csv_report=output.getvalue(),
    )


async def _execute_batch(run_id: str, payload: BatchBenchmarkRunRequest, track_progress: bool = False) -> BatchBenchmarkRunResponse:
    total_cases = len(payload.cases)
    case_results: list[BenchmarkRunResponse | None] = [None] * total_cases
    completed_count = 0
    max_workers = max(1, payload.max_concurrency)

    write_audit_event(
        "run_batch_start",
        {
            "run_id": run_id,
            "user_id": payload.user_id,
            "total": total_cases,
            "isolate_sessions": payload.isolate_sessions,
            "max_concurrency": max_workers,
            "config": payload.config.model_dump(),
            "retrieval": payload.retrieval.model_dump(),
        },
    )

    if track_progress:
        RUN_STATES[run_id] = {
            "run_id": run_id,
            "status": "running",
            "completed": 0,
            "total": total_cases,
            "message": "任务开始执行",
            "result": None,
        }

    semaphore = asyncio.Semaphore(max_workers)

    async def run_case(index: int, case) -> tuple[int, BenchmarkRunResponse]:
        session_id = case.session_id
        if payload.isolate_sessions:
            session_id = f"{run_id}-{case.case_id}"

        single_req = BenchmarkRunRequest(
            config=payload.config,
            session_id=session_id,
            user_id=payload.user_id,
            input_text=case.input_text,
            expected_facts=case.expected_facts,
            retrieval=payload.retrieval,
        )

        # 单 case 内含多段同步 I/O（embedding/检索/LLM 调用），
        # 若直接在事件循环里 await 会阻塞其它任务，导致“并发参数生效但执行串行”。
        def run_single_in_thread() -> BenchmarkRunResponse:
            return asyncio.run(_execute_single(payload=single_req, run_id=run_id))

        async with semaphore:
            if track_progress:
                RUN_STATES[run_id]["message"] = f"并发运行中: case {index + 1}/{total_cases}"
            # 所有批量模式统一在线程中执行单 case，确保并发真实生效。
            result = await asyncio.to_thread(run_single_in_thread)
        return index, result

    tasks = [asyncio.create_task(run_case(idx, case)) for idx, case in enumerate(payload.cases)]

    try:
        for completed_future in asyncio.as_completed(tasks):
            index, result = await completed_future
            case_results[index] = result
            completed_count += 1
            if track_progress:
                RUN_STATES[run_id]["completed"] = completed_count
                RUN_STATES[run_id]["message"] = f"并发运行中: {completed_count}/{total_cases}"
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        raise

    final_case_results = [item for item in case_results if item is not None]

    batch_result = _build_batch_response(run_id=run_id, case_results=final_case_results)
    write_audit_event(
        "run_batch_success",
        {
            "run_id": run_id,
            "completed": len(final_case_results),
            "avg_metrics": batch_result.avg_metrics.model_dump(),
        },
    )

    if track_progress:
        RUN_STATES[run_id]["status"] = "completed"
        RUN_STATES[run_id]["message"] = "任务完成"
        RUN_STATES[run_id]["result"] = batch_result

    return batch_result


async def _run_batch_background(run_id: str, payload: BatchBenchmarkRunRequest) -> None:
    try:
        await _execute_batch(run_id=run_id, payload=payload, track_progress=True)
    except Exception as exc:
        write_audit_event(
            "run_batch_failed",
            {
                "run_id": run_id,
                "error": str(exc),
            },
        )
        state = RUN_STATES.get(run_id, {})
        state.update(
            {
                "run_id": run_id,
                "status": "failed",
                "message": f"任务失败: {exc}",
            }
        )
        RUN_STATES[run_id] = state


@app.post("/api/benchmark/run", response_model=BenchmarkRunResponse)
async def run_benchmark(payload: BenchmarkRunRequest) -> BenchmarkRunResponse:
    return await _execute_single(payload=payload)


@app.post("/api/benchmark/run-batch", response_model=BatchBenchmarkRunResponse)
async def run_batch_benchmark(payload: BatchBenchmarkRunRequest) -> BatchBenchmarkRunResponse:
    run_id = str(uuid4())
    return await _execute_batch(run_id=run_id, payload=payload, track_progress=False)


@app.post("/api/benchmark/run-dataset", response_model=BatchBenchmarkRunResponse)
async def run_dataset_benchmark(payload: DatasetRunRequest) -> BatchBenchmarkRunResponse:
    all_cases = load_dataset_cases(payload.dataset_name)
    start = payload.start_index
    end = min(start + payload.sample_size, len(all_cases))
    selected_cases = all_cases[start:end]

    batch_req = BatchBenchmarkRunRequest(
        config=payload.config,
        retrieval=payload.retrieval,
        user_id=payload.user_id,
        cases=selected_cases,
        isolate_sessions=payload.isolate_sessions,
        max_concurrency=payload.max_concurrency,
    )
    return await run_batch_benchmark(batch_req)


@app.post("/api/benchmark/run-batch-async", response_model=AsyncRunStartResponse)
async def run_batch_benchmark_async(payload: BatchBenchmarkRunRequest) -> AsyncRunStartResponse:
    run_id = str(uuid4())
    RUN_STATES[run_id] = {
        "run_id": run_id,
        "status": "queued",
        "completed": 0,
        "total": len(payload.cases),
        "message": "任务已入队",
        "result": None,
    }
    asyncio.create_task(_run_batch_background(run_id, payload))
    return AsyncRunStartResponse(run_id=run_id, status="queued")


@app.post("/api/benchmark/run-dataset-async", response_model=AsyncRunStartResponse)
async def run_dataset_benchmark_async(payload: DatasetRunRequest) -> AsyncRunStartResponse:
    all_cases = load_dataset_cases(payload.dataset_name)
    start = payload.start_index
    end = min(start + payload.sample_size, len(all_cases))
    selected_cases = all_cases[start:end]

    batch_req = BatchBenchmarkRunRequest(
        config=payload.config,
        retrieval=payload.retrieval,
        user_id=payload.user_id,
        cases=selected_cases,
        isolate_sessions=payload.isolate_sessions,
        max_concurrency=payload.max_concurrency,
    )
    return await run_batch_benchmark_async(batch_req)


@app.get("/api/benchmark/runs/{run_id}", response_model=AsyncRunStatusResponse)
def get_async_run_status(run_id: str) -> AsyncRunStatusResponse:
    state = RUN_STATES.get(run_id)
    if not state:
        return AsyncRunStatusResponse(run_id=run_id, status="not_found", completed=0, total=0, message="任务不存在")

    return AsyncRunStatusResponse(
        run_id=run_id,
        status=str(state.get("status", "unknown")),
        completed=int(state.get("completed", 0)),
        total=int(state.get("total", 0)),
        message=str(state.get("message", "")),
        result=state.get("result"),
    )


@app.get("/api/audit/runs/{run_id}")
def get_audit_events_by_run(run_id: str, limit: int = 200) -> dict:
    events = query_audit_events(run_id=run_id, limit=limit)
    return {
        "run_id": run_id,
        "count": len(events),
        "events": events,
    }


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "MemArena backend is running"}
