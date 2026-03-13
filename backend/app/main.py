from __future__ import annotations

import csv
import io
from uuid import uuid4

from fastapi import FastAPI
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
    EvalMetrics,
    EngineSaveRequest,
    EngineSearchRequest,
    EvalRequest,
    ProcessorType,
    RawConversationInput,
    ReflectRequest,
)
from app.registry import build_assembler, build_engine, build_processor, build_reflector

app = FastAPI(title="MemArena Backend", version="0.1.0")

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
        "similarity_strategies": ["inverse_distance", "exp_decay", "linear"],
    }


async def _execute_single(payload: BenchmarkRunRequest, run_id: str | None = None) -> BenchmarkRunResponse:
    cur_run_id = run_id or str(uuid4())

    # 初始化模型工厂，用于引擎检索和 LLM-as-a-Judge
    llm_client = ProviderFactory.build_llm(payload.config.llm_provider)
    embedding_client = ProviderFactory.build_embedding(payload.config.embedding_provider)

    processor = build_processor(payload.config.processor)
    engine = build_engine(
        payload.config.engine,
        embedding_client=embedding_client,
        collection_name=payload.retrieval.collection_name,
    )
    assembler = build_assembler(payload.config.assembler)
    reflector = build_reflector(payload.config.reflector)
    bench = LLMJudgeBench(llm_client=llm_client)

    raw_input = RawConversationInput(
        session_id=payload.session_id,
        user_id=payload.user_id,
        message=payload.input_text,
        metadata={
            "llm_preview": llm_client.generate("provider handshake"),
            "embedding_preview": embedding_client.embed(payload.input_text),
        },
    )

    processor_output = processor.process(raw_input)
    save_result = engine.save(EngineSaveRequest(source=processor_output.source, chunks=processor_output.chunks))

    search_result = engine.search(
        EngineSearchRequest(
            session_id=payload.session_id,
            query=payload.input_text,
            top_k=payload.retrieval.top_k,
            filters={
                "min_relevance": payload.retrieval.min_relevance,
                "collection_name": payload.retrieval.collection_name,
                "similarity_strategy": payload.retrieval.similarity_strategy,
                "keyword_rerank": payload.retrieval.keyword_rerank,
            },
        )
    )

    assemble_result = assembler.assemble(
        AssembleRequest(user_query=payload.input_text, memory_hits=search_result.hits)
    )

    eval_result = bench.evaluate(
        EvalRequest(
            run_id=cur_run_id,
            assembled_prompt=assemble_result.prompt,
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

    return BenchmarkRunResponse(
        run_id=cur_run_id,
        config=payload.config,
        save_result=save_result,
        search_result=search_result,
        assemble_result=assemble_result,
        eval_result=eval_result,
        reflector_result=reflector_result,
    )


@app.post("/api/benchmark/run", response_model=BenchmarkRunResponse)
async def run_benchmark(payload: BenchmarkRunRequest) -> BenchmarkRunResponse:
    return await _execute_single(payload=payload)


@app.post("/api/benchmark/run-batch", response_model=BatchBenchmarkRunResponse)
async def run_batch_benchmark(payload: BatchBenchmarkRunRequest) -> BatchBenchmarkRunResponse:
    run_id = str(uuid4())
    case_results: list[BenchmarkRunResponse] = []

    for case in payload.cases:
        single_req = BenchmarkRunRequest(
            config=payload.config,
            session_id=case.session_id,
            user_id=payload.user_id,
            input_text=case.input_text,
            expected_facts=case.expected_facts,
            retrieval=payload.retrieval,
        )
        result = await _execute_single(payload=single_req, run_id=run_id)
        case_results.append(result)

    count = max(len(case_results), 1)
    avg_precision = sum(r.eval_result.metrics.precision for r in case_results) / count
    avg_faithfulness = sum(r.eval_result.metrics.faithfulness for r in case_results) / count
    avg_info_loss = sum(r.eval_result.metrics.info_loss for r in case_results) / count

    avg_metrics = EvalMetrics(
        precision=avg_precision,
        faithfulness=avg_faithfulness,
        info_loss=avg_info_loss,
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["index", "precision", "faithfulness", "info_loss", "judge_rationale"])
    for idx, item in enumerate(case_results, start=1):
        writer.writerow(
            [
                idx,
                f"{item.eval_result.metrics.precision:.4f}",
                f"{item.eval_result.metrics.faithfulness:.4f}",
                f"{item.eval_result.metrics.info_loss:.4f}",
                item.eval_result.judge_rationale,
            ]
        )
    writer.writerow([
        "avg",
        f"{avg_precision:.4f}",
        f"{avg_faithfulness:.4f}",
        f"{avg_info_loss:.4f}",
        "",
    ])

    return BatchBenchmarkRunResponse(
        run_id=run_id,
        case_results=case_results,
        avg_metrics=avg_metrics,
        csv_report=output.getvalue(),
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "MemArena backend is running"}
