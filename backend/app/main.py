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
    AssemblerType,
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
    EngineType,
    EntityExtractorMethod,
    EvalRequest,
    MemoryChunk,
    ProcessorType,
    RawConversationInput,
    ReflectRequest,
    ReflectorLLMMode,
    ReflectorType,
    ShortTermMemoryMode,
    GlobalModelConfigResponse,
    GlobalModelConfigUpdateRequest,
    GlobalModelConnectivityTestRequest,
    GlobalModelConnectivityTestResponse,
    GlobalModelConnectivityItem,
    ProviderType,
)
from app.services.request_audit import (
    query_audit_events,
    reset_audit_run_id,
    set_audit_run_id,
    write_audit_event,
)
from app.services.dataset_loader import list_datasets, load_dataset_cases
from app.services.global_model_config import env_file_path, read_global_model_config, save_global_model_config
from app.services.short_term_memory import ShortTermMemoryManager
from app.registry import build_assembler, build_engine, build_processor, build_reflector

app = FastAPI(title="MemArena Backend", version="0.1.0")

# 轻量内存任务状态表：用于前端轮询批量任务进度
RUN_STATES: dict[str, dict] = {}
STM_MANAGER = ShortTermMemoryManager()


def _preview_text(value: str, limit: int = 280) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + " ..."


def _trace_chunks(chunks: list[MemoryChunk], limit: int = 12) -> list[dict]:
    out: list[dict] = []
    for chunk in chunks[:limit]:
        meta = chunk.metadata or {}
        out.append(
            {
                "chunk_id": chunk.chunk_id,
                "role": str(meta.get("role", "unknown")),
                "tags": list(chunk.tags or []),
                "content_preview": _preview_text(chunk.content, 220),
                "metadata_keys": sorted(list(meta.keys())),
            }
        )
    return out


def _trace_hits(hits, limit: int = 12) -> list[dict]:
    out: list[dict] = []
    for hit in hits[:limit]:
        meta = hit.metadata or {}
        out.append(
            {
                "chunk_id": hit.chunk_id,
                "relevance": float(hit.relevance),
                "role": str(meta.get("role", "unknown")),
                "stm": bool(meta.get("stm", False)),
                "content_preview": _preview_text(hit.content, 220),
                "metadata_keys": sorted(list(meta.keys())),
            }
        )
    return out


def _is_llm_failure_text(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return True
    return lowered.startswith("llm provider call failed:")


def _test_llm_client(module: str, client) -> GlobalModelConnectivityItem:
    purpose = f"connectivity_test_{module}"
    output = client.generate(
        "只输出 OK。",
        system_prompt="你是连通性测试助手，只返回 OK。",
        purpose=purpose,
    )
    output_preview = _preview_text(output, 180)
    ok = not _is_llm_failure_text(output)
    return GlobalModelConnectivityItem(
        module=module,
        kind="llm",
        provider=str(client.provider),
        model=str(client.model),
        endpoint=str(client.endpoint),
        ok=ok,
        note="",
        error="" if ok else output_preview,
        output_preview=output_preview,
    )


def _test_embedding_client(client) -> GlobalModelConnectivityItem:
    vec = client.embed("connectivity test")
    dim = len(vec) if isinstance(vec, list) else 0
    ok = dim > 0
    return GlobalModelConnectivityItem(
        module="embedding",
        kind="embedding",
        provider=str(client.provider),
        model=str(client.model),
        endpoint=str(client.endpoint),
        ok=ok,
        note=(f"dim={dim}" if ok else "empty embedding"),
        error=("" if ok else "embedding output is empty"),
        output_preview=(f"vector_dim={dim}" if ok else ""),
    )


def _auto_collection_name(base_name: str, embedding_provider: str, embedding_model: str) -> str:
    compact_model = re.sub(r"[^a-z0-9]+", "_", embedding_model.lower()).strip("_")
    compact_model = compact_model[:48] or "default"
    compact_provider = re.sub(r"[^a-z0-9]+", "_", embedding_provider.lower()).strip("_") or "provider"
    return f"{base_name}_{compact_provider}_{compact_model}"


def _validate_processor_engine_mapping(payload: BenchmarkRunRequest) -> None:
    if payload.config.processor != ProcessorType.entity_extractor:
        return

    method = payload.config.entity_extractor_method
    if method in {EntityExtractorMethod.llm_triple, EntityExtractorMethod.spacy_llm_triple}:
        if payload.config.engine != EngineType.graph_engine:
            raise ValueError("EntityExtractor triple modes require engine=GraphEngine")
        return

    if method in {
        EntityExtractorMethod.llm_attribute,
        EntityExtractorMethod.spacy_llm_attribute,
        EntityExtractorMethod.mem0_user_facts,
        EntityExtractorMethod.mem0_agent_facts,
        EntityExtractorMethod.mem0_dual_facts,
    }:
        if payload.config.engine != EngineType.relational_engine:
            raise ValueError("EntityExtractor attribute/mem0 modes require engine=RelationalEngine")
        return

    if payload.config.engine != EngineType.relational_engine:
        raise ValueError("EntityExtractor attribute modes require engine=RelationalEngine")


def _merge_memory_hits(
    ltm_hits,
    stm_hits,
    top_k: int,
):
    merged = []
    seen_content: set[str] = set()

    for hit in [*stm_hits, *ltm_hits]:
        norm_content = " ".join(str(hit.content).split()).strip().lower()
        if not norm_content:
            norm_content = str(hit.chunk_id)
        if norm_content in seen_content:
            continue
        seen_content.add(norm_content)
        merged.append(hit)

    merged.sort(key=lambda x: float(x.relevance), reverse=True)
    return merged[: max(1, int(top_k))]


def _extract_writeback_updates(reflector_result, min_confidence: float) -> list[dict]:
    if reflector_result is None:
        return []
    stats = reflector_result.stats or {}
    updates = stats.get("proposed_resolutions", [])
    if not isinstance(updates, list):
        return []
    out: list[dict] = []
    for item in updates:
        if not isinstance(item, dict):
            continue
        confidence = float(item.get("confidence", 0.0) or 0.0)
        if confidence < min_confidence:
            continue
        out.append(item)
    return out


def _apply_reflector_writeback(
    *,
    engine,
    session_id: str,
    reflector_result,
    min_confidence: float,
) -> int:
    updates = _extract_writeback_updates(reflector_result, min_confidence=min_confidence)
    if not updates:
        return 0

    control_chunk = MemoryChunk(
        chunk_id=f"{session_id}-reflect-writeback-{uuid4().hex[:8]}",
        session_id=session_id,
        content=(
            "Reflector auto writeback control chunk. "
            "It suppresses stale conflicting values during retrieval ranking."
        ),
        tags=["reflector", "writeback", "control"],
        metadata={
            "memory_control_chunk": True,
            "reflector": str(reflector_result.reflector.value),
            "resolution_updates": updates,
            "min_confidence": min_confidence,
        },
    )
    engine.save(EngineSaveRequest(source=ProcessorType.raw_logger, chunks=[control_chunk]))
    return len(updates)

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
        "summarizer_methods": ["llm", "kmeans"],
        "entity_extractor_methods": [item.value for item in EntityExtractorMethod],
        "engines": ["VectorEngine", "GraphEngine", "RelationalEngine"],
        "assemblers": [item.value for item in AssemblerType],
        "reflectors": [item.value for item in ReflectorType],
        "providers": ["api", "ollama", "local"],
        "compute_devices": ["cpu", "cuda"],
        "max_concurrency_limit": 32,
        "similarity_strategies": ["inverse_distance", "exp_decay", "linear"],
        "short_term_modes": [item.value for item in ShortTermMemoryMode],
        "reflector_llm_modes": [item.value for item in ReflectorLLMMode],
        "reflector_writeback": {
            "supported": True,
            "default_auto": False,
            "default_min_confidence": 0.75,
        },
    }


@app.get("/api/datasets")
def datasets() -> dict:
    return {"datasets": list_datasets()}


@app.get("/api/config/global-models", response_model=GlobalModelConfigResponse)
def get_global_model_config() -> GlobalModelConfigResponse:
    return GlobalModelConfigResponse(config=read_global_model_config(), env_file=str(env_file_path()))


@app.post("/api/config/global-models", response_model=GlobalModelConfigResponse)
def update_global_model_config(payload: GlobalModelConfigUpdateRequest) -> GlobalModelConfigResponse:
    save_global_model_config(payload.config)
    return GlobalModelConfigResponse(config=read_global_model_config(), env_file=str(env_file_path()))


@app.post("/api/config/global-models/test", response_model=GlobalModelConnectivityTestResponse)
def test_global_model_connectivity(payload: GlobalModelConnectivityTestRequest) -> GlobalModelConnectivityTestResponse:
    modules = [str(m).strip().lower() for m in payload.modules if str(m).strip()]
    if not modules:
        modules = ["chat", "judge", "summarizer", "entity", "reflector", "embedding"]

    results: list[GlobalModelConnectivityItem] = []

    builders = {
        "chat": ProviderFactory.build_chat_llm,
        "judge": ProviderFactory.build_judge_llm,
        "summarizer": ProviderFactory.build_summarizer_llm,
        "entity": ProviderFactory.build_entity_llm,
        "reflector": ProviderFactory.build_reflector_llm,
    }

    for module in modules:
        if module in builders:
            client = builders[module](None, settings.local_infer_device)
            results.append(_test_llm_client(module, client))
            continue

        if module == "embedding":
            emb_provider = ProviderType(str(settings.embedding_provider or settings.default_embedding_provider))
            emb_client = ProviderFactory.build_embedding(emb_provider, settings.local_infer_device)
            results.append(_test_embedding_client(emb_client))
            continue

        results.append(
            GlobalModelConnectivityItem(
                module=module,
                kind="unknown",
                provider="",
                model="",
                endpoint="",
                ok=False,
                note="",
                error="unsupported module",
                output_preview="",
            )
        )

    passed = sum(1 for item in results if item.ok)
    return GlobalModelConnectivityTestResponse(
        tested_modules=modules,
        passed=passed,
        total=len(results),
        results=results,
    )


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
                "summarizer": str(payload.config.summarizer_llm_provider or payload.config.llm_provider),
                "entity": str(payload.config.entity_llm_provider or payload.config.llm_provider),
                "reflector": str(payload.config.reflector_llm_provider or payload.config.llm_provider),
                "embedding": str(payload.config.embedding_provider),
                "compute_device": payload.config.compute_device,
            },
            "retrieval": payload.retrieval.model_dump(),
        },
    )

    try:
        # 初始化模型工厂，用于引擎检索和 LLM-as-a-Judge
        _validate_processor_engine_mapping(payload)

        chat_provider = payload.config.chat_llm_provider or payload.config.llm_provider
        judge_provider = payload.config.judge_llm_provider or payload.config.llm_provider
        summarizer_provider = payload.config.summarizer_llm_provider or payload.config.llm_provider
        entity_provider = payload.config.entity_llm_provider or payload.config.llm_provider
        reflector_provider = payload.config.reflector_llm_provider or payload.config.llm_provider
        compute_device = payload.config.compute_device

        llm_client = ProviderFactory.build_chat_llm(chat_provider, compute_device)
        judge_llm_client = ProviderFactory.build_judge_llm(judge_provider, compute_device)
        summarizer_llm_client = ProviderFactory.build_summarizer_llm(summarizer_provider, compute_device)
        entity_llm_client = ProviderFactory.build_entity_llm(entity_provider, compute_device)
        reflector_llm_client = ProviderFactory.build_reflector_llm(reflector_provider, compute_device)
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

        processor = build_processor(
            payload.config.processor,
            summarizer_method=payload.config.summarizer_method,
            entity_extractor_method=payload.config.entity_extractor_method,
            summarizer_llm_client=summarizer_llm_client,
            entity_llm_client=entity_llm_client,
        )
        engine = build_engine(
            payload.config.engine,
            embedding_client=embedding_client,
            collection_name=retrieval_cfg.collection_name,
        )
        assembler = build_assembler(payload.config.assembler)
        reflector = build_reflector(
            payload.config.reflector,
            reflection_llm_client=reflector_llm_client,
            llm_mode=retrieval_cfg.reflector_llm_mode,
        )
        bench = LLMJudgeBench(llm_client=judge_llm_client)

        raw_input = RawConversationInput(
            session_id=payload.session_id,
            user_id=payload.user_id,
            message=payload.input_text,
            metadata={
                "role": "user",
                "assistant_message": payload.assistant_message or "",
                "llm_preview": f"{llm_client.provider}:{llm_client.model}",
                "embedding_preview": embedding_client.embed(payload.input_text),
            },
        )

        STM_MANAGER.ingest(
            session_id=payload.session_id,
            text=payload.input_text,
            mode=retrieval_cfg.short_term_mode,
            summary_keep_recent_turns=retrieval_cfg.stm_summary_keep_recent_turns,
            role="user",
            llm_client=summarizer_llm_client,
        )

        processor_output = processor.process(raw_input)
        save_result = engine.save(EngineSaveRequest(source=processor_output.source, chunks=processor_output.chunks))

        search_request = EngineSearchRequest(
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

        if (
            payload.config.assembler == AssemblerType.reasoning_chain
            and payload.config.engine == EngineType.graph_engine
            and hasattr(engine, "search_with_reasoning")
        ):
            reasoning_hops = int(retrieval_cfg.reasoning_hops)
            search_result = engine.search_with_reasoning(
                search_request,
                hops=reasoning_hops,
                max_chains=settings.graph_reasoning_max_chains,
            )
        else:
            search_result = engine.search(search_request)

        stm_hits = STM_MANAGER.retrieve(
            session_id=payload.session_id,
            query=payload.input_text,
            mode=retrieval_cfg.short_term_mode,
            top_k=retrieval_cfg.top_k,
            window_turns=retrieval_cfg.stm_window_turns,
            token_budget=retrieval_cfg.stm_token_budget,
            summary_keep_recent_turns=retrieval_cfg.stm_summary_keep_recent_turns,
        )
        effective_hits = _merge_memory_hits(search_result.hits, stm_hits, retrieval_cfg.top_k)
        search_result = search_result.model_copy(update={"hits": effective_hits})

        assemble_result = assembler.assemble(
            AssembleRequest(
                user_query=payload.input_text,
                memory_hits=search_result.hits,
                token_budget=retrieval_cfg.max_context_tokens or settings.context_token_budget,
            )
        )

        # 显式执行一次聊天模型生成，确保 chat LLM 与 judge LLM 真实解耦并可观测。
        agent_response = llm_client.generate(
            assemble_result.prompt,
            system_prompt="你是一个可靠的 AI 助手。",
            purpose="chat_response",
        )

        STM_MANAGER.ingest(
            session_id=payload.session_id,
            text=agent_response,
            mode=retrieval_cfg.short_term_mode,
            summary_keep_recent_turns=retrieval_cfg.stm_summary_keep_recent_turns,
            role="assistant",
            llm_client=summarizer_llm_client,
        )

        # 将助手回复按同一 Processor 规则单独写入记忆，供后续轮次检索。
        assistant_input = RawConversationInput(
            session_id=payload.session_id,
            user_id=payload.user_id,
            message="",
            metadata={
                "role": "assistant",
                "assistant_message": agent_response,
                "llm_preview": f"{llm_client.provider}:{llm_client.model}",
            },
        )
        assistant_output = processor.process(assistant_input)
        assistant_chunks = [c for c in assistant_output.chunks if str((c.metadata or {}).get("role", "")).strip().lower() == "assistant"]
        if assistant_chunks:
            engine.save(EngineSaveRequest(source=assistant_output.source, chunks=assistant_chunks))

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
        writeback_updates = 0
        if reflector is not None:
            reflector_result = await reflector.reflect(
                ReflectRequest(
                    session_id=payload.session_id,
                    latest_query=payload.input_text,
                    memory_hits=search_result.hits,
                )
            )
            if retrieval_cfg.reflector_auto_writeback:
                writeback_updates = _apply_reflector_writeback(
                    engine=engine,
                    session_id=payload.session_id,
                    reflector_result=reflector_result,
                    min_confidence=float(retrieval_cfg.reflector_writeback_min_confidence),
                )

        audit_events = query_audit_events(run_id=cur_run_id, limit=600)
        llm_calls: list[dict] = []
        for evt in audit_events:
            if str(evt.get("event_type", "")) != "llm_generate":
                continue
            llm_calls.append(
                {
                    "ts": str(evt.get("ts", "")),
                    "purpose": str(evt.get("purpose", "")),
                    "provider": str(evt.get("provider", "")),
                    "model": str(evt.get("model", "")),
                    "ok": bool(evt.get("ok", False)),
                    "duration_ms": float(evt.get("duration_ms", 0.0) or 0.0),
                    "prompt_length": int(evt.get("prompt_length", 0) or 0),
                    "system_prompt_preview": str(evt.get("system_prompt_preview", "")),
                    "prompt_preview": str(evt.get("prompt_preview", "")),
                    "response_preview": str(evt.get("response_preview", "")),
                    "error": str(evt.get("error", "")),
                    "note": str(evt.get("note", "")),
                }
            )

        module_trace = {
            "config": payload.config.model_dump(),
            "processor": {
                "input": {
                    "session_id": payload.session_id,
                    "user_id": payload.user_id,
                    "message": payload.input_text,
                    "assistant_message": payload.assistant_message or "",
                    "processor": payload.config.processor.value,
                },
                "output": {
                    "source": processor_output.source.value,
                    "chunk_count": len(processor_output.chunks),
                    "chunks": _trace_chunks(processor_output.chunks),
                },
            },
            "engine_save": {
                "input": {
                    "source": processor_output.source.value,
                    "chunk_count": len(processor_output.chunks),
                    "chunks": _trace_chunks(processor_output.chunks),
                },
                "output": save_result.model_dump(),
            },
            "engine_search": {
                "input": search_request.model_dump(),
                "output": {
                    "engine": search_result.engine.value,
                    "hit_count": len(search_result.hits),
                    "hits": _trace_hits(search_result.hits),
                },
            },
            "assembler": {
                "input": {
                    "user_query": payload.input_text,
                    "memory_hit_count": len(search_result.hits),
                    "token_budget": retrieval_cfg.max_context_tokens or settings.context_token_budget,
                },
                "output": {
                    "assembler": assemble_result.assembler.value,
                    "prompt_preview": _preview_text(assemble_result.prompt, 1000),
                    "preview_blocks_count": len(assemble_result.preview_blocks),
                },
            },
            "chat_generation": {
                "input": {
                    "system_prompt": "你是一个可靠的 AI 助手。",
                    "prompt_preview": _preview_text(assemble_result.prompt, 800),
                },
                "output": {
                    "response_preview": _preview_text(agent_response, 500),
                },
            },
            "assistant_memory_writeback": {
                "input": {
                    "assistant_message_preview": _preview_text(agent_response, 300),
                },
                "output": {
                    "chunk_count": len(assistant_chunks),
                    "chunks": _trace_chunks(assistant_chunks),
                },
            },
            "evaluation": {
                "input": {
                    "expected_facts": payload.expected_facts,
                    "retrieved_count": len(search_result.hits),
                    "response_preview": _preview_text(agent_response, 220),
                },
                "output": {
                    "metrics": eval_result.metrics.model_dump(),
                    "judge_rationale_preview": _preview_text(eval_result.judge_rationale, 500),
                },
            },
            "reflector": {
                "input": {
                    "enabled": reflector is not None,
                    "reflector": payload.config.reflector.value,
                    "llm_mode": retrieval_cfg.reflector_llm_mode.value,
                    "memory_hit_count": len(search_result.hits),
                    "latest_query": payload.input_text,
                },
                "output": reflector_result.model_dump() if reflector_result is not None else {"reflector": "None", "insights": [], "stats": {}},
            },
            "stm": {
                "input": {
                    "mode": retrieval_cfg.short_term_mode.value,
                    "window_turns": retrieval_cfg.stm_window_turns,
                    "token_budget": retrieval_cfg.stm_token_budget,
                    "summary_keep_recent_turns": retrieval_cfg.stm_summary_keep_recent_turns,
                },
                "output": {
                    "stm_hit_count": len(stm_hits),
                    "stm_hits": _trace_hits(stm_hits),
                },
            },
            "llm_calls": llm_calls,
        }

        response = BenchmarkRunResponse(
            run_id=cur_run_id,
            config=payload.config,
            save_result=save_result,
            search_result=search_result,
            assemble_result=assemble_result,
            generated_response=agent_response,
            eval_result=eval_result,
            reflector_result=reflector_result,
            module_trace=module_trace,
        )
        write_audit_event(
            "run_single_success",
            {
                "run_id": cur_run_id,
                "session_id": payload.session_id,
                "hits": len(search_result.hits),
                "stm_mode": retrieval_cfg.short_term_mode.value,
                "stm_hits": len(stm_hits),
                "reflector_writeback_enabled": retrieval_cfg.reflector_auto_writeback,
                "reflector_writeback_updates": writeback_updates,
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
    avg_convergence_speed = avg_optional_metric("convergence_speed")
    avg_context_distraction = avg_optional_metric("context_distraction")

    def extract_reasoning_stats(item: BenchmarkRunResponse) -> tuple[float | None, float | None, float | None, float | None]:
        hits = item.search_result.hits
        if not hits:
            return None, None, None, None
        metadata = hits[0].metadata or {}
        details = metadata.get("reasoning_chain_details", [])
        if not isinstance(details, list) or not details:
            chains = metadata.get("reasoning_chains", [])
            if isinstance(chains, list) and chains:
                return float(len(chains)), None, None, None
            return None, None, None, None

        chain_count = float(len(details))
        priorities: list[float] = []
        hops: list[float] = []
        seed_touch_cnt = 0
        for d in details:
            if not isinstance(d, dict):
                continue
            if "priority" in d:
                try:
                    priorities.append(float(d.get("priority")))
                except Exception:
                    pass
            if "hop" in d:
                try:
                    hops.append(float(d.get("hop")))
                except Exception:
                    pass
            if bool(d.get("seed_touch", False)):
                seed_touch_cnt += 1

        avg_priority = (sum(priorities) / len(priorities)) if priorities else None
        avg_hop = (sum(hops) / len(hops)) if hops else None
        seed_touch_ratio = (seed_touch_cnt / len(details)) if details else None
        return chain_count, avg_priority, avg_hop, seed_touch_ratio

    reasoning_rows = [extract_reasoning_stats(r) for r in case_results]

    def avg_reasoning(idx: int) -> float | None:
        vals = [row[idx] for row in reasoning_rows if row[idx] is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    avg_reasoning_chain_count = avg_reasoning(0)
    avg_reasoning_priority = avg_reasoning(1)
    avg_reasoning_hop = avg_reasoning(2)
    avg_reasoning_seed_touch_ratio = avg_reasoning(3)

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
        convergence_speed=avg_convergence_speed,
        context_distraction=avg_context_distraction,
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
        "convergence_speed",
        "context_distraction",
        "reasoning_chain_count",
        "reasoning_avg_priority",
        "reasoning_avg_hop",
        "reasoning_seed_touch_ratio",
        "judge_rationale",
    ])
    for idx, item in enumerate(case_results, start=1):
        chain_count, avg_priority, avg_hop, seed_touch_ratio = extract_reasoning_stats(item)
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
                "" if item.eval_result.metrics.convergence_speed is None else f"{item.eval_result.metrics.convergence_speed:.4f}",
                "" if item.eval_result.metrics.context_distraction is None else f"{item.eval_result.metrics.context_distraction:.4f}",
                "" if chain_count is None else f"{chain_count:.0f}",
                "" if avg_priority is None else f"{avg_priority:.4f}",
                "" if avg_hop is None else f"{avg_hop:.4f}",
                "" if seed_touch_ratio is None else f"{seed_touch_ratio:.4f}",
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
        "" if avg_convergence_speed is None else f"{avg_convergence_speed:.4f}",
        "" if avg_context_distraction is None else f"{avg_context_distraction:.4f}",
        "" if avg_reasoning_chain_count is None else f"{avg_reasoning_chain_count:.4f}",
        "" if avg_reasoning_priority is None else f"{avg_reasoning_priority:.4f}",
        "" if avg_reasoning_hop is None else f"{avg_reasoning_hop:.4f}",
        "" if avg_reasoning_seed_touch_ratio is None else f"{avg_reasoning_seed_touch_ratio:.4f}",
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
