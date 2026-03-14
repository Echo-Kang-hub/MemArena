from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# 统一的枚举定义，保证前后端配置与后端分发时不会出现魔法字符串
class ProcessorType(str, Enum):
    raw_logger = "RawLogger"
    summarizer = "Summarizer"
    entity_extractor = "EntityExtractor"


class EngineType(str, Enum):
    vector_engine = "VectorEngine"
    graph_engine = "GraphEngine"
    relational_engine = "RelationalEngine"


class AssemblerType(str, Enum):
    system_injector = "SystemInjector"
    xml_tagging = "XMLTagging"
    timeline_rollover = "TimelineRollover"
    reverse_timeline = "ReverseTimeline"
    ranked_pruning = "RankedPruning"
    reasoning_chain = "ReasoningChain"


class ReflectorType(str, Enum):
    none = "None"
    generative_reflection = "GenerativeReflection"
    conflict_resolver = "ConflictResolver"
    consolidator = "Consolidator"
    conflict_consolidator = "ConflictConsolidator"
    decay_filter = "DecayFilter"
    insight_linker = "InsightLinker"
    abstraction_reflector = "AbstractionReflector"


class ShortTermMemoryMode(str, Enum):
    none = "None"
    sliding_window = "SlidingWindow"
    token_buffer = "TokenBuffer"
    rolling_summary = "RollingSummary"
    working_memory_blackboard = "WorkingMemoryBlackboard"


class ProviderType(str, Enum):
    api = "api"
    ollama = "ollama"
    local = "local"


class SummarizerMethod(str, Enum):
    llm = "llm"
    kmeans = "kmeans"


class EntityExtractorMethod(str, Enum):
    llm_triple = "llm_triple"
    llm_attribute = "llm_attribute"
    spacy_llm_triple = "spacy_llm_triple"
    spacy_llm_attribute = "spacy_llm_attribute"
    mem0_user_facts = "mem0_user_facts"
    mem0_agent_facts = "mem0_agent_facts"
    mem0_dual_facts = "mem0_dual_facts"


class ReflectorLLMMode(str, Enum):
    heuristic = "Heuristic"
    llm = "LLM"
    llm_with_fallback = "LLMWithFallback"


# 原始输入（单条对话）
class RawConversationInput(BaseModel):
    session_id: str = Field(..., description="会话唯一标识")
    user_id: str = Field(default="anonymous", description="用户标识")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="输入时间")
    message: str = Field(..., description="用户输入文本")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加信息")


# 处理器输出：交给 Memory Engine 的标准写入单元
class MemoryChunk(BaseModel):
    chunk_id: str
    session_id: str
    content: str
    tags: list[str] = Field(default_factory=list)
    score_hint: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessorOutput(BaseModel):
    source: ProcessorType
    chunks: list[MemoryChunk]


# 记忆引擎写入/检索输入输出协议
class EngineSaveRequest(BaseModel):
    source: ProcessorType
    chunks: list[MemoryChunk]


class EngineSaveResult(BaseModel):
    engine: EngineType
    saved_count: int
    message: str


class EngineSearchRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 5
    filters: dict[str, Any] = Field(default_factory=dict)


class MemoryHit(BaseModel):
    chunk_id: str
    content: str
    relevance: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class EngineSearchResult(BaseModel):
    engine: EngineType
    hits: list[MemoryHit]


# 上下文组装输入输出协议
class AssembleRequest(BaseModel):
    user_query: str
    memory_hits: list[MemoryHit]
    system_prompt: str = "你是一个可靠的 AI 助手。"
    token_budget: int | None = None


class AssembleResult(BaseModel):
    assembler: AssemblerType
    prompt: str
    preview_blocks: list[dict[str, Any]] = Field(default_factory=list)


# 异步反思输入输出协议
class ReflectRequest(BaseModel):
    session_id: str
    latest_query: str
    memory_hits: list[MemoryHit]


class ReflectResult(BaseModel):
    reflector: ReflectorType
    insights: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


# 评估协议
class EvalCase(BaseModel):
    input_text: str
    expected_facts: list[str] = Field(default_factory=list)


class EvalRequest(BaseModel):
    run_id: str
    assembled_prompt: str
    generated_response: str = ""
    retrieved: list[MemoryHit]
    expected_facts: list[str] = Field(default_factory=list)


class EvalMetrics(BaseModel):
    precision: float
    faithfulness: float
    info_loss: float
    recall_at_k: float | None = None
    qa_accuracy: float | None = None
    qa_f1: float | None = None
    consistency_score: float | None = None
    rejection_rate: float | None = None
    rejection_correctness_unknown: float | None = None
    convergence_speed: float | None = None
    context_distraction: float | None = None


class EvalResult(BaseModel):
    bench: str = "LLMJudgeBench"
    metrics: EvalMetrics
    judge_rationale: str
    raw_judge_output: str | None = None


# 前端发起运行时的统一配置与响应
class BenchmarkConfig(BaseModel):
    processor: ProcessorType
    engine: EngineType
    assembler: AssemblerType
    reflector: ReflectorType = ReflectorType.none
    llm_provider: ProviderType = ProviderType.api
    chat_llm_provider: ProviderType | None = None
    judge_llm_provider: ProviderType | None = None
    summarizer_llm_provider: ProviderType | None = None
    entity_llm_provider: ProviderType | None = None
    reflector_llm_provider: ProviderType | None = None
    embedding_provider: ProviderType
    summarizer_method: SummarizerMethod = SummarizerMethod.llm
    entity_extractor_method: EntityExtractorMethod = EntityExtractorMethod.llm_triple
    compute_device: str = Field(default="cpu", pattern="^(cpu|cuda)$")


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=5, ge=1, le=50)
    min_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    collection_name: str = Field(default="memarena_memory")
    similarity_strategy: str = Field(default="inverse_distance")
    keyword_rerank: bool = False
    max_context_tokens: int | None = Field(default=None, ge=64, le=8192)
    reasoning_hops: int = Field(default=1, ge=1, le=3)
    short_term_mode: ShortTermMemoryMode = ShortTermMemoryMode.none
    stm_window_turns: int = Field(default=5, ge=1, le=30)
    stm_token_budget: int = Field(default=2000, ge=128, le=16000)
    stm_summary_keep_recent_turns: int = Field(default=4, ge=1, le=12)
    reflector_auto_writeback: bool = False
    reflector_writeback_min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    reflector_llm_mode: ReflectorLLMMode = ReflectorLLMMode.llm_with_fallback


class BenchmarkRunRequest(BaseModel):
    config: BenchmarkConfig
    session_id: str = "demo-session"
    user_id: str = "demo-user"
    input_text: str
    assistant_message: str | None = None
    expected_facts: list[str] = Field(default_factory=list)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


class BenchmarkRunResponse(BaseModel):
    run_id: str
    config: BenchmarkConfig
    save_result: EngineSaveResult
    search_result: EngineSearchResult
    assemble_result: AssembleResult
    generated_response: str = ""
    eval_result: EvalResult
    reflector_result: ReflectResult | None = None


class BatchCase(BaseModel):
    case_id: str
    input_text: str
    expected_facts: list[str] = Field(default_factory=list)
    session_id: str = "batch-session"


class BatchBenchmarkRunRequest(BaseModel):
    config: BenchmarkConfig
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    user_id: str = "batch-user"
    cases: list[BatchCase]
    isolate_sessions: bool = True
    max_concurrency: int = Field(default=1, ge=1, le=32)


class BatchBenchmarkRunResponse(BaseModel):
    run_id: str
    case_results: list[BenchmarkRunResponse]
    avg_metrics: EvalMetrics
    csv_report: str


class DatasetRunRequest(BaseModel):
    dataset_name: str
    config: BenchmarkConfig
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    user_id: str = "dataset-user"
    sample_size: int = Field(default=10, ge=1)
    start_index: int = Field(default=0, ge=0)
    isolate_sessions: bool = True
    max_concurrency: int = Field(default=1, ge=1, le=32)


class AsyncRunStartResponse(BaseModel):
    run_id: str
    status: str


class AsyncRunStatusResponse(BaseModel):
    run_id: str
    status: str
    completed: int
    total: int
    message: str = ""
    result: BatchBenchmarkRunResponse | None = None
