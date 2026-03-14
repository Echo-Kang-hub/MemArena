export type ProviderType = 'api' | 'ollama' | 'local';
export type ComputeDevice = 'cpu' | 'cuda';
export type SummarizerMethod = 'llm' | 'kmeans';
export type EntityExtractorMethod =
  | 'llm_triple'
  | 'llm_attribute'
  | 'spacy_llm_triple'
  | 'spacy_llm_attribute'
  | 'mem0_user_facts'
  | 'mem0_agent_facts'
  | 'mem0_dual_facts';
export type ReflectorLLMMode = 'Heuristic' | 'LLM' | 'LLMWithFallback';
export type ShortTermMemoryMode =
  | 'None'
  | 'SlidingWindow'
  | 'TokenBuffer'
  | 'RollingSummary'
  | 'WorkingMemoryBlackboard';

export interface BenchmarkConfig {
  processor: 'RawLogger' | 'Summarizer' | 'EntityExtractor';
  engine: 'VectorEngine' | 'GraphEngine' | 'RelationalEngine';
  assembler:
    | 'SystemInjector'
    | 'XMLTagging'
    | 'TimelineRollover'
    | 'ReverseTimeline'
    | 'RankedPruning'
    | 'ReasoningChain';
  reflector:
    | 'None'
    | 'GenerativeReflection'
    | 'ConflictResolver'
    | 'Consolidator'
    | 'ConflictConsolidator'
    | 'DecayFilter'
    | 'InsightLinker'
    | 'AbstractionReflector';
  llm_provider: ProviderType;
  chat_llm_provider?: ProviderType;
  judge_llm_provider?: ProviderType;
  summarizer_llm_provider?: ProviderType;
  entity_llm_provider?: ProviderType;
  reflector_llm_provider?: ProviderType;
  embedding_provider: ProviderType;
  summarizer_method?: SummarizerMethod;
  entity_extractor_method?: EntityExtractorMethod;
  compute_device?: ComputeDevice;
}

export interface BenchmarkRunRequest {
  config: BenchmarkConfig;
  session_id: string;
  user_id: string;
  input_text: string;
  assistant_message?: string;
  expected_facts: string[];
  retrieval: {
    top_k: number;
    min_relevance: number;
    collection_name: string;
    similarity_strategy: 'inverse_distance' | 'exp_decay' | 'linear';
    keyword_rerank: boolean;
    max_context_tokens?: number | null;
    reasoning_hops?: number;
    short_term_mode?: ShortTermMemoryMode;
    stm_window_turns?: number;
    stm_token_budget?: number;
    stm_summary_keep_recent_turns?: number;
    reflector_auto_writeback?: boolean;
    reflector_writeback_min_confidence?: number;
    reflector_llm_mode?: ReflectorLLMMode;
  };
}

export interface EvalMetrics {
  precision: number;
  faithfulness: number;
  info_loss: number;
  recall_at_k?: number | null;
  qa_accuracy?: number | null;
  qa_f1?: number | null;
  consistency_score?: number | null;
  rejection_rate?: number | null;
  rejection_correctness_unknown?: number | null;
  convergence_speed?: number | null;
  context_distraction?: number | null;
}

export interface BenchmarkRunResponse {
  run_id: string;
  generated_response?: string;
  assemble_result: {
    prompt: string;
    preview_blocks: Array<{ role: string; text: string }>;
  };
  eval_result: {
    metrics: EvalMetrics;
    judge_rationale: string;
    raw_judge_output?: string | null;
  };
  module_trace?: Record<string, unknown>;
  search_result: {
    hits: Array<{
      content: string;
      relevance: number;
      metadata?: {
        role?: 'user' | 'assistant' | 'unknown';
        stm?: boolean;
        reasoning_chains?: string[];
        reasoning_seed_entities?: string[];
        reasoning_chain_details?: Array<{
          chain: string;
          hop: number;
          seed_touch: boolean;
          lexical_overlap: number;
          priority: number;
        }>;
        [key: string]: unknown;
      };
    }>;
  };
}

export interface BatchCase {
  case_id: string;
  input_text: string;
  expected_facts: string[];
  session_id: string;
}

export interface BatchBenchmarkRunRequest {
  config: BenchmarkConfig;
  retrieval: BenchmarkRunRequest['retrieval'];
  user_id: string;
  cases: BatchCase[];
  isolate_sessions?: boolean;
  max_concurrency?: number;
}

export interface BatchBenchmarkRunResponse {
  run_id: string;
  case_results: BenchmarkRunResponse[];
  avg_metrics: EvalMetrics;
  csv_report: string;
}

export interface DatasetRunRequest {
  dataset_name: string;
  config: BenchmarkConfig;
  retrieval: BenchmarkRunRequest['retrieval'];
  user_id: string;
  sample_size: number;
  start_index: number;
  isolate_sessions?: boolean;
  max_concurrency?: number;
}

export interface DatasetSummary {
  name: string;
  file: string;
  count: number;
}

export interface AsyncRunStartResponse {
  run_id: string;
  status: string;
}

export interface AsyncRunStatusResponse {
  run_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'not_found' | string;
  completed: number;
  total: number;
  message: string;
  result?: BatchBenchmarkRunResponse | null;
}

export interface GlobalModelConfig {
  default_llm_provider: string;
  default_embedding_provider: string;
  chat_llm_provider: string;
  chat_api_base_url: string;
  chat_api_key: string;
  chat_api_model: string;
  chat_ollama_base_url: string;
  chat_ollama_model: string;
  chat_local_model_path: string;
  judge_llm_provider: string;
  judge_api_base_url: string;
  judge_api_key: string;
  judge_api_model: string;
  judge_ollama_base_url: string;
  judge_ollama_model: string;
  judge_local_model_path: string;
  summarizer_llm_provider: string;
  summarizer_api_base_url: string;
  summarizer_api_key: string;
  summarizer_api_model: string;
  summarizer_ollama_base_url: string;
  summarizer_ollama_model: string;
  summarizer_local_model_path: string;
  entity_llm_provider: string;
  entity_api_base_url: string;
  entity_api_key: string;
  entity_api_model: string;
  entity_ollama_base_url: string;
  entity_ollama_model: string;
  entity_local_model_path: string;
  reflector_llm_provider: string;
  reflector_api_base_url: string;
  reflector_api_key: string;
  reflector_api_model: string;
  reflector_ollama_base_url: string;
  reflector_ollama_model: string;
  reflector_local_model_path: string;
  embedding_provider: string;
  embedding_api_base_url: string;
  embedding_api_key: string;
  embedding_api_model: string;
  embedding_ollama_base_url: string;
  embedding_ollama_model: string;
  embedding_local_model_path: string;
  local_infer_device: string;
}

export interface GlobalModelConfigResponse {
  config: GlobalModelConfig;
  env_file: string;
}

export interface GlobalConnectivityItem {
  module: string;
  kind: string;
  provider: string;
  model: string;
  endpoint: string;
  ok: boolean;
  note: string;
  error: string;
  output_preview: string;
}

export interface GlobalConnectivityTestResponse {
  tested_modules: string[];
  passed: number;
  total: number;
  results: GlobalConnectivityItem[];
}
