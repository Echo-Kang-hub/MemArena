export type ProviderType = 'api' | 'ollama' | 'local';
export type ComputeDevice = 'cpu' | 'cuda';

export interface BenchmarkConfig {
  processor: 'RawLogger' | 'Summarizer' | 'EntityExtractor';
  engine: 'VectorEngine' | 'GraphEngine' | 'RelationalEngine';
  assembler: 'SystemInjector' | 'XMLTagging' | 'TimelineRollover';
  reflector: 'None' | 'GenerativeReflection' | 'ConflictResolver';
  llm_provider: ProviderType;
  chat_llm_provider?: ProviderType;
  judge_llm_provider?: ProviderType;
  embedding_provider: ProviderType;
  compute_device?: ComputeDevice;
}

export interface BenchmarkRunRequest {
  config: BenchmarkConfig;
  session_id: string;
  user_id: string;
  input_text: string;
  expected_facts: string[];
  retrieval: {
    top_k: number;
    min_relevance: number;
    collection_name: string;
    similarity_strategy: 'inverse_distance' | 'exp_decay' | 'linear';
    keyword_rerank: boolean;
  };
}

export interface EvalMetrics {
  precision: number;
  faithfulness: number;
  info_loss: number;
}

export interface BenchmarkRunResponse {
  run_id: string;
  assemble_result: {
    prompt: string;
    preview_blocks: Array<{ role: string; text: string }>;
  };
  eval_result: {
    metrics: EvalMetrics;
    judge_rationale: string;
    raw_judge_output?: string | null;
  };
  search_result: {
    hits: Array<{ content: string; relevance: number }>;
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
