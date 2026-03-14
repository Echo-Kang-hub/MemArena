<script setup lang="ts">
import MarkdownIt from 'markdown-it';
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import MetricBars from './components/MetricBars.vue';
import {
  getAuditEventsByRun,
  getAsyncRunStatus,
  getGlobalModelConfig,
  listDatasets,
  runBatchBenchmarkAsync,
  runBenchmarkWithTimeout,
  runDatasetBenchmarkAsync,
  testGlobalModelConnectivity,
  updateGlobalModelConfig,
} from './api/client';
import type {
  BatchBenchmarkRunResponse,
  BenchmarkConfig,
  BenchmarkRunResponse,
  DatasetSummary,
  GlobalConnectivityTestResponse,
  GlobalModelConfig,
  ReflectorLLMMode
} from './types';

const config = ref<BenchmarkConfig>({
  processor: 'RawLogger',
  engine: 'VectorEngine',
  assembler: 'SystemInjector',
  reflector: 'None',
  llm_provider: 'api',
  chat_llm_provider: 'api',
  judge_llm_provider: 'api',
  summarizer_llm_provider: 'api',
  entity_llm_provider: 'api',
  reflector_llm_provider: 'api',
  embedding_provider: 'ollama',
  summarizer_method: 'llm',
  entity_extractor_method: 'llm_triple',
  compute_device: 'cpu'
});

const inputText = ref('我下周要去上海出差，记得提醒我带护照和会议材料。');
const expectedFactsRaw = ref('上海\n护照\n会议材料');
const datasetJson = ref('');
const loading = ref(false);
const error = ref('');
const result = ref<BenchmarkRunResponse | null>(null);
const batchResult = ref<BatchBenchmarkRunResponse | null>(null);
const retrievalTopK = ref(5);
const minRelevance = ref(0);
const collectionName = ref('memarena_memory');
const similarityStrategy = ref<'inverse_distance' | 'exp_decay' | 'linear'>('inverse_distance');
const keywordRerank = ref(false);
const maxContextTokens = ref(1200);
const reasoningHops = ref(1);
const shortTermMode = ref<'None' | 'SlidingWindow' | 'TokenBuffer' | 'RollingSummary' | 'WorkingMemoryBlackboard'>('None');
const stmWindowTurns = ref(5);
const stmTokenBudget = ref(2000);
const stmSummaryKeepRecentTurns = ref(4);
const reflectorAutoWriteback = ref(false);
const reflectorWritebackMinConfidence = ref(0.75);
const reflectorLlmMode = ref<ReflectorLLMMode>('LLMWithFallback');
const datasetCases = ref<Array<{ case_id: string; input_text: string; expected_facts: string[]; session_id: string }>>([]);
const builtinDatasets = ref<DatasetSummary[]>([]);
const builtinDatasetsLoading = ref(false);
const builtinDatasetsMessage = ref('');
const selectedDatasetName = ref('');
const datasetSampleSize = ref(5);
const datasetStartIndex = ref(0);
const isolateSessions = ref(true);
const maxConcurrency = ref(3);
const batchCaseCount = ref(5);
const requestTimeoutMs = ref(120000);
const includeRawJudgeInMarkdown = ref(false);
const progressText = ref('');
const entityHealthState = ref<
  | {
      level: 'warning' | 'info';
      message: string;
      details: string;
      recentFailures: Array<{
        ts: string;
        provider: string;
        model: string;
        purpose: string;
        status: string;
        error: string;
      }>;
    }
  | null
>(null);
const entityDiagnosticCopyFeedback = ref('');
const batchDiagnosticCopyFeedback = ref('');
const elapsedMs = ref(0);
const lastRunDurationMs = ref<number | null>(null);
const globalConfigLoading = ref(false);
const globalConfigSaving = ref(false);
const globalConfigMessage = ref('');
const globalConfigEnvFile = ref('');
const globalConnectivityTesting = ref(false);
const globalConnectivityMessage = ref('');
const globalConnectivity = ref<GlobalConnectivityTestResponse | null>(null);
const globalLastKnownGood = ref<Record<string, { at: string; snapshot: Record<string, string> }>>({});
const globalModelConfig = ref<GlobalModelConfig>({
  default_llm_provider: 'api',
  default_embedding_provider: 'ollama',
  chat_llm_provider: '',
  chat_api_base_url: '',
  chat_api_key: '',
  chat_api_model: '',
  chat_ollama_base_url: '',
  chat_ollama_model: '',
  chat_local_model_path: '',
  judge_llm_provider: '',
  judge_api_base_url: '',
  judge_api_key: '',
  judge_api_model: '',
  judge_ollama_base_url: '',
  judge_ollama_model: '',
  judge_local_model_path: '',
  summarizer_llm_provider: '',
  summarizer_api_base_url: '',
  summarizer_api_key: '',
  summarizer_api_model: '',
  summarizer_ollama_base_url: '',
  summarizer_ollama_model: '',
  summarizer_local_model_path: '',
  entity_llm_provider: '',
  entity_api_base_url: '',
  entity_api_key: '',
  entity_api_model: '',
  entity_ollama_base_url: '',
  entity_ollama_model: '',
  entity_local_model_path: '',
  reflector_llm_provider: '',
  reflector_api_base_url: '',
  reflector_api_key: '',
  reflector_api_model: '',
  reflector_ollama_base_url: '',
  reflector_ollama_model: '',
  reflector_local_model_path: '',
  embedding_provider: '',
  embedding_api_base_url: '',
  embedding_api_key: '',
  embedding_api_model: '',
  embedding_ollama_base_url: '',
  embedding_ollama_model: '',
  embedding_local_model_path: '',
  local_infer_device: 'cpu'
});

let timerHandle: ReturnType<typeof setInterval> | null = null;
let runStartTs = 0;

const processors = ['RawLogger', 'Summarizer', 'EntityExtractor'] as const;
const engines = ['VectorEngine', 'GraphEngine', 'RelationalEngine'] as const;
const assemblers = [
  'SystemInjector',
  'XMLTagging',
  'TimelineRollover',
  'ReverseTimeline',
  'RankedPruning',
  'ReasoningChain'
] as const;
const reflectors = [
  'None',
  'GenerativeReflection',
  'ConflictResolver',
  'Consolidator',
  'ConflictConsolidator',
  'DecayFilter',
  'InsightLinker',
  'AbstractionReflector'
] as const;
const providers = ['api', 'ollama', 'local'] as const;
const summarizerMethods = ['llm', 'kmeans'] as const;
const entityExtractorMethods = [
  'llm_triple',
  'llm_attribute',
  'spacy_llm_triple',
  'spacy_llm_attribute',
  'mem0_user_facts',
  'mem0_agent_facts',
  'mem0_dual_facts'
] as const;
const reflectorLlmModes = ['Heuristic', 'LLM', 'LLMWithFallback'] as const;
const computeDevices = ['cpu', 'cuda'] as const;
const shortTermModes = [
  'None',
  'SlidingWindow',
  'TokenBuffer',
  'RollingSummary',
  'WorkingMemoryBlackboard'
] as const;

const GLOBAL_LAST_KNOWN_GOOD_STORAGE_KEY = 'memarena.globalModelLastKnownGood.v1';
const KNOWN_GOOD_MODULES = ['chat', 'judge', 'summarizer', 'entity', 'reflector', 'embedding'] as const;
type KnownGoodModule = (typeof KNOWN_GOOD_MODULES)[number];

function loadLastKnownGoodSnapshots() {
  if (typeof window === 'undefined') return;
  try {
    const raw = window.localStorage.getItem(GLOBAL_LAST_KNOWN_GOOD_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object') {
      globalLastKnownGood.value = parsed as Record<string, { at: string; snapshot: Record<string, string> }>;
    }
  } catch {
    // ignore local cache parsing issues
  }
}

function persistLastKnownGoodSnapshots() {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(GLOBAL_LAST_KNOWN_GOOD_STORAGE_KEY, JSON.stringify(globalLastKnownGood.value));
  } catch {
    // ignore localStorage failures
  }
}

function moduleSnapshot(moduleName: string): Record<string, string> {
  const m = globalModelConfig.value;
  if (moduleName === 'chat') {
    return {
      provider: m.chat_llm_provider,
      api_base_url: m.chat_api_base_url,
      api_model: m.chat_api_model,
      ollama_base_url: m.chat_ollama_base_url,
      ollama_model: m.chat_ollama_model,
      local_model_path: m.chat_local_model_path,
    };
  }
  if (moduleName === 'judge') {
    return {
      provider: m.judge_llm_provider,
      api_base_url: m.judge_api_base_url,
      api_model: m.judge_api_model,
      ollama_base_url: m.judge_ollama_base_url,
      ollama_model: m.judge_ollama_model,
      local_model_path: m.judge_local_model_path,
    };
  }
  if (moduleName === 'summarizer') {
    return {
      provider: m.summarizer_llm_provider,
      api_base_url: m.summarizer_api_base_url,
      api_model: m.summarizer_api_model,
      ollama_base_url: m.summarizer_ollama_base_url,
      ollama_model: m.summarizer_ollama_model,
      local_model_path: m.summarizer_local_model_path,
    };
  }
  if (moduleName === 'entity') {
    return {
      provider: m.entity_llm_provider,
      api_base_url: m.entity_api_base_url,
      api_model: m.entity_api_model,
      ollama_base_url: m.entity_ollama_base_url,
      ollama_model: m.entity_ollama_model,
      local_model_path: m.entity_local_model_path,
    };
  }
  if (moduleName === 'reflector') {
    return {
      provider: m.reflector_llm_provider,
      api_base_url: m.reflector_api_base_url,
      api_model: m.reflector_api_model,
      ollama_base_url: m.reflector_ollama_base_url,
      ollama_model: m.reflector_ollama_model,
      local_model_path: m.reflector_local_model_path,
    };
  }
  if (moduleName === 'embedding') {
    return {
      provider: m.embedding_provider,
      api_base_url: m.embedding_api_base_url,
      api_model: m.embedding_api_model,
      ollama_base_url: m.embedding_ollama_base_url,
      ollama_model: m.embedding_ollama_model,
      local_model_path: m.embedding_local_model_path,
      local_infer_device: m.local_infer_device,
    };
  }
  return {};
}

function markModuleAsKnownGood(moduleName: string) {
  globalLastKnownGood.value[moduleName] = {
    at: new Date().toISOString(),
    snapshot: moduleSnapshot(moduleName),
  };
}

function applySnapshotToModule(moduleName: KnownGoodModule, snapshot: Record<string, string>) {
  const m = globalModelConfig.value;
  if (moduleName === 'chat') {
    m.chat_llm_provider = snapshot.provider ?? m.chat_llm_provider;
    m.chat_api_base_url = snapshot.api_base_url ?? m.chat_api_base_url;
    m.chat_api_model = snapshot.api_model ?? m.chat_api_model;
    m.chat_ollama_base_url = snapshot.ollama_base_url ?? m.chat_ollama_base_url;
    m.chat_ollama_model = snapshot.ollama_model ?? m.chat_ollama_model;
    m.chat_local_model_path = snapshot.local_model_path ?? m.chat_local_model_path;
    return;
  }
  if (moduleName === 'judge') {
    m.judge_llm_provider = snapshot.provider ?? m.judge_llm_provider;
    m.judge_api_base_url = snapshot.api_base_url ?? m.judge_api_base_url;
    m.judge_api_model = snapshot.api_model ?? m.judge_api_model;
    m.judge_ollama_base_url = snapshot.ollama_base_url ?? m.judge_ollama_base_url;
    m.judge_ollama_model = snapshot.ollama_model ?? m.judge_ollama_model;
    m.judge_local_model_path = snapshot.local_model_path ?? m.judge_local_model_path;
    return;
  }
  if (moduleName === 'summarizer') {
    m.summarizer_llm_provider = snapshot.provider ?? m.summarizer_llm_provider;
    m.summarizer_api_base_url = snapshot.api_base_url ?? m.summarizer_api_base_url;
    m.summarizer_api_model = snapshot.api_model ?? m.summarizer_api_model;
    m.summarizer_ollama_base_url = snapshot.ollama_base_url ?? m.summarizer_ollama_base_url;
    m.summarizer_ollama_model = snapshot.ollama_model ?? m.summarizer_ollama_model;
    m.summarizer_local_model_path = snapshot.local_model_path ?? m.summarizer_local_model_path;
    return;
  }
  if (moduleName === 'entity') {
    m.entity_llm_provider = snapshot.provider ?? m.entity_llm_provider;
    m.entity_api_base_url = snapshot.api_base_url ?? m.entity_api_base_url;
    m.entity_api_model = snapshot.api_model ?? m.entity_api_model;
    m.entity_ollama_base_url = snapshot.ollama_base_url ?? m.entity_ollama_base_url;
    m.entity_ollama_model = snapshot.ollama_model ?? m.entity_ollama_model;
    m.entity_local_model_path = snapshot.local_model_path ?? m.entity_local_model_path;
    return;
  }
  if (moduleName === 'reflector') {
    m.reflector_llm_provider = snapshot.provider ?? m.reflector_llm_provider;
    m.reflector_api_base_url = snapshot.api_base_url ?? m.reflector_api_base_url;
    m.reflector_api_model = snapshot.api_model ?? m.reflector_api_model;
    m.reflector_ollama_base_url = snapshot.ollama_base_url ?? m.reflector_ollama_base_url;
    m.reflector_ollama_model = snapshot.ollama_model ?? m.reflector_ollama_model;
    m.reflector_local_model_path = snapshot.local_model_path ?? m.reflector_local_model_path;
    return;
  }
  m.embedding_provider = snapshot.provider ?? m.embedding_provider;
  m.embedding_api_base_url = snapshot.api_base_url ?? m.embedding_api_base_url;
  m.embedding_api_model = snapshot.api_model ?? m.embedding_api_model;
  m.embedding_ollama_base_url = snapshot.ollama_base_url ?? m.embedding_ollama_base_url;
  m.embedding_ollama_model = snapshot.ollama_model ?? m.embedding_ollama_model;
  m.embedding_local_model_path = snapshot.local_model_path ?? m.embedding_local_model_path;
  m.local_infer_device = snapshot.local_infer_device ?? m.local_infer_device;
}

function restoreModuleFromKnownGood(moduleName: KnownGoodModule) {
  const item = globalLastKnownGood.value[moduleName];
  if (!item || !item.snapshot) {
    globalConfigMessage.value = `模块 ${moduleName} 暂无可恢复的最近可用配置。`;
    return;
  }
  applySnapshotToModule(moduleName, item.snapshot);
  globalConfigMessage.value = `已恢复 ${moduleName} 最近可用配置（请保存到 .env 并建议重测）。`;
}

function restoreAllFromKnownGood() {
  let restored = 0;
  KNOWN_GOOD_MODULES.forEach((moduleName) => {
    const item = globalLastKnownGood.value[moduleName];
    if (item?.snapshot) {
      applySnapshotToModule(moduleName, item.snapshot);
      restored += 1;
    }
  });
  globalConfigMessage.value = restored > 0
    ? `已恢复 ${restored} 个模块的最近可用配置（请保存到 .env 并建议重测）。`
    : '暂无可恢复的最近可用配置。';
}

function knownGoodState(moduleName: string) {
  const item = globalLastKnownGood.value[moduleName];
  if (!item) {
    return { exists: false, matched: false, at: '' };
  }
  const current = moduleSnapshot(moduleName);
  const matched = JSON.stringify(current) === JSON.stringify(item.snapshot);
  return { exists: true, matched, at: item.at };
}

const md = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true
});

const usingUploadedBatchCases = computed(() => datasetCases.value.length > 0);
const plannedBatchCaseCount = computed(() => {
  if (usingUploadedBatchCases.value) {
    return datasetCases.value.length;
  }
  return Math.max(1, Math.min(200, Number(batchCaseCount.value) || 1));
});
const batchInputModeLabel = computed(() =>
  usingUploadedBatchCases.value ? 'JSON 上传测试集模式' : '单条输入自动生成模式'
);
const isSummarizerProcessor = computed(() => config.value.processor === 'Summarizer');
const isEntityExtractorProcessor = computed(() => config.value.processor === 'EntityExtractor');
const showStmWindow = computed(() => shortTermMode.value === 'SlidingWindow');
const showStmTokenBudget = computed(() => shortTermMode.value === 'TokenBuffer');
const showStmRolling = computed(() => shortTermMode.value === 'RollingSummary');

const isEntityTripleMode = computed(() => {
  const method = config.value.entity_extractor_method;
  return method === 'llm_triple' || method === 'spacy_llm_triple';
});

function normalizeEntityEngineMapping() {
  if (!isEntityExtractorProcessor.value) return;
  config.value.engine = isEntityTripleMode.value ? 'GraphEngine' : 'RelationalEngine';
}

watch(() => config.value.processor, () => {
  normalizeEntityEngineMapping();
});

watch(() => config.value.entity_extractor_method, () => {
  normalizeEntityEngineMapping();
});

function clearEntityHealthWarning() {
  entityHealthState.value = null;
  entityDiagnosticCopyFeedback.value = '';
}

async function copyEntityDiagnostic() {
  if (!entityHealthState.value) {
    return;
  }
  const lines = [
    `[Entity Health] level=${entityHealthState.value.level}`,
    entityHealthState.value.message,
    entityHealthState.value.details,
    '',
    'Recent failures:'
  ];

  entityHealthState.value.recentFailures.forEach((evt, idx) => {
    lines.push(
      `${idx + 1}. ${evt.ts} | ${evt.purpose} | ${evt.provider}/${evt.model} | status=${evt.status}`,
      `   ${evt.error}`
    );
  });

  const text = lines.join('\n');
  try {
    await navigator.clipboard.writeText(text);
    entityDiagnosticCopyFeedback.value = '诊断信息已复制到剪贴板';
  } catch {
    entityDiagnosticCopyFeedback.value = '复制失败，请手动复制下方失败事件';
  }
}

async function copyBatchDiagnostic() {
  if (!batchResult.value) {
    return;
  }

  const m = batchResult.value.avg_metrics;
  const lines = [
    `[Batch Diagnostic] run_id=${batchResult.value.run_id}`,
    `cases=${batchResult.value.case_results.length}`,
    `avg_precision=${m.precision}`,
    `avg_faithfulness=${m.faithfulness}`,
    `avg_info_loss=${m.info_loss}`,
    `avg_recall_at_k=${m.recall_at_k ?? 'N/A'}`,
    `avg_qa_accuracy=${m.qa_accuracy ?? 'N/A'}`,
    `avg_qa_f1=${m.qa_f1 ?? 'N/A'}`,
    `avg_consistency_score=${m.consistency_score ?? 'N/A'}`,
    `avg_rejection_rate=${m.rejection_rate ?? 'N/A'}`,
    `avg_rejection_correctness_unknown=${m.rejection_correctness_unknown ?? 'N/A'}`,
    `avg_convergence_speed=${m.convergence_speed ?? 'N/A'}`,
    `avg_context_distraction=${m.context_distraction ?? 'N/A'}`
  ];

  if (entityHealthState.value) {
    lines.push('', '[Entity Health]', `level=${entityHealthState.value.level}`, entityHealthState.value.message, entityHealthState.value.details, '', 'Recent failures:');
    entityHealthState.value.recentFailures.forEach((evt, idx) => {
      lines.push(
        `${idx + 1}. ${evt.ts} | ${evt.purpose} | ${evt.provider}/${evt.model} | status=${evt.status}`,
        `   ${evt.error}`
      );
    });
  }

  try {
    await navigator.clipboard.writeText(lines.join('\n'));
    batchDiagnosticCopyFeedback.value = '批量诊断信息已复制到剪贴板';
  } catch {
    batchDiagnosticCopyFeedback.value = '复制失败，请手动复制当前批次摘要';
  }
}

async function refreshEntityHealthWarning(runId: string) {
  if (config.value.processor !== 'EntityExtractor') {
    clearEntityHealthWarning();
    return;
  }
  try {
    const audit = await getAuditEventsByRun(runId, 500, requestTimeoutMs.value);
    const events = Array.isArray(audit.events) ? audit.events : [];
    const entityEvents = events.filter((evt) => {
      const eventType = String(evt?.event_type || '');
      const purpose = String(evt?.purpose || '');
      return eventType === 'llm_generate' && purpose.startsWith('entity_extract_');
    });
    const failed = entityEvents.filter((evt) => evt?.ok === false);
    if (failed.length === 0) {
      clearEntityHealthWarning();
      return;
    }

    const latestFailed = failed[failed.length - 1] as Record<string, unknown>;
    const latestError = String(latestFailed?.error || 'unknown error');
    const has401Or429 = failed.some((evt) => {
      const errorText = String(evt?.error || '');
      return errorText.includes('401') || errorText.includes('429');
    });

    const provider = String(latestFailed?.provider || 'unknown');
    const model = String(latestFailed?.model || 'unknown');
    const purpose = String(latestFailed?.purpose || 'unknown');
    const statusMatch = latestError.match(/\b(401|429|4\d\d|5\d\d)\b/);
    const statusCode = statusMatch ? statusMatch[1] : 'unknown';
    const recentFailures = failed
      .slice(-3)
      .reverse()
      .map((evt) => {
        const errorText = String(evt?.error || 'unknown error');
        const statusHit = errorText.match(/\b(401|429|4\d\d|5\d\d)\b/);
        return {
          ts: String(evt?.ts || '-'),
          provider: String(evt?.provider || 'unknown'),
          model: String(evt?.model || 'unknown'),
          purpose: String(evt?.purpose || 'unknown'),
          status: statusHit ? statusHit[1] : 'unknown',
          error: errorText.split('\n')[0]
        };
      });

    const details = `provider=${provider} | model=${model} | purpose=${purpose} | status=${statusCode}`;

    entityHealthState.value = has401Or429
      ? {
          level: 'warning',
          message: 'Entity LLM 调用出现 401/429，系统已自动降级为启发式结构化抽取（结果可用，但质量上限受限）。',
          details,
          recentFailures
        }
      : {
          level: 'info',
          message: '检测到 Entity LLM 调用异常，系统已自动降级为启发式结构化抽取。',
          details,
          recentFailures
        };
  } catch {
    // 审计接口不可用时不阻断主流程
  }
}

function startRunTimer() {
  stopRunTimer();
  runStartTs = Date.now();
  elapsedMs.value = 0;
  timerHandle = setInterval(() => {
    elapsedMs.value = Date.now() - runStartTs;
  }, 200);
}

function stopRunTimer() {
  if (timerHandle) {
    clearInterval(timerHandle);
    timerHandle = null;
  }
  if (runStartTs > 0) {
    elapsedMs.value = Date.now() - runStartTs;
  }
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds
      .toString()
      .padStart(2, '0')}`;
  }
  return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

const runningDurationLabel = computed(() => formatDuration(elapsedMs.value));
const finishedDurationLabel = computed(() => {
  if (lastRunDurationMs.value == null) {
    return '';
  }
  return formatDuration(lastRunDurationMs.value);
});

const singleReasoningChains = computed(() => {
  if (!result.value) {
    return [] as string[];
  }
  const firstHit = result.value.search_result.hits[0];
  const meta = firstHit?.metadata;
  const chains = meta && Array.isArray(meta.reasoning_chains)
    ? (meta.reasoning_chains as unknown[])
    : [];
  return chains.map((c) => String(c)).filter(Boolean).slice(0, 20);
});

const singleReasoningSeeds = computed(() => {
  if (!result.value) {
    return [] as string[];
  }
  const firstHit = result.value.search_result.hits[0];
  const meta = firstHit?.metadata;
  const seeds = meta && Array.isArray(meta.reasoning_seed_entities)
    ? (meta.reasoning_seed_entities as unknown[])
    : [];
  return seeds.map((s) => String(s)).filter(Boolean).slice(0, 20);
});

const singleReasoningChainDetails = computed(() => {
  if (!result.value) {
    return [] as Array<{ chain: string; hop: number; seed_touch: boolean; lexical_overlap: number; priority: number }>;
  }
  const firstHit = result.value.search_result.hits[0];
  const details = firstHit?.metadata?.reasoning_chain_details;
  if (!Array.isArray(details)) {
    return [];
  }
  return details
    .map((d) => ({
      chain: String(d?.chain ?? ''),
      hop: Number(d?.hop ?? 0),
      seed_touch: Boolean(d?.seed_touch),
      lexical_overlap: Number(d?.lexical_overlap ?? 0),
      priority: Number(d?.priority ?? 0)
    }))
    .filter((d) => d.chain)
    .slice(0, 20);
});

const singleDerivedRows = computed(() => {
  if (!result.value) {
    return [] as Array<{ label: string; value: string; hint: string }>;
  }

  const precision = result.value.eval_result.metrics.precision;
  const coverage = result.value.eval_result.metrics.faithfulness;
  const infoLoss = result.value.eval_result.metrics.info_loss;
  const f1 = precision + coverage > 0 ? (2 * precision * coverage) / (precision + coverage) : 0;
  const retention = 1 - infoLoss;
  const hallucinationRisk = 1 - precision;
  const hitCount = result.value.search_result.hits.length;
  const avgRelevance =
    hitCount > 0
      ? result.value.search_result.hits.reduce((sum, hit) => sum + hit.relevance, 0) / hitCount
      : 0;

  const extraRows: Array<{ label: string; value: string; hint: string }> = [];
  if (result.value.eval_result.metrics.recall_at_k != null) {
    extraRows.push({
      label: 'Recall@K',
      value: `${(result.value.eval_result.metrics.recall_at_k * 100).toFixed(1)}%`,
      hint: '检索结果中命中目标事实的召回能力。'
    });
  }
  if (result.value.eval_result.metrics.qa_accuracy != null) {
    extraRows.push({
      label: 'QA Accuracy',
      value: `${(result.value.eval_result.metrics.qa_accuracy * 100).toFixed(1)}%`,
      hint: '回答是否完整覆盖目标事实（样本级）。'
    });
  }
  if (result.value.eval_result.metrics.qa_f1 != null) {
    extraRows.push({
      label: 'QA F1',
      value: `${(result.value.eval_result.metrics.qa_f1 * 100).toFixed(1)}%`,
      hint: '回答文本与目标事实的重叠平衡分。'
    });
  }
  if (result.value.eval_result.metrics.consistency_score != null) {
    extraRows.push({
      label: 'Consistency Score',
      value: `${(result.value.eval_result.metrics.consistency_score * 100).toFixed(1)}%`,
      hint: '回答是否与已知事实自洽，越高越稳定。'
    });
  }
  if (result.value.eval_result.metrics.rejection_rate != null) {
    extraRows.push({
      label: 'Rejection Rate',
      value: `${(result.value.eval_result.metrics.rejection_rate * 100).toFixed(1)}%`,
      hint: '模型实际拒答行为比例（行为信号，不代表是否拒答正确）。'
    });
  }
  if (result.value.eval_result.metrics.rejection_correctness_unknown != null) {
    extraRows.push({
      label: 'Rejection@Unknown',
      value: `${(result.value.eval_result.metrics.rejection_correctness_unknown * 100).toFixed(1)}%`,
      hint: '仅在未知问题子集统计：拒答是否正确。'
    });
  }
  if (result.value.eval_result.metrics.convergence_speed != null) {
    extraRows.push({
      label: 'Convergence Speed',
      value: `${result.value.eval_result.metrics.convergence_speed.toFixed(2)} cycles`,
      hint: '纠错后错误事实被洗掉所需周期估计，越低越好。'
    });
  }
  if (result.value.eval_result.metrics.context_distraction != null) {
    extraRows.push({
      label: 'Context Distraction',
      value: `${(result.value.eval_result.metrics.context_distraction * 100).toFixed(1)}%`,
      hint: '注入上下文中的噪声占比估计，越低越好。'
    });
  }

  return [
    ...extraRows,
    { label: 'F1 (P&R Balance)', value: `${(f1 * 100).toFixed(1)}%`, hint: '平衡精确率与覆盖率，避免只高其一。' },
    { label: 'Retention (1-InfoLoss)', value: `${(retention * 100).toFixed(1)}%`, hint: '保真保留程度，越高越好。' },
    { label: 'Hallucination Risk (1-P)', value: `${(hallucinationRisk * 100).toFixed(1)}%`, hint: '检索噪声或幻觉风险，越低越好。' },
    { label: 'Retrieved Hits', value: `${hitCount}`, hint: '本次检索命中条数。' },
    { label: 'Avg Hit Relevance', value: avgRelevance.toFixed(3), hint: '命中文档平均相关度。' }
  ];
});

const batchDerivedRows = computed(() => {
  if (!batchResult.value || batchResult.value.case_results.length === 0) {
    return [] as Array<{ label: string; value: string; hint: string }>;
  }

  const cases = batchResult.value.case_results;
  const f1Values = cases.map((r) => {
    const p = r.eval_result.metrics.precision;
    const c = r.eval_result.metrics.faithfulness;
    return p + c > 0 ? (2 * p * c) / (p + c) : 0;
  });
  const meanF1 = f1Values.reduce((sum, v) => sum + v, 0) / f1Values.length;
  const sortedF1 = [...f1Values].sort((a, b) => a - b);
  const medianF1 = sortedF1[Math.floor(sortedF1.length / 2)];
  const std = (values: number[]) => {
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
    return Math.sqrt(variance);
  };
  const precisionStd = std(cases.map((r) => r.eval_result.metrics.precision));
  const faithfulnessStd = std(cases.map((r) => r.eval_result.metrics.faithfulness));
  const infoLossStd = std(cases.map((r) => r.eval_result.metrics.info_loss));
  const passCount = cases.filter(
    (r) =>
      r.eval_result.metrics.precision >= 0.8 &&
      r.eval_result.metrics.faithfulness >= 0.8 &&
      r.eval_result.metrics.info_loss <= 0.2
  ).length;
  const passRate = passCount / cases.length;
  const worstF1 = sortedF1[0];

  const reasoningDetails = cases
    .map((r) => r.search_result.hits?.[0]?.metadata?.reasoning_chain_details)
    .filter((x): x is Array<{ chain: string; hop: number; seed_touch: boolean; lexical_overlap: number; priority: number }> => Array.isArray(x));
  const reasoningChainCounts = reasoningDetails.map((arr) => arr.length).filter((v) => v > 0);
  const reasoningPriorities = reasoningDetails.flatMap((arr) => arr.map((d) => Number(d.priority)).filter((v) => Number.isFinite(v)));
  const reasoningHops = reasoningDetails.flatMap((arr) => arr.map((d) => Number(d.hop)).filter((v) => Number.isFinite(v)));
  const reasoningSeedTouches = reasoningDetails.flatMap((arr) => arr.map((d) => Number(d.seed_touch ? 1 : 0)));

  const extraRows: Array<{ label: string; value: string; hint: string }> = [];
  if (batchResult.value.avg_metrics.recall_at_k != null) {
    extraRows.push({
      label: 'Avg Recall@K',
      value: `${(batchResult.value.avg_metrics.recall_at_k * 100).toFixed(1)}%`,
      hint: '批量平均检索召回能力。'
    });
  }
  if (batchResult.value.avg_metrics.qa_accuracy != null) {
    extraRows.push({
      label: 'Avg QA Accuracy',
      value: `${(batchResult.value.avg_metrics.qa_accuracy * 100).toFixed(1)}%`,
      hint: '批量平均回答正确覆盖率。'
    });
  }
  if (batchResult.value.avg_metrics.qa_f1 != null) {
    extraRows.push({
      label: 'Avg QA F1',
      value: `${(batchResult.value.avg_metrics.qa_f1 * 100).toFixed(1)}%`,
      hint: '批量平均回答语义重叠得分。'
    });
  }
  if (batchResult.value.avg_metrics.consistency_score != null) {
    extraRows.push({
      label: 'Avg Consistency Score',
      value: `${(batchResult.value.avg_metrics.consistency_score * 100).toFixed(1)}%`,
      hint: '批量平均一致性表现。'
    });
  }
  if (batchResult.value.avg_metrics.rejection_rate != null) {
    extraRows.push({
      label: 'Avg Rejection Rate',
      value: `${(batchResult.value.avg_metrics.rejection_rate * 100).toFixed(1)}%`,
      hint: '批量实际拒答行为比例（行为信号）。'
    });
  }
  if (batchResult.value.avg_metrics.rejection_correctness_unknown != null) {
    extraRows.push({
      label: 'Avg Rejection@Unknown',
      value: `${(batchResult.value.avg_metrics.rejection_correctness_unknown * 100).toFixed(1)}%`,
      hint: '仅未知问题子集统计的批量拒答正确率。'
    });
  }
  if (batchResult.value.avg_metrics.convergence_speed != null) {
    extraRows.push({
      label: 'Avg Convergence Speed',
      value: `${batchResult.value.avg_metrics.convergence_speed.toFixed(2)} cycles`,
      hint: '批量平均纠错收敛周期估计，越低越好。'
    });
  }
  if (batchResult.value.avg_metrics.context_distraction != null) {
    extraRows.push({
      label: 'Avg Context Distraction',
      value: `${(batchResult.value.avg_metrics.context_distraction * 100).toFixed(1)}%`,
      hint: '批量平均上下文干扰度，越低越好。'
    });
  }
  if (reasoningChainCounts.length > 0) {
    const meanChainCount = reasoningChainCounts.reduce((s, v) => s + v, 0) / reasoningChainCounts.length;
    extraRows.push({
      label: 'Avg Reasoning Chain Count',
      value: meanChainCount.toFixed(2),
      hint: '单样本推理链数量，过高可能引入噪声。'
    });
  }
  if (reasoningPriorities.length > 0) {
    const meanPriority = reasoningPriorities.reduce((s, v) => s + v, 0) / reasoningPriorities.length;
    extraRows.push({
      label: 'Avg Reasoning Priority',
      value: meanPriority.toFixed(3),
      hint: '链路优先级均值，越高代表链路与问题更贴近。'
    });
  }
  if (reasoningHops.length > 0) {
    const meanHop = reasoningHops.reduce((s, v) => s + v, 0) / reasoningHops.length;
    extraRows.push({
      label: 'Avg Reasoning Hop',
      value: meanHop.toFixed(2),
      hint: '推理链平均跳数，越低通常越直接。'
    });
  }
  if (reasoningSeedTouches.length > 0) {
    const seedRatio = reasoningSeedTouches.reduce((s, v) => s + Number(v), 0) / reasoningSeedTouches.length;
    extraRows.push({
      label: 'Reasoning Seed Touch Ratio',
      value: `${(seedRatio * 100).toFixed(1)}%`,
      hint: '链路是否命中种子实体的比例，越高越稳定。'
    });
  }

  return [
    ...extraRows,
    { label: 'Pass Rate (P>=0.8/F>=0.8/L<=0.2)', value: `${(passRate * 100).toFixed(1)}%`, hint: '达标样本比例，避免只看均值。' },
    { label: 'Mean F1', value: `${(meanF1 * 100).toFixed(1)}%`, hint: '批量整体平衡表现。' },
    { label: 'Median F1', value: `${(medianF1 * 100).toFixed(1)}%`, hint: '中位水平，降低极端值影响。' },
    { label: 'Worst-case F1', value: `${(worstF1 * 100).toFixed(1)}%`, hint: '最差样本性能，反映可靠性下限。' },
    { label: 'Std(P/F/L)', value: `${precisionStd.toFixed(3)} / ${faithfulnessStd.toFixed(3)} / ${infoLossStd.toFixed(3)}`, hint: '波动越小说明系统越稳定。' }
  ];
});

const singleSafetySignals = computed(() => {
  if (!result.value) {
    return null;
  }
  const m = result.value.eval_result.metrics;
  if (m.rejection_rate == null && m.rejection_correctness_unknown == null) {
    return null;
  }
  return {
    rejectionRate: m.rejection_rate,
    rejectionUnknownCorrectness: m.rejection_correctness_unknown
  };
});

const batchSafetySignals = computed(() => {
  if (!batchResult.value) {
    return null;
  }
  const m = batchResult.value.avg_metrics;
  const unknownCount = batchResult.value.case_results.filter(
    (r) => r.eval_result.metrics.rejection_correctness_unknown != null
  ).length;
  const knownCount = batchResult.value.case_results.length - unknownCount;

  if (m.rejection_rate == null && m.rejection_correctness_unknown == null) {
    return null;
  }

  return {
    rejectionRate: m.rejection_rate,
    rejectionUnknownCorrectness: m.rejection_correctness_unknown,
    unknownCount,
    knownCount,
    unknownRatio: batchResult.value.case_results.length > 0 ? unknownCount / batchResult.value.case_results.length : 0
  };
});

const batchSafetyInterpretation = computed(() => {
  if (!batchSafetySignals.value) {
    return '';
  }
  const s = batchSafetySignals.value;
  if (s.unknownCount === 0) {
    return '当前批次没有未知样本，Rejection@Unknown 不具代表性。';
  }
  if (s.rejectionUnknownCorrectness == null) {
    return '当前批次缺少可用的未知样本正确率统计。';
  }
  if (s.rejectionUnknownCorrectness >= 0.8) {
    return '未知问题拒答正确率较高，安全性表现良好。';
  }
  if (s.rejectionUnknownCorrectness >= 0.5) {
    return '未知问题拒答正确率中等，建议继续优化拒答模板与反例库。';
  }
  return '未知问题拒答正确率偏低，建议优先排查误答模式与拒答策略。';
});

const singleRoleGroupedHits = computed(() => {
  if (!result.value) {
    return null;
  }
  const hits = result.value.search_result.hits;
  const byRole = (subset: typeof hits) => {
    const user = subset.filter((h) => String((h.metadata?.role as string | undefined) || '').toLowerCase() === 'user');
    const assistant = subset.filter((h) => String((h.metadata?.role as string | undefined) || '').toLowerCase() === 'assistant');
    const other = subset.filter((h) => {
      const role = String((h.metadata?.role as string | undefined) || '').toLowerCase();
      return role !== 'user' && role !== 'assistant';
    });
    return { user, assistant, other };
  };

  const stm = byRole(hits.filter((h) => Boolean(h.metadata?.stm)));
  const ltm = byRole(hits.filter((h) => !Boolean(h.metadata?.stm)));
  return { stm, ltm };
});

function asPrettyJson(value: unknown): string {
  try {
    return JSON.stringify(value ?? {}, null, 2);
  } catch {
    return String(value ?? '');
  }
}

const singleModuleTraceJson = computed(() => asPrettyJson(result.value?.module_trace ?? {}));

function moduleTraceMarkdownSection(trace: unknown): string {
  return ['## Module Trace', '```json', asPrettyJson(trace), '```'].join('\n');
}

type PromptCheckRow = {
  purpose: string;
  provider: string;
  model: string;
  severity: 'ok' | 'warn' | 'error';
  summary: string;
  details: string;
};

function buildPromptQualityChecks(moduleTrace: Record<string, unknown> | undefined): PromptCheckRow[] {
  const llmCalls = Array.isArray((moduleTrace as any)?.llm_calls) ? ((moduleTrace as any).llm_calls as Array<Record<string, unknown>>) : [];
  const expectJson = (purpose: string) => {
    const p = String(purpose || '');
    return p === 'judge' || p.startsWith('entity_extract_') || p.startsWith('reflector_');
  };

  const checks: PromptCheckRow[] = [];
  llmCalls.forEach((call) => {
    const purpose = String(call.purpose || '');
    const provider = String(call.provider || 'unknown');
    const model = String(call.model || 'unknown');
    const ok = Boolean(call.ok);
    const promptPreview = String(call.prompt_preview || '');
    const systemPreview = String(call.system_prompt_preview || '');
    const promptLength = Number(call.prompt_length || 0);
    const error = String(call.error || '');
    const needsJson = expectJson(purpose);
    const promptLower = (promptPreview + '\n' + systemPreview).toLowerCase();
    const hasJsonConstraint = /json|输出\s*json|only valid json|strict json/.test(promptLower);

    if (!ok) {
      checks.push({
        purpose,
        provider,
        model,
        severity: 'error',
        summary: 'LLM 调用失败',
        details: error || 'unknown error'
      });
      return;
    }

    if (needsJson && !hasJsonConstraint) {
      checks.push({
        purpose,
        provider,
        model,
        severity: 'warn',
        summary: '疑似缺少 JSON 输出约束',
        details: '该 purpose 下游通常按结构化结果消费，建议在 prompt 或 system prompt 明确 JSON schema。'
      });
    } else if (promptLength > 0 && promptLength < 40) {
      checks.push({
        purpose,
        provider,
        model,
        severity: 'warn',
        summary: 'Prompt 过短',
        details: `prompt_length=${promptLength}，可能上下文不足导致输出不稳定。`
      });
    } else {
      checks.push({
        purpose,
        provider,
        model,
        severity: 'ok',
        summary: 'Prompt 检查通过',
        details: `prompt_length=${promptLength}`
      });
    }
  });

  return checks;
}

const singlePromptQualityChecks = computed(() => {
  const trace = (result.value?.module_trace ?? {}) as Record<string, unknown>;
  return buildPromptQualityChecks(trace);
});

function promptChecksMarkdownSection(rows: PromptCheckRow[]): string {
  if (!rows.length) {
    return ['## Prompt Quality Checks', '- (none)'].join('\n');
  }
  return [
    '## Prompt Quality Checks',
    ...rows.map((r, i) => `- [${i + 1}] ${r.severity.toUpperCase()} | ${r.purpose} | ${r.provider}/${r.model} | ${r.summary} | ${r.details}`)
  ].join('\n');
}

const markdownReport = computed(() => {
  const formatRoleLines = (
    hits: Array<{ content: string; relevance: number; metadata?: Record<string, unknown> }>,
    role: 'user' | 'assistant' | 'other'
  ) => {
    const roleHits = hits.filter((hit) => {
      const r = String((hit.metadata?.role as string | undefined) || '').toLowerCase();
      if (role === 'other') {
        return r !== 'user' && r !== 'assistant';
      }
      return r === role;
    });
    return roleHits.length
      ? roleHits
          .map((hit, idx) => `- [${idx + 1}] score=${hit.relevance.toFixed(4)}\n  - ${hit.content}`)
          .join('\n')
      : '- (none)';
  };

  if (result.value) {
    const r = result.value;
    const stmHits = r.search_result.hits.filter((hit) => Boolean(hit.metadata?.stm));
    const ltmHits = r.search_result.hits.filter((hit) => !Boolean(hit.metadata?.stm));
    const stmUserLines = formatRoleLines(stmHits, 'user');
    const stmAssistantLines = formatRoleLines(stmHits, 'assistant');
    const stmOtherLines = formatRoleLines(stmHits, 'other');
    const ltmUserLines = formatRoleLines(ltmHits, 'user');
    const ltmAssistantLines = formatRoleLines(ltmHits, 'assistant');
    const ltmOtherLines = formatRoleLines(ltmHits, 'other');

    return [
      '# MemArena Test Report',
      '',
      '## Meta',
      `- run_id: ${r.run_id}`,
      `- mode: single`,
      `- generated_at: ${new Date().toISOString()}`,
      '',
      '## Agent Reply',
      '',
      r.generated_response || '(empty)',
      '',
      '## Agent Real-time Memory (STM)',
      '### User',
      stmUserLines,
      '### Assistant',
      stmAssistantLines,
      '### Other',
      stmOtherLines,
      '',
      '## Agent Real-time Memory (LTM)',
      '### User',
      ltmUserLines,
      '### Assistant',
      ltmAssistantLines,
      '### Other',
      ltmOtherLines,
      '',
      moduleTraceMarkdownSection(r.module_trace ?? {}),
      '',
      promptChecksMarkdownSection(singlePromptQualityChecks.value),
      '',
      '## Metrics',
      `- precision: ${r.eval_result.metrics.precision}`,
      `- faithfulness: ${r.eval_result.metrics.faithfulness}`,
      `- info_loss: ${r.eval_result.metrics.info_loss}`,
      `- recall_at_k: ${r.eval_result.metrics.recall_at_k ?? 'N/A'}`,
      `- qa_accuracy: ${r.eval_result.metrics.qa_accuracy ?? 'N/A'}`,
      `- qa_f1: ${r.eval_result.metrics.qa_f1 ?? 'N/A'}`,
      `- consistency_score: ${r.eval_result.metrics.consistency_score ?? 'N/A'}`,
      `- rejection_rate: ${r.eval_result.metrics.rejection_rate ?? 'N/A'}`,
      `- rejection_correctness_unknown: ${r.eval_result.metrics.rejection_correctness_unknown ?? 'N/A'}`,
      `- convergence_speed: ${r.eval_result.metrics.convergence_speed ?? 'N/A'}`,
      `- context_distraction: ${r.eval_result.metrics.context_distraction ?? 'N/A'}`,
      ...(singleReasoningChainDetails.value.length > 0
        ? [
            '',
            '## Reasoning Quality',
            `- seed_entities: ${singleReasoningSeeds.value.join(', ') || '(none)'}`,
            `- chain_count: ${singleReasoningChainDetails.value.length}`,
            `- avg_priority: ${(
              singleReasoningChainDetails.value.reduce((s, d) => s + Number(d.priority || 0), 0) /
              Math.max(singleReasoningChainDetails.value.length, 1)
            ).toFixed(4)}`,
            `- avg_hop: ${(
              singleReasoningChainDetails.value.reduce((s, d) => s + Number(d.hop || 0), 0) /
              Math.max(singleReasoningChainDetails.value.length, 1)
            ).toFixed(4)}`
          ]
        : []),
      '',
      '## Judge Rationale',
      r.eval_result.judge_rationale || '(empty)',
      ...(includeRawJudgeInMarkdown.value
        ? [
            '',
            '## Raw Judge Output',
            r.eval_result.raw_judge_output || '(empty)'
          ]
        : [])
    ].join('\n');
  }

  if (batchResult.value) {
    const b = batchResult.value;
    const caseSections = b.case_results
      .map((c, idx) => {
        const stmHits = c.search_result.hits.filter((hit) => Boolean(hit.metadata?.stm));
        const ltmHits = c.search_result.hits.filter((hit) => !Boolean(hit.metadata?.stm));
        const stmUser = formatRoleLines(stmHits, 'user').replace(/^- /gm, '  - ');
        const stmAssistant = formatRoleLines(stmHits, 'assistant').replace(/^- /gm, '  - ');
        const stmOther = formatRoleLines(stmHits, 'other').replace(/^- /gm, '  - ');
        const ltmUser = formatRoleLines(ltmHits, 'user').replace(/^- /gm, '  - ');
        const ltmAssistant = formatRoleLines(ltmHits, 'assistant').replace(/^- /gm, '  - ');
        const ltmOther = formatRoleLines(ltmHits, 'other').replace(/^- /gm, '  - ');

        return [
          `### Case ${idx + 1}`,
          `- run_id: ${c.run_id}`,
          '- Agent Reply:',
          c.generated_response || '(empty)',
          '- Agent Real-time Memory (STM):',
          '  - User:',
          stmUser,
          '  - Assistant:',
          stmAssistant,
          '  - Other:',
          stmOther,
          '- Agent Real-time Memory (LTM):',
          '  - User:',
          ltmUser,
          '  - Assistant:',
          ltmAssistant,
          '  - Other:',
          ltmOther,
          '#### Module Trace',
          '```json',
          asPrettyJson(c.module_trace ?? {}),
          '```',
          '#### Prompt Quality Checks',
          ...promptChecksMarkdownSection(buildPromptQualityChecks((c.module_trace ?? {}) as Record<string, unknown>)).split('\n').slice(1),
          '- Metrics:',
          `  - precision: ${c.eval_result.metrics.precision}`,
          `  - faithfulness: ${c.eval_result.metrics.faithfulness}`,
          `  - info_loss: ${c.eval_result.metrics.info_loss}`,
          `  - rejection_rate: ${c.eval_result.metrics.rejection_rate ?? 'N/A'}`,
          `  - rejection_correctness_unknown: ${c.eval_result.metrics.rejection_correctness_unknown ?? 'N/A'}`,
          ...(includeRawJudgeInMarkdown.value
            ? [
                '  - raw_judge_output:',
                `    ${c.eval_result.raw_judge_output || '(empty)'}`
              ]
            : []),
          ''
        ].join('\n');
      })
      .join('\n');

    return [
      '# MemArena Test Report',
      '',
      '## Meta',
      `- run_id: ${b.run_id}`,
      `- mode: batch`,
      `- cases: ${b.case_results.length}`,
      `- generated_at: ${new Date().toISOString()}`,
      '',
      '## Batch Avg Metrics',
      `- precision: ${b.avg_metrics.precision}`,
      `- faithfulness: ${b.avg_metrics.faithfulness}`,
      `- info_loss: ${b.avg_metrics.info_loss}`,
      `- rejection_rate: ${b.avg_metrics.rejection_rate ?? 'N/A'}`,
      `- rejection_correctness_unknown: ${b.avg_metrics.rejection_correctness_unknown ?? 'N/A'}`,
      `- convergence_speed: ${b.avg_metrics.convergence_speed ?? 'N/A'}`,
      `- context_distraction: ${b.avg_metrics.context_distraction ?? 'N/A'}`,
      '',
      '## Case Details',
      caseSections
    ].join('\n');
  }

  return '';
});

const renderedMarkdownReport = computed(() => {
  if (!markdownReport.value.trim()) {
    return '';
  }
  return md.render(markdownReport.value);
});

function downloadMarkdownReport() {
  if (!markdownReport.value) return;
  const blob = new Blob([markdownReport.value], { type: 'text/markdown;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  const runId = result.value?.run_id || batchResult.value?.run_id || 'report';
  link.href = url;
  link.download = `memarena-report-${runId}.md`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

onBeforeUnmount(() => {
  stopRunTimer();
});

async function loadBuiltinDatasets(showMessage = false) {
  builtinDatasetsLoading.value = true;
  if (showMessage) {
    builtinDatasetsMessage.value = '';
  }
  try {
    builtinDatasets.value = await listDatasets();
    if (!selectedDatasetName.value && builtinDatasets.value.length > 0) {
      selectedDatasetName.value = builtinDatasets.value[0].name;
      const count = Number(builtinDatasets.value[0].count ?? 0);
      datasetSampleSize.value = count > 0 ? Math.min(5, count) : 5;
    }
    if (showMessage) {
      builtinDatasetsMessage.value = `已刷新 ${builtinDatasets.value.length} 个数据集。`;
    }
  } catch {
    builtinDatasetsMessage.value = '刷新失败，请检查后端服务。';
  } finally {
    builtinDatasetsLoading.value = false;
  }
}

loadBuiltinDatasets();

async function loadGlobalModelConfigPanel() {
  globalConfigLoading.value = true;
  globalConfigMessage.value = '';
  try {
    const resp = await getGlobalModelConfig(requestTimeoutMs.value);
    globalModelConfig.value = { ...resp.config };
    globalConfigEnvFile.value = resp.env_file;
  } catch {
    globalConfigMessage.value = '加载全局模型配置失败，请检查后端服务。';
  } finally {
    globalConfigLoading.value = false;
  }
}

async function saveGlobalModelConfigPanel() {
  globalConfigSaving.value = true;
  globalConfigMessage.value = '';
  try {
    const resp = await updateGlobalModelConfig(globalModelConfig.value, requestTimeoutMs.value);
    globalModelConfig.value = { ...resp.config };
    globalConfigEnvFile.value = resp.env_file;
    globalConfigMessage.value = '全局模型配置已保存到 .env（新请求将使用最新配置）。';
  } catch {
    globalConfigMessage.value = '保存失败，请检查输入格式与后端服务。';
  } finally {
    globalConfigSaving.value = false;
  }
}

const defaultConnectivityModules = [...KNOWN_GOOD_MODULES];

async function runGlobalConnectivityTest(modules: string[] = defaultConnectivityModules) {
  globalConnectivityTesting.value = true;
  globalConnectivityMessage.value = '';
  try {
    const resp = await testGlobalModelConnectivity(modules, requestTimeoutMs.value);
    globalConnectivity.value = resp;
    resp.results.forEach((row) => {
      if (row.ok) {
        markModuleAsKnownGood(String(row.module || '').toLowerCase());
      }
    });
    persistLastKnownGoodSnapshots();
    globalConnectivityMessage.value = `连通性测试完成：${resp.passed}/${resp.total} 通过。`;
  } catch {
    globalConnectivityMessage.value = '连通性测试失败，请检查后端服务与网络。';
  } finally {
    globalConnectivityTesting.value = false;
  }
}

function connectivityResult(moduleName: string) {
  return globalConnectivity.value?.results?.find((x) => String(x.module || '').toLowerCase() === moduleName.toLowerCase()) || null;
}

loadGlobalModelConfigPanel();
loadLastKnownGoodSnapshots();

function handleDatasetUpload(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    datasetJson.value = String(reader.result || '');
    try {
      const parsed = JSON.parse(datasetJson.value);
      if (Array.isArray(parsed) && parsed.length > 0) {
        datasetCases.value = parsed.map((item: any, idx: number) => ({
          case_id: String(item.case_id || `case-${idx + 1}`),
          input_text: String(item.input_text || ''),
          expected_facts: Array.isArray(item.expected_facts) ? item.expected_facts.map(String) : [],
          session_id: String(item.session_id || 'batch-session')
        }));
        if (datasetCases.value[0]?.input_text) {
          inputText.value = datasetCases.value[0].input_text;
        }
      }
    } catch {
      // 用户可上传非结构化文本，这里不强制报错
    }
  };
  reader.readAsText(file);
}

async function onRunBenchmark() {
  loading.value = true;
  error.value = '';
  batchResult.value = null;
  clearEntityHealthWarning();
  progressText.value = '';
  lastRunDurationMs.value = null;
  startRunTimer();

  try {
    normalizeEntityEngineMapping();

    const expectedFacts = expectedFactsRaw.value
      .split('\n')
      .map((v: string) => v.trim())
      .filter(Boolean);

    result.value = await runBenchmarkWithTimeout({
      config: config.value,
      session_id: 'ui-session-001',
      user_id: 'ui-user-001',
      input_text: inputText.value,
      expected_facts: expectedFacts,
      retrieval: {
        top_k: retrievalTopK.value,
        min_relevance: minRelevance.value,
        collection_name: collectionName.value,
        similarity_strategy: similarityStrategy.value,
        keyword_rerank: keywordRerank.value,
        max_context_tokens: maxContextTokens.value,
        reasoning_hops: reasoningHops.value,
        short_term_mode: shortTermMode.value,
        stm_window_turns: stmWindowTurns.value,
        stm_token_budget: stmTokenBudget.value,
        stm_summary_keep_recent_turns: stmSummaryKeepRecentTurns.value,
        reflector_auto_writeback: reflectorAutoWriteback.value,
        reflector_writeback_min_confidence: reflectorWritebackMinConfidence.value,
        reflector_llm_mode: reflectorLlmMode.value
      }
    }, requestTimeoutMs.value);

    if (result.value?.run_id) {
      await refreshEntityHealthWarning(result.value.run_id);
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : '运行失败，请检查后端服务。';
  } finally {
    stopRunTimer();
    lastRunDurationMs.value = elapsedMs.value;
    loading.value = false;
  }
}

async function onRunBatchBenchmark() {
  loading.value = true;
  error.value = '';
  result.value = null;
  clearEntityHealthWarning();
  progressText.value = '';
  lastRunDurationMs.value = null;
  startRunTimer();

  try {
    normalizeEntityEngineMapping();

    const expectedFacts = expectedFactsRaw.value
      .split('\n')
      .map((v: string) => v.trim())
      .filter(Boolean);

    const useUploadedCases = datasetCases.value.length > 0;
    if (!useUploadedCases && !inputText.value.trim()) {
      throw new Error('请填写单条输入，或上传 JSON 数组测试集。');
    }

    const generatedCaseCount = Math.max(1, Math.min(200, Number(batchCaseCount.value) || 1));
    const generatedCases = Array.from({ length: generatedCaseCount }, (_, idx) => ({
      case_id: `generated-${idx + 1}`,
      input_text: inputText.value,
      expected_facts: expectedFacts,
      session_id: 'batch-session'
    }));

    const casesToRun = useUploadedCases ? datasetCases.value : generatedCases;

    progressText.value = `Batch Progress: 0/${casesToRun.length} (submitting)`;

    const startResp = await runBatchBenchmarkAsync({
      config: config.value,
      user_id: 'ui-batch-user',
      isolate_sessions: isolateSessions.value,
      max_concurrency: maxConcurrency.value,
      retrieval: {
        top_k: retrievalTopK.value,
        min_relevance: minRelevance.value,
        collection_name: collectionName.value,
        similarity_strategy: similarityStrategy.value,
        keyword_rerank: keywordRerank.value,
        max_context_tokens: maxContextTokens.value,
        reasoning_hops: reasoningHops.value,
        short_term_mode: shortTermMode.value,
        stm_window_turns: stmWindowTurns.value,
        stm_token_budget: stmTokenBudget.value,
        stm_summary_keep_recent_turns: stmSummaryKeepRecentTurns.value,
        reflector_auto_writeback: reflectorAutoWriteback.value,
        reflector_writeback_min_confidence: reflectorWritebackMinConfidence.value,
        reflector_llm_mode: reflectorLlmMode.value
      },
      cases: casesToRun
    }, requestTimeoutMs.value);

    progressText.value = `Batch Progress: 0/${casesToRun.length} (queued)`;

    while (true) {
      const status = await getAsyncRunStatus(startResp.run_id, requestTimeoutMs.value);
      progressText.value = `Batch Progress: ${status.completed}/${status.total} (${status.status})`;
      if (status.status === 'completed' && status.result) {
        batchResult.value = status.result;
        if (status.result.run_id) {
          await refreshEntityHealthWarning(status.result.run_id);
        }
        break;
      }
      if (status.status === 'failed' || status.status === 'not_found') {
        throw new Error(status.message || '批量任务失败');
      }
      await new Promise((resolve) => setTimeout(resolve, 800));
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : '批量运行失败，请检查后端服务。';
  } finally {
    stopRunTimer();
    lastRunDurationMs.value = elapsedMs.value;
    loading.value = false;
  }
}

async function onRunBuiltinDataset() {
  loading.value = true;
  error.value = '';
  result.value = null;
  clearEntityHealthWarning();
  progressText.value = '';
  lastRunDurationMs.value = null;
  startRunTimer();

  try {
    normalizeEntityEngineMapping();

    if (!selectedDatasetName.value) {
      throw new Error('请先选择内置数据集。');
    }

    const selectedMeta = builtinDatasets.value.find((d) => d.name === selectedDatasetName.value);
    const available = Math.max(0, (selectedMeta?.count ?? 0) - datasetStartIndex.value);
    const plannedTotal = Math.max(0, Math.min(datasetSampleSize.value, available));
    progressText.value = `Dataset Progress: 0/${plannedTotal} (submitting)`;

    const startResp = await runDatasetBenchmarkAsync({
      dataset_name: selectedDatasetName.value,
      config: config.value,
      user_id: 'ui-dataset-user',
      sample_size: datasetSampleSize.value,
      start_index: datasetStartIndex.value,
      isolate_sessions: isolateSessions.value,
      max_concurrency: maxConcurrency.value,
      retrieval: {
        top_k: retrievalTopK.value,
        min_relevance: minRelevance.value,
        collection_name: collectionName.value,
        similarity_strategy: similarityStrategy.value,
        keyword_rerank: keywordRerank.value,
        max_context_tokens: maxContextTokens.value,
        reasoning_hops: reasoningHops.value,
        short_term_mode: shortTermMode.value,
        stm_window_turns: stmWindowTurns.value,
        stm_token_budget: stmTokenBudget.value,
        stm_summary_keep_recent_turns: stmSummaryKeepRecentTurns.value,
        reflector_auto_writeback: reflectorAutoWriteback.value,
        reflector_writeback_min_confidence: reflectorWritebackMinConfidence.value,
        reflector_llm_mode: reflectorLlmMode.value
      }
    }, requestTimeoutMs.value);

    progressText.value = `Dataset Progress: 0/${plannedTotal} (queued)`;

    while (true) {
      const status = await getAsyncRunStatus(startResp.run_id, requestTimeoutMs.value);
      progressText.value = `Dataset Progress: ${status.completed}/${status.total} (${status.status})`;
      if (status.status === 'completed' && status.result) {
        batchResult.value = status.result;
        if (status.result.run_id) {
          await refreshEntityHealthWarning(status.result.run_id);
        }
        break;
      }
      if (status.status === 'failed' || status.status === 'not_found') {
        throw new Error(status.message || '内置数据集任务失败');
      }
      await new Promise((resolve) => setTimeout(resolve, 800));
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : '内置数据集运行失败。';
  } finally {
    stopRunTimer();
    lastRunDurationMs.value = elapsedMs.value;
    loading.value = false;
  }
}

function downloadCsvReport() {
  if (!batchResult.value?.csv_report) return;
  const blob = new Blob([batchResult.value.csv_report], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `memarena-report-${batchResult.value.run_id}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
</script>

<template>
  <div class="min-h-screen bg-aurora text-slate-100">
    <div class="mx-auto grid max-w-[1320px] grid-cols-1 gap-6 p-4 md:p-8 lg:grid-cols-3">
      <section class="panel lg:col-span-1">
        <h2 class="panel-title">Configuration Panel</h2>
        <div class="grid gap-3">
          <label class="field">
            <span>Processor</span>
            <select v-model="config.processor" class="select">
              <option v-for="x in processors" :key="x" :value="x">{{ x }}</option>
            </select>
          </label>
          <label v-if="isSummarizerProcessor" class="field">
            <span>Summarizer Method</span>
            <select v-model="config.summarizer_method" class="select">
              <option v-for="x in summarizerMethods" :key="x" :value="x">{{ x }}</option>
            </select>
          </label>
          <label v-if="isEntityExtractorProcessor" class="field">
            <span>EntityExtractor Method</span>
            <select v-model="config.entity_extractor_method" class="select">
              <option v-for="x in entityExtractorMethods" :key="x" :value="x">{{ x }}</option>
            </select>
          </label>
          <label class="field">
            <span>Engine</span>
            <select v-model="config.engine" class="select">
              <option v-for="x in engines" :key="x" :value="x">{{ x }}</option>
            </select>
          </label>
          <p v-if="isEntityExtractorProcessor" class="rounded-lg border border-slate-700/70 bg-slate-900/50 p-2 text-xs text-slate-300">
            EntityExtractor 自动映射引擎：
            <span class="font-semibold text-arena-mint">{{ isEntityTripleMode ? 'Triple -> GraphEngine' : 'Attribute -> RelationalEngine' }}</span>
          </p>
          <label class="field">
            <span>Assembler</span>
            <select v-model="config.assembler" class="select">
              <option v-for="x in assemblers" :key="x" :value="x">{{ x }}</option>
            </select>
          </label>
          <label class="field">
            <span>Reflector</span>
            <select v-model="config.reflector" class="select">
              <option v-for="x in reflectors" :key="x" :value="x">{{ x }}</option>
            </select>
          </label>
          <label class="field">
            <span>Chat LLM Provider</span>
            <select v-model="config.chat_llm_provider" class="select">
              <option v-for="x in providers" :key="x" :value="x">{{ x }}</option>
            </select>
          </label>
          <label class="field">
            <span>Judge LLM Provider</span>
            <select v-model="config.judge_llm_provider" class="select">
              <option v-for="x in providers" :key="x" :value="x">{{ x }}</option>
            </select>
          </label>
          <label v-if="isSummarizerProcessor" class="field">
            <span>Summarizer LLM Provider</span>
            <select v-model="config.summarizer_llm_provider" class="select">
              <option v-for="x in providers" :key="x" :value="x">{{ x }}</option>
            </select>
          </label>
          <label v-if="isEntityExtractorProcessor" class="field">
            <span>Entity LLM Provider</span>
            <select v-model="config.entity_llm_provider" class="select">
              <option v-for="x in providers" :key="x" :value="x">{{ x }}</option>
            </select>
          </label>
          <label v-if="config.reflector !== 'None'" class="field">
            <span>Reflector LLM Provider</span>
            <select v-model="config.reflector_llm_provider" class="select">
              <option v-for="x in providers" :key="x" :value="x">{{ x }}</option>
            </select>
          </label>
          <label class="field">
            <span>Embedding Provider</span>
            <select v-model="config.embedding_provider" class="select">
              <option v-for="x in providers" :key="x" :value="x">{{ x }}</option>
            </select>
          </label>
          <label class="field">
            <span>Compute Device (Local)</span>
            <select v-model="config.compute_device" class="select">
              <option v-for="x in computeDevices" :key="x" :value="x">{{ x }}</option>
            </select>
          </label>

          <div class="global-config-panel mt-2 rounded-lg border border-slate-600/60 bg-slate-900/40 p-3">
            <div class="mb-2 flex items-center justify-between">
              <p class="text-xs font-semibold text-arena-mint">全局大模型设置（写入 .env）</p>
              <div class="flex items-center gap-2">
                <button
                  class="rounded border border-slate-500/60 bg-slate-900/60 px-2 py-1 text-[11px] text-slate-100 transition hover:border-slate-300"
                  :disabled="globalConfigLoading"
                  @click="loadGlobalModelConfigPanel"
                >
                  {{ globalConfigLoading ? '加载中...' : '刷新' }}
                </button>
                <button
                  class="rounded border border-arena-cyan/60 bg-arena-cyan/20 px-2 py-1 text-[11px] text-arena-mint transition hover:border-arena-cyan"
                  :disabled="globalConnectivityTesting"
                  @click="runGlobalConnectivityTest()"
                >
                  {{ globalConnectivityTesting ? '测试中...' : '一键测试连通性' }}
                </button>
                <button
                  class="rounded border border-emerald-500/60 bg-emerald-900/30 px-2 py-1 text-[11px] text-emerald-200 transition hover:border-emerald-300"
                  @click="restoreAllFromKnownGood()"
                >
                  恢复全部最近可用
                </button>
              </div>
            </div>
            <p class="mb-2 break-all text-[11px] text-slate-400">{{ globalConfigEnvFile || '.env' }}</p>
            <p v-if="globalConnectivityMessage" class="mb-2 text-[11px] text-slate-300">{{ globalConnectivityMessage }}</p>

            <div class="global-default-provider-grid">
              <label class="field global-default-provider-field">
                <span class="global-default-provider-label">DEFAULT_LLM_PROVIDER</span>
                <input v-model="globalModelConfig.default_llm_provider" type="text" class="select" />
              </label>
              <label class="field global-default-provider-field">
                <span class="global-default-provider-label">DEFAULT_EMBEDDING_PROVIDER</span>
                <input v-model="globalModelConfig.default_embedding_provider" type="text" class="select" />
              </label>
            </div>

            <details class="mt-2 rounded border border-slate-700/60 bg-slate-900/40 p-2" open>
              <summary class="cursor-pointer text-xs text-slate-200">Chat</summary>
              <div class="mt-2 flex items-center justify-end gap-2">
                <button class="rounded border border-slate-500/60 bg-slate-900/60 px-2 py-1 text-[11px] text-slate-100" :disabled="globalConnectivityTesting" @click="runGlobalConnectivityTest(['chat'])">测试 Chat</button>
                <button class="rounded border border-emerald-500/60 bg-emerald-900/30 px-2 py-1 text-[11px] text-emerald-200" :disabled="!knownGoodState('chat').exists" @click="restoreModuleFromKnownGood('chat')">恢复最近可用</button>
              </div>
              <p v-if="connectivityResult('chat')" class="mt-1 text-[11px]" :class="connectivityResult('chat')?.ok ? 'text-emerald-300' : 'text-red-300'">
                {{ connectivityResult('chat')?.ok ? '可用' : '不可用' }} | {{ connectivityResult('chat')?.provider }}/{{ connectivityResult('chat')?.model }} | {{ connectivityResult('chat')?.error || connectivityResult('chat')?.note }}
              </p>
              <p v-if="knownGoodState('chat').exists" class="mt-1 text-[11px]" :class="knownGoodState('chat').matched ? 'text-emerald-300' : 'text-arena-amber'">
                最近可用配置：{{ knownGoodState('chat').matched ? '与当前一致' : '已变更（建议重测）' }} | {{ knownGoodState('chat').at }}
              </p>
              <div class="mt-2 grid grid-cols-1 gap-2 lg:grid-cols-2">
                <label class="field"><span>CHAT_LLM_PROVIDER</span><input v-model="globalModelConfig.chat_llm_provider" type="text" class="select" /></label>
                <label class="field"><span>CHAT_API_MODEL</span><input v-model="globalModelConfig.chat_api_model" type="text" class="select" /></label>
                <label class="field lg:col-span-2"><span>CHAT_API_BASE_URL</span><input v-model="globalModelConfig.chat_api_base_url" type="text" class="select" /></label>
                <label class="field lg:col-span-2"><span>CHAT_API_KEY</span><input v-model="globalModelConfig.chat_api_key" type="password" class="select" /></label>
                <label class="field"><span>CHAT_OLLAMA_BASE_URL</span><input v-model="globalModelConfig.chat_ollama_base_url" type="text" class="select" /></label>
                <label class="field"><span>CHAT_OLLAMA_MODEL</span><input v-model="globalModelConfig.chat_ollama_model" type="text" class="select" /></label>
                <label class="field lg:col-span-2"><span>CHAT_LOCAL_MODEL_PATH</span><input v-model="globalModelConfig.chat_local_model_path" type="text" class="select" /></label>
              </div>
            </details>

            <details class="mt-2 rounded border border-slate-700/60 bg-slate-900/40 p-2">
              <summary class="cursor-pointer text-xs text-slate-200">Judge</summary>
              <div class="mt-2 flex items-center justify-end gap-2">
                <button class="rounded border border-slate-500/60 bg-slate-900/60 px-2 py-1 text-[11px] text-slate-100" :disabled="globalConnectivityTesting" @click="runGlobalConnectivityTest(['judge'])">测试 Judge</button>
                <button class="rounded border border-emerald-500/60 bg-emerald-900/30 px-2 py-1 text-[11px] text-emerald-200" :disabled="!knownGoodState('judge').exists" @click="restoreModuleFromKnownGood('judge')">恢复最近可用</button>
              </div>
              <p v-if="connectivityResult('judge')" class="mt-1 text-[11px]" :class="connectivityResult('judge')?.ok ? 'text-emerald-300' : 'text-red-300'">
                {{ connectivityResult('judge')?.ok ? '可用' : '不可用' }} | {{ connectivityResult('judge')?.provider }}/{{ connectivityResult('judge')?.model }} | {{ connectivityResult('judge')?.error || connectivityResult('judge')?.note }}
              </p>
              <p v-if="knownGoodState('judge').exists" class="mt-1 text-[11px]" :class="knownGoodState('judge').matched ? 'text-emerald-300' : 'text-arena-amber'">
                最近可用配置：{{ knownGoodState('judge').matched ? '与当前一致' : '已变更（建议重测）' }} | {{ knownGoodState('judge').at }}
              </p>
              <div class="mt-2 grid grid-cols-1 gap-2 lg:grid-cols-2">
                <label class="field"><span>JUDGE_LLM_PROVIDER</span><input v-model="globalModelConfig.judge_llm_provider" type="text" class="select" /></label>
                <label class="field"><span>JUDGE_API_MODEL</span><input v-model="globalModelConfig.judge_api_model" type="text" class="select" /></label>
                <label class="field lg:col-span-2"><span>JUDGE_API_BASE_URL</span><input v-model="globalModelConfig.judge_api_base_url" type="text" class="select" /></label>
                <label class="field lg:col-span-2"><span>JUDGE_API_KEY</span><input v-model="globalModelConfig.judge_api_key" type="password" class="select" /></label>
                <label class="field"><span>JUDGE_OLLAMA_BASE_URL</span><input v-model="globalModelConfig.judge_ollama_base_url" type="text" class="select" /></label>
                <label class="field"><span>JUDGE_OLLAMA_MODEL</span><input v-model="globalModelConfig.judge_ollama_model" type="text" class="select" /></label>
                <label class="field lg:col-span-2"><span>JUDGE_LOCAL_MODEL_PATH</span><input v-model="globalModelConfig.judge_local_model_path" type="text" class="select" /></label>
              </div>
            </details>

            <details class="mt-2 rounded border border-slate-700/60 bg-slate-900/40 p-2">
              <summary class="cursor-pointer text-xs text-slate-200">Summarizer</summary>
              <div class="mt-2 flex items-center justify-end gap-2">
                <button class="rounded border border-slate-500/60 bg-slate-900/60 px-2 py-1 text-[11px] text-slate-100" :disabled="globalConnectivityTesting" @click="runGlobalConnectivityTest(['summarizer'])">测试 Summarizer</button>
                <button class="rounded border border-emerald-500/60 bg-emerald-900/30 px-2 py-1 text-[11px] text-emerald-200" :disabled="!knownGoodState('summarizer').exists" @click="restoreModuleFromKnownGood('summarizer')">恢复最近可用</button>
              </div>
              <p v-if="connectivityResult('summarizer')" class="mt-1 text-[11px]" :class="connectivityResult('summarizer')?.ok ? 'text-emerald-300' : 'text-red-300'">
                {{ connectivityResult('summarizer')?.ok ? '可用' : '不可用' }} | {{ connectivityResult('summarizer')?.provider }}/{{ connectivityResult('summarizer')?.model }} | {{ connectivityResult('summarizer')?.error || connectivityResult('summarizer')?.note }}
              </p>
              <p v-if="knownGoodState('summarizer').exists" class="mt-1 text-[11px]" :class="knownGoodState('summarizer').matched ? 'text-emerald-300' : 'text-arena-amber'">
                最近可用配置：{{ knownGoodState('summarizer').matched ? '与当前一致' : '已变更（建议重测）' }} | {{ knownGoodState('summarizer').at }}
              </p>
              <div class="mt-2 grid grid-cols-1 gap-2 lg:grid-cols-2">
                <label class="field"><span>SUMMARIZER_LLM_PROVIDER</span><input v-model="globalModelConfig.summarizer_llm_provider" type="text" class="select" /></label>
                <label class="field"><span>SUMMARIZER_API_MODEL</span><input v-model="globalModelConfig.summarizer_api_model" type="text" class="select" /></label>
                <label class="field lg:col-span-2"><span>SUMMARIZER_API_BASE_URL</span><input v-model="globalModelConfig.summarizer_api_base_url" type="text" class="select" /></label>
                <label class="field lg:col-span-2"><span>SUMMARIZER_API_KEY</span><input v-model="globalModelConfig.summarizer_api_key" type="password" class="select" /></label>
              </div>
            </details>

            <details class="mt-2 rounded border border-slate-700/60 bg-slate-900/40 p-2">
              <summary class="cursor-pointer text-xs text-slate-200">Entity</summary>
              <div class="mt-2 flex items-center justify-end gap-2">
                <button class="rounded border border-slate-500/60 bg-slate-900/60 px-2 py-1 text-[11px] text-slate-100" :disabled="globalConnectivityTesting" @click="runGlobalConnectivityTest(['entity'])">测试 Entity</button>
                <button class="rounded border border-emerald-500/60 bg-emerald-900/30 px-2 py-1 text-[11px] text-emerald-200" :disabled="!knownGoodState('entity').exists" @click="restoreModuleFromKnownGood('entity')">恢复最近可用</button>
              </div>
              <p v-if="connectivityResult('entity')" class="mt-1 text-[11px]" :class="connectivityResult('entity')?.ok ? 'text-emerald-300' : 'text-red-300'">
                {{ connectivityResult('entity')?.ok ? '可用' : '不可用' }} | {{ connectivityResult('entity')?.provider }}/{{ connectivityResult('entity')?.model }} | {{ connectivityResult('entity')?.error || connectivityResult('entity')?.note }}
              </p>
              <p v-if="knownGoodState('entity').exists" class="mt-1 text-[11px]" :class="knownGoodState('entity').matched ? 'text-emerald-300' : 'text-arena-amber'">
                最近可用配置：{{ knownGoodState('entity').matched ? '与当前一致' : '已变更（建议重测）' }} | {{ knownGoodState('entity').at }}
              </p>
              <div class="mt-2 grid grid-cols-1 gap-2 lg:grid-cols-2">

                <label class="field"><span>ENTITY_LLM_PROVIDER</span><input v-model="globalModelConfig.entity_llm_provider" type="text" class="select" /></label>
                <label class="field"><span>ENTITY_API_MODEL</span><input v-model="globalModelConfig.entity_api_model" type="text" class="select" /></label>
                <label class="field lg:col-span-2"><span>ENTITY_API_BASE_URL</span><input v-model="globalModelConfig.entity_api_base_url" type="text" class="select" /></label>
                <label class="field lg:col-span-2"><span>ENTITY_API_KEY</span><input v-model="globalModelConfig.entity_api_key" type="password" class="select" /></label>
              </div>
            </details>

            <details class="mt-2 rounded border border-slate-700/60 bg-slate-900/40 p-2">
              <summary class="cursor-pointer text-xs text-slate-200">Reflector</summary>
              <div class="mt-2 flex items-center justify-end gap-2">
                <button class="rounded border border-slate-500/60 bg-slate-900/60 px-2 py-1 text-[11px] text-slate-100" :disabled="globalConnectivityTesting" @click="runGlobalConnectivityTest(['reflector'])">测试 Reflector</button>
                <button class="rounded border border-emerald-500/60 bg-emerald-900/30 px-2 py-1 text-[11px] text-emerald-200" :disabled="!knownGoodState('reflector').exists" @click="restoreModuleFromKnownGood('reflector')">恢复最近可用</button>
              </div>
              <p v-if="connectivityResult('reflector')" class="mt-1 text-[11px]" :class="connectivityResult('reflector')?.ok ? 'text-emerald-300' : 'text-red-300'">
                {{ connectivityResult('reflector')?.ok ? '可用' : '不可用' }} | {{ connectivityResult('reflector')?.provider }}/{{ connectivityResult('reflector')?.model }} | {{ connectivityResult('reflector')?.error || connectivityResult('reflector')?.note }}
              </p>
              <p v-if="knownGoodState('reflector').exists" class="mt-1 text-[11px]" :class="knownGoodState('reflector').matched ? 'text-emerald-300' : 'text-arena-amber'">
                最近可用配置：{{ knownGoodState('reflector').matched ? '与当前一致' : '已变更（建议重测）' }} | {{ knownGoodState('reflector').at }}
              </p>
              <div class="mt-2 grid grid-cols-1 gap-2 lg:grid-cols-2">

                <label class="field"><span>REFLECTOR_LLM_PROVIDER</span><input v-model="globalModelConfig.reflector_llm_provider" type="text" class="select" /></label>
                <label class="field"><span>REFLECTOR_API_MODEL</span><input v-model="globalModelConfig.reflector_api_model" type="text" class="select" /></label>
                <label class="field lg:col-span-2"><span>REFLECTOR_API_BASE_URL</span><input v-model="globalModelConfig.reflector_api_base_url" type="text" class="select" /></label>
                <label class="field lg:col-span-2"><span>REFLECTOR_API_KEY</span><input v-model="globalModelConfig.reflector_api_key" type="password" class="select" /></label>
              </div>
            </details>

            <details class="mt-2 rounded border border-slate-700/60 bg-slate-900/40 p-2">
              <summary class="cursor-pointer text-xs text-slate-200">Embedding / Local Device</summary>
              <div class="mt-2 flex items-center justify-end gap-2">
                <button class="rounded border border-slate-500/60 bg-slate-900/60 px-2 py-1 text-[11px] text-slate-100" :disabled="globalConnectivityTesting" @click="runGlobalConnectivityTest(['embedding'])">测试 Embedding</button>
                <button class="rounded border border-emerald-500/60 bg-emerald-900/30 px-2 py-1 text-[11px] text-emerald-200" :disabled="!knownGoodState('embedding').exists" @click="restoreModuleFromKnownGood('embedding')">恢复最近可用</button>
              </div>
              <p v-if="connectivityResult('embedding')" class="mt-1 text-[11px]" :class="connectivityResult('embedding')?.ok ? 'text-emerald-300' : 'text-red-300'">
                {{ connectivityResult('embedding')?.ok ? '可用' : '不可用' }} | {{ connectivityResult('embedding')?.provider }}/{{ connectivityResult('embedding')?.model }} | {{ connectivityResult('embedding')?.error || connectivityResult('embedding')?.note }}
              </p>
              <p v-if="knownGoodState('embedding').exists" class="mt-1 text-[11px]" :class="knownGoodState('embedding').matched ? 'text-emerald-300' : 'text-arena-amber'">
                最近可用配置：{{ knownGoodState('embedding').matched ? '与当前一致' : '已变更（建议重测）' }} | {{ knownGoodState('embedding').at }}
              </p>
              <div class="mt-2 grid grid-cols-1 gap-2 lg:grid-cols-2">
                <label class="field"><span>EMBEDDING_PROVIDER</span><input v-model="globalModelConfig.embedding_provider" type="text" class="select" /></label>
                <label class="field"><span>EMBEDDING_API_MODEL</span><input v-model="globalModelConfig.embedding_api_model" type="text" class="select" /></label>
                <label class="field lg:col-span-2"><span>EMBEDDING_API_BASE_URL</span><input v-model="globalModelConfig.embedding_api_base_url" type="text" class="select" /></label>
                <label class="field lg:col-span-2"><span>EMBEDDING_API_KEY</span><input v-model="globalModelConfig.embedding_api_key" type="password" class="select" /></label>
                <label class="field"><span>EMBEDDING_OLLAMA_BASE_URL</span><input v-model="globalModelConfig.embedding_ollama_base_url" type="text" class="select" /></label>
                <label class="field"><span>EMBEDDING_OLLAMA_MODEL</span><input v-model="globalModelConfig.embedding_ollama_model" type="text" class="select" /></label>
                <label class="field lg:col-span-2"><span>EMBEDDING_LOCAL_MODEL_PATH</span><input v-model="globalModelConfig.embedding_local_model_path" type="text" class="select" /></label>
                <label class="field"><span>LOCAL_INFER_DEVICE</span><input v-model="globalModelConfig.local_infer_device" type="text" class="select" /></label>
              </div>
            </details>

            <div class="mt-2 flex items-center gap-2">
              <button
                class="rounded-lg bg-arena-amber px-3 py-1 text-xs font-semibold text-slate-900"
                :disabled="globalConfigSaving"
                @click="saveGlobalModelConfigPanel"
              >
                {{ globalConfigSaving ? '保存中...' : '保存到 .env' }}
              </button>
              <span v-if="globalConfigMessage" class="text-[11px] text-slate-300">{{ globalConfigMessage }}</span>
            </div>
          </div>

          <details v-if="config.engine === 'VectorEngine'" class="mt-2 rounded-lg border border-slate-600/60 bg-slate-900/40 p-3">
            <summary class="cursor-pointer text-xs font-semibold text-arena-mint">Vector Retrieval Params (Chroma)</summary>
            <label class="field">
              <span>Top K</span>
              <input v-model.number="retrievalTopK" type="number" min="1" max="50" class="select" />
            </label>
            <label class="field mt-2">
              <span>Min Relevance (0~1)</span>
              <input v-model.number="minRelevance" type="number" min="0" max="1" step="0.01" class="select" />
            </label>
            <label class="field mt-2">
              <span>Collection Name</span>
              <input v-model="collectionName" type="text" class="select" />
            </label>
            <label class="field mt-2">
              <span>Similarity Strategy</span>
              <select v-model="similarityStrategy" class="select">
                <option value="inverse_distance">inverse_distance</option>
                <option value="exp_decay">exp_decay</option>
                <option value="linear">linear</option>
              </select>
            </label>
            <label class="mt-2 flex items-center gap-2 text-sm text-slate-200">
              <input v-model="keywordRerank" type="checkbox" />
              <span>Enable Keyword Rerank (0.3 blend)</span>
            </label>
          </details>

          <details class="mt-2 rounded-lg border border-slate-600/60 bg-slate-900/40 p-3">
            <summary class="cursor-pointer text-xs font-semibold text-arena-mint">Context Assembly Params</summary>
            <label class="field">
              <span>Max Context Tokens (RankedPruning)</span>
              <input v-model.number="maxContextTokens" type="number" min="64" max="8192" class="select" />
            </label>
            <label class="field mt-2">
              <span>Reasoning Hops (ReasoningChain + GraphEngine)</span>
              <input v-model.number="reasoningHops" type="number" min="1" max="3" class="select" />
            </label>
          </details>

          <details class="mt-2 rounded-lg border border-slate-600/60 bg-slate-900/40 p-3">
            <summary class="cursor-pointer text-xs font-semibold text-arena-mint">Short-term Memory Params</summary>
            <label class="field">
              <span>Short-term Mode</span>
              <select v-model="shortTermMode" class="select">
                <option v-for="x in shortTermModes" :key="x" :value="x">{{ x }}</option>
              </select>
            </label>
            <label v-if="showStmWindow" class="field mt-2">
              <span>Sliding Window Turns</span>
              <input v-model.number="stmWindowTurns" type="number" min="1" max="30" class="select" />
            </label>
            <label v-if="showStmTokenBudget" class="field mt-2">
              <span>Token Buffer Budget</span>
              <input v-model.number="stmTokenBudget" type="number" min="128" max="16000" class="select" />
            </label>
            <label v-if="showStmRolling" class="field mt-2">
              <span>Rolling Summary Keep Recent Turns</span>
              <input v-model.number="stmSummaryKeepRecentTurns" type="number" min="1" max="12" class="select" />
            </label>
            <p class="mt-2 text-xs text-slate-400">
              LTM 负责长期检索；STM 负责最近上下文焦点，二者会在后端合并排序后再交给 Assembler。
            </p>
          </details>

          <details class="mt-2 rounded-lg border border-slate-600/60 bg-slate-900/40 p-3">
            <summary class="cursor-pointer text-xs font-semibold text-arena-mint">Reflector Writeback</summary>
            <label class="field">
              <span>Reflector LLM Mode</span>
              <select v-model="reflectorLlmMode" class="select">
                <option v-for="x in reflectorLlmModes" :key="x" :value="x">{{ x }}</option>
              </select>
            </label>
            <label class="flex items-center gap-2 text-sm text-slate-200">
              <input v-model="reflectorAutoWriteback" type="checkbox" />
              <span>Enable Auto Writeback (ConflictResolver)</span>
            </label>
            <label class="field mt-2">
              <span>Writeback Min Confidence</span>
              <input
                v-model.number="reflectorWritebackMinConfidence"
                type="number"
                min="0"
                max="1"
                step="0.05"
                class="select"
              />
            </label>
            <p class="mt-2 text-xs text-slate-400">
              自动写回会向会话写入控制块，用于抑制被裁决为过时的冲突值检索分数；建议仅在纠错场景开启。
            </p>
          </details>
        </div>
      </section>

      <section class="panel lg:col-span-1">
        <h2 class="panel-title">Dataset Area</h2>
        <div class="space-y-3">
          <label class="field">
            <span>单条输入</span>
            <textarea v-model="inputText" rows="4" class="textarea" />
          </label>
          <label class="field">
            <span>Expected Facts (每行一条)</span>
            <textarea v-model="expectedFactsRaw" rows="4" class="textarea" />
          </label>
          <label class="field">
            <span>上传 JSON 测试集 (可选)</span>
            <input type="file" accept="application/json" class="input-file" @change="handleDatasetUpload" />
          </label>
          <label class="field">
            <span>Batch Case Count（未上传 JSON 时生效）</span>
            <input v-model.number="batchCaseCount" type="number" min="1" max="200" class="select" />
          </label>
          <p class="text-xs text-slate-400">
            未上传 JSON 时，Run Batch 会使用“单条输入 + Expected Facts”自动生成 N 条测试样本。
          </p>

          <div class="rounded-lg border border-slate-600/60 bg-slate-900/40 p-3">
            <div class="mb-2 flex items-center justify-between gap-2">
              <p class="text-xs font-semibold text-arena-mint">内置数据集</p>
              <button
                class="rounded border border-slate-500/60 bg-slate-900/60 px-2 py-1 text-[11px] text-slate-100 transition hover:border-slate-300"
                :disabled="builtinDatasetsLoading"
                @click="loadBuiltinDatasets(true)"
              >
                {{ builtinDatasetsLoading ? '刷新中...' : '刷新数据集列表' }}
              </button>
            </div>
            <p v-if="builtinDatasetsMessage" class="mb-2 text-[11px] text-slate-300">{{ builtinDatasetsMessage }}</p>
            <label class="field">
              <span>Dataset</span>
              <select v-model="selectedDatasetName" class="select" :disabled="builtinDatasetsLoading">
                <option v-for="d in builtinDatasets" :key="d.name" :value="d.name">
                  {{ d.name }} ({{ d.count >= 0 ? d.count : '?' }})
                </option>
              </select>
            </label>
            <div class="mt-2 grid grid-cols-2 gap-2">
              <label class="field">
                <span>Sample Size</span>
                <input v-model.number="datasetSampleSize" type="number" min="1" class="select" />
              </label>
              <label class="field">
                <span>Start Index</span>
                <input v-model.number="datasetStartIndex" type="number" min="0" class="select" />
              </label>
            </div>
            <label class="mt-2 flex items-center gap-2 text-sm text-slate-200">
              <input v-model="isolateSessions" type="checkbox" />
              <span>每个 case 独立会话（推荐，批量内 case 之间互不串扰）</span>
            </label>
            <label class="field mt-2">
              <span>Batch Max Concurrency</span>
              <input v-model.number="maxConcurrency" type="number" min="1" max="32" class="select" />
            </label>
            <label class="field mt-2">
              <span>请求超时 (ms)</span>
              <input v-model.number="requestTimeoutMs" type="number" min="5000" step="1000" class="select" />
            </label>
          </div>
        </div>

        <div class="mt-5">
          <p class="mb-2 text-sm font-semibold text-slate-300">Execution</p>
          <p class="mb-2 rounded-lg border border-slate-600/60 bg-slate-900/50 p-2 text-xs text-slate-300">
            Run Batch 输入模式：{{ batchInputModeLabel }}，本次将运行 {{ plannedBatchCaseCount }} 条 case。
          </p>
          <div class="grid grid-cols-1 gap-2 sm:grid-cols-3">
            <button class="run-btn w-full" :disabled="loading" @click="onRunBenchmark">
              {{ loading ? 'Running...' : 'Run Benchmark' }}
            </button>
            <button class="run-btn w-full" :disabled="loading" @click="onRunBatchBenchmark">
              {{ loading ? 'Running...' : 'Run Batch' }}
            </button>
            <button class="run-btn w-full" :disabled="loading" @click="onRunBuiltinDataset">
              {{ loading ? 'Running...' : 'Run Built-in Dataset' }}
            </button>
          </div>
        </div>

        <p v-if="error" class="mt-3 rounded-lg border border-red-500/40 bg-red-500/10 p-2 text-sm text-red-200">
          {{ error }}
        </p>
        <div
          v-if="entityHealthState"
          class="mt-3 rounded-lg border p-2"
          :class="
            entityHealthState.level === 'warning'
              ? 'border-arena-amber/50 bg-arena-amber/10 text-arena-amber'
              : 'border-arena-cyan/50 bg-arena-cyan/10 text-arena-mint'
          "
        >
          <p class="text-sm">{{ entityHealthState.message }}</p>
          <p class="mt-1 text-xs opacity-90">{{ entityHealthState.details }}</p>
          <div class="mt-2 flex items-center gap-2">
            <button
              type="button"
              class="rounded border border-slate-500/60 bg-slate-900/60 px-2 py-1 text-xs text-slate-100 transition hover:border-slate-300"
              @click="copyEntityDiagnostic"
            >
              复制诊断信息
            </button>
            <span v-if="entityDiagnosticCopyFeedback" class="text-xs text-slate-200/90">{{ entityDiagnosticCopyFeedback }}</span>
          </div>
          <details class="mt-2 rounded border border-slate-600/50 bg-slate-950/40 p-2 text-xs text-slate-100">
            <summary class="cursor-pointer select-none text-slate-200">查看最近 3 条失败事件</summary>
            <ul class="mt-2 list-disc space-y-1 pl-4">
              <li v-for="(evt, idx) in entityHealthState.recentFailures" :key="`${evt.ts}-${idx}`">
                <span class="text-slate-300">{{ evt.ts }}</span>
                <span> | {{ evt.purpose }} | {{ evt.provider }}/{{ evt.model }} | status={{ evt.status }}</span>
                <div class="text-slate-400">{{ evt.error }}</div>
              </li>
            </ul>
          </details>
        </div>
        <p v-if="loading && progressText" class="mt-3 rounded-lg border border-arena-cyan/40 bg-arena-cyan/10 p-2 text-sm text-arena-mint">
          {{ progressText }}
        </p>
        <p v-if="loading" class="mt-2 rounded-lg border border-arena-amber/40 bg-arena-amber/10 p-2 text-sm text-arena-amber">
          Elapsed: {{ runningDurationLabel }}
        </p>
        <p v-else-if="lastRunDurationMs !== null" class="mt-2 rounded-lg border border-slate-500/40 bg-slate-800/60 p-2 text-sm text-slate-200">
          Last Run Duration: {{ finishedDurationLabel }}
        </p>
      </section>

      <section class="panel lg:col-span-1">
        <h2 class="panel-title">Results Dashboard</h2>
        <p
          class="mb-3 rounded-lg border border-slate-600/60 bg-slate-900/50 p-2 text-xs text-slate-300"
          title="LLM Judge: Precision/Faithfulness/InfoLoss。&#10;规则法: Recall@K、QA Accuracy/F1、Consistency、Rejection/Rejection@Unknown、Convergence Speed、Context Distraction。"
        >
          评测口径：Precision/Faithfulness/InfoLoss 由 LLM Judge 给分；Recall@K、QA Accuracy/F1、Consistency、Rejection/Rejection@Unknown、Convergence Speed、Context Distraction 为规则法自动评估。
        </p>
        <label class="mb-3 flex items-center gap-2 rounded-lg border border-slate-600/60 bg-slate-900/40 p-2 text-xs text-slate-200">
          <input v-model="includeRawJudgeInMarkdown" type="checkbox" />
          <span>导出 .md 时包含 Raw Judge Output</span>
        </label>

        <div v-if="result" class="space-y-4">
          <MetricBars :metrics="result.eval_result.metrics" />

          <div v-if="singleSafetySignals" class="rounded-xl border border-slate-600/60 bg-slate-900/60 p-3">
            <h3 class="mb-2 text-sm font-semibold text-arena-mint">Safety Signals: Behavior vs Correctness</h3>
            <div class="grid grid-cols-1 gap-2 sm:grid-cols-2">
              <div class="rounded-lg border border-slate-700/60 bg-slate-900/40 p-2">
                <p class="text-xs font-semibold text-slate-200">Rejection Rate（行为率）</p>
                <p class="mt-1 text-xs text-arena-mint">
                  {{ singleSafetySignals.rejectionRate == null ? 'N/A' : `${(singleSafetySignals.rejectionRate * 100).toFixed(1)}%` }}
                </p>
              </div>
              <div class="rounded-lg border border-slate-700/60 bg-slate-900/40 p-2">
                <p class="text-xs font-semibold text-slate-200">Rejection@Unknown（正确率）</p>
                <p class="mt-1 text-xs text-arena-mint">
                  {{ singleSafetySignals.rejectionUnknownCorrectness == null ? 'N/A（仅未知样本）' : `${(singleSafetySignals.rejectionUnknownCorrectness * 100).toFixed(1)}%` }}
                </p>
              </div>
            </div>
          </div>

          <div class="rounded-xl border border-slate-600/60 bg-slate-900/60 p-3">
            <h3 class="mb-2 text-sm font-semibold text-arena-mint">Additional Signals</h3>
            <div class="space-y-2">
              <div v-for="item in singleDerivedRows" :key="item.label" class="rounded-lg border border-slate-700/60 bg-slate-900/40 p-2">
                <div class="flex items-center justify-between">
                  <span class="text-xs font-semibold text-slate-200">{{ item.label }}</span>
                  <span class="text-xs text-arena-mint">{{ item.value }}</span>
                </div>
                <p class="mt-1 text-[11px] text-slate-400">{{ item.hint }}</p>
              </div>
            </div>
          </div>

          <div class="rounded-xl border border-slate-600/60 bg-slate-900/60 p-3">
            <h3 class="mb-2 text-sm font-semibold text-arena-mint">Prompt Preview</h3>
            <pre class="max-h-60 overflow-auto text-xs text-slate-200">{{ result.assemble_result.prompt }}</pre>
          </div>

          <div class="rounded-xl border border-slate-600/60 bg-slate-900/60 p-3">
            <h3 class="mb-2 text-sm font-semibold text-arena-mint">Module IO Trace</h3>
            <p class="mb-2 text-[11px] text-slate-400">
              逐模块输入输出追踪（Processor/Engine/Assembler/Chat/Eval/Reflector/STM/LLM Calls），用于快速定位异常模块。
            </p>
            <pre class="max-h-72 overflow-auto text-xs text-slate-200">{{ singleModuleTraceJson }}</pre>
          </div>

          <div class="rounded-xl border border-slate-600/60 bg-slate-900/60 p-3">
            <h3 class="mb-2 text-sm font-semibold text-arena-mint">Prompt Quality Checks</h3>
            <ul class="space-y-2 text-xs text-slate-200">
              <li
                v-for="(row, idx) in singlePromptQualityChecks"
                :key="`prompt-check-${idx}-${row.purpose}`"
                class="rounded border border-slate-700/60 bg-slate-900/40 p-2"
              >
                <div class="flex items-center justify-between gap-2">
                  <span class="font-semibold">{{ row.purpose }} | {{ row.provider }}/{{ row.model }}</span>
                  <span
                    class="rounded px-2 py-0.5 text-[10px] font-semibold"
                    :class="
                      row.severity === 'error'
                        ? 'bg-red-500/20 text-red-200'
                        : row.severity === 'warn'
                        ? 'bg-arena-amber/20 text-arena-amber'
                        : 'bg-emerald-500/20 text-emerald-200'
                    "
                  >
                    {{ row.severity.toUpperCase() }}
                  </span>
                </div>
                <p class="mt-1">{{ row.summary }}</p>
                <p class="mt-1 text-[11px] text-slate-400">{{ row.details }}</p>
              </li>
              <li v-if="singlePromptQualityChecks.length === 0" class="text-slate-400">(none)</li>
            </ul>
          </div>

          <div v-if="singleRoleGroupedHits" class="rounded-xl border border-slate-600/60 bg-slate-900/60 p-3">
            <h3 class="mb-2 text-sm font-semibold text-arena-mint">Retrieved Memory by Role</h3>
            <div class="grid grid-cols-1 gap-3 md:grid-cols-2">
              <div class="rounded-lg border border-slate-700/60 bg-slate-900/40 p-2">
                <p class="text-xs font-semibold text-slate-200">STM / User</p>
                <ul class="mt-1 max-h-28 list-disc overflow-auto pl-4 text-xs text-slate-300">
                  <li v-for="(h, i) in singleRoleGroupedHits.stm.user" :key="`stm-u-${i}`">[{{ h.relevance.toFixed(4) }}] {{ h.content }}</li>
                  <li v-if="singleRoleGroupedHits.stm.user.length === 0">(none)</li>
                </ul>
                <p class="mt-2 text-xs font-semibold text-slate-200">STM / Assistant</p>
                <ul class="mt-1 max-h-28 list-disc overflow-auto pl-4 text-xs text-slate-300">
                  <li v-for="(h, i) in singleRoleGroupedHits.stm.assistant" :key="`stm-a-${i}`">[{{ h.relevance.toFixed(4) }}] {{ h.content }}</li>
                  <li v-if="singleRoleGroupedHits.stm.assistant.length === 0">(none)</li>
                </ul>
                <p class="mt-2 text-xs font-semibold text-slate-200">STM / Other</p>
                <ul class="mt-1 max-h-28 list-disc overflow-auto pl-4 text-xs text-slate-300">
                  <li v-for="(h, i) in singleRoleGroupedHits.stm.other" :key="`stm-o-${i}`">[{{ h.relevance.toFixed(4) }}] {{ h.content }}</li>
                  <li v-if="singleRoleGroupedHits.stm.other.length === 0">(none)</li>
                </ul>
              </div>

              <div class="rounded-lg border border-slate-700/60 bg-slate-900/40 p-2">
                <p class="text-xs font-semibold text-slate-200">LTM / User</p>
                <ul class="mt-1 max-h-28 list-disc overflow-auto pl-4 text-xs text-slate-300">
                  <li v-for="(h, i) in singleRoleGroupedHits.ltm.user" :key="`ltm-u-${i}`">[{{ h.relevance.toFixed(4) }}] {{ h.content }}</li>
                  <li v-if="singleRoleGroupedHits.ltm.user.length === 0">(none)</li>
                </ul>
                <p class="mt-2 text-xs font-semibold text-slate-200">LTM / Assistant</p>
                <ul class="mt-1 max-h-28 list-disc overflow-auto pl-4 text-xs text-slate-300">
                  <li v-for="(h, i) in singleRoleGroupedHits.ltm.assistant" :key="`ltm-a-${i}`">[{{ h.relevance.toFixed(4) }}] {{ h.content }}</li>
                  <li v-if="singleRoleGroupedHits.ltm.assistant.length === 0">(none)</li>
                </ul>
                <p class="mt-2 text-xs font-semibold text-slate-200">LTM / Other</p>
                <ul class="mt-1 max-h-28 list-disc overflow-auto pl-4 text-xs text-slate-300">
                  <li v-for="(h, i) in singleRoleGroupedHits.ltm.other" :key="`ltm-o-${i}`">[{{ h.relevance.toFixed(4) }}] {{ h.content }}</li>
                  <li v-if="singleRoleGroupedHits.ltm.other.length === 0">(none)</li>
                </ul>
              </div>
            </div>
          </div>

          <div v-if="singleReasoningChains.length > 0" class="rounded-xl border border-slate-600/60 bg-slate-900/60 p-3">
            <h3 class="mb-2 text-sm font-semibold text-arena-mint">Reasoning Chains</h3>
            <p v-if="singleReasoningSeeds.length > 0" class="mb-2 text-[11px] text-slate-400">
              Seed Entities: {{ singleReasoningSeeds.join(', ') }}
            </p>
            <div v-if="singleReasoningChainDetails.length > 0" class="mb-2 overflow-auto rounded-md border border-slate-700/60 bg-slate-900/40">
              <table class="min-w-full text-[11px] text-slate-200">
                <thead class="bg-slate-800/70 text-slate-300">
                  <tr>
                    <th class="px-2 py-1 text-left">Priority</th>
                    <th class="px-2 py-1 text-left">Hop</th>
                    <th class="px-2 py-1 text-left">Seed</th>
                    <th class="px-2 py-1 text-left">Overlap</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(d, idx) in singleReasoningChainDetails" :key="`detail-${idx}-${d.chain}`" class="border-t border-slate-800/60">
                    <td class="px-2 py-1">{{ d.priority.toFixed(3) }}</td>
                    <td class="px-2 py-1">{{ d.hop }}</td>
                    <td class="px-2 py-1">{{ d.seed_touch ? 'Y' : 'N' }}</td>
                    <td class="px-2 py-1">{{ (d.lexical_overlap * 100).toFixed(1) }}%</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <ul class="max-h-40 list-disc space-y-1 overflow-auto pl-5 text-xs text-slate-200">
              <li v-for="(chain, idx) in singleReasoningChains" :key="`${idx}-${chain}`">{{ chain }}</li>
            </ul>
          </div>

          <div class="rounded-xl border border-slate-600/60 bg-slate-900/60 p-3">
            <h3 class="mb-2 text-sm font-semibold text-arena-mint">Judge Rationale</h3>
            <p class="text-xs text-slate-200">{{ result.eval_result.judge_rationale }}</p>
          </div>

          <div class="rounded-xl border border-slate-600/60 bg-slate-900/60 p-3">
            <h3 class="mb-2 text-sm font-semibold text-arena-mint">Raw Judge Output</h3>
            <pre class="max-h-48 overflow-auto text-xs text-slate-200">{{ result.eval_result.raw_judge_output || 'N/A' }}</pre>
          </div>

          <div class="rounded-xl border border-slate-600/60 bg-slate-900/60 p-3">
            <h3 class="mb-2 text-sm font-semibold text-arena-mint">Markdown Report Preview</h3>
            <div class="markdown-preview max-h-64 overflow-auto" v-html="renderedMarkdownReport"></div>
            <button class="mt-2 rounded-lg bg-arena-amber px-3 py-1 text-xs font-semibold text-slate-900" @click="downloadMarkdownReport">
              Download .md Report
            </button>
          </div>
        </div>

        <div v-else-if="batchResult" class="space-y-4">
          <MetricBars :metrics="batchResult.avg_metrics" />
          <div v-if="batchSafetySignals" class="rounded-xl border border-slate-600/60 bg-slate-900/60 p-3">
            <h3 class="mb-2 text-sm font-semibold text-arena-mint">Safety Signals: Behavior vs Correctness</h3>
            <div class="grid grid-cols-1 gap-2 sm:grid-cols-2">
              <div class="rounded-lg border border-slate-700/60 bg-slate-900/40 p-2">
                <p class="text-xs font-semibold text-slate-200">Avg Rejection Rate（行为率）</p>
                <p class="mt-1 text-xs text-arena-mint">
                  {{ batchSafetySignals.rejectionRate == null ? 'N/A' : `${(batchSafetySignals.rejectionRate * 100).toFixed(1)}%` }}
                </p>
              </div>
              <div class="rounded-lg border border-slate-700/60 bg-slate-900/40 p-2">
                <p class="text-xs font-semibold text-slate-200">Avg Rejection@Unknown（正确率）</p>
                <p class="mt-1 text-xs text-arena-mint">
                  {{ batchSafetySignals.rejectionUnknownCorrectness == null ? 'N/A（仅未知样本）' : `${(batchSafetySignals.rejectionUnknownCorrectness * 100).toFixed(1)}%` }}
                </p>
              </div>
            </div>
            <p class="mt-2 text-[11px] text-slate-400">
              样本分布：Unknown {{ batchSafetySignals.unknownCount }} / Known {{ batchSafetySignals.knownCount }}
            </p>
            <p class="mt-1 text-[11px] text-slate-400">
              Unknown 占比：{{ (batchSafetySignals.unknownRatio * 100).toFixed(1) }}%
            </p>
            <p class="mt-2 rounded-md border border-slate-700/60 bg-slate-900/40 p-2 text-[11px] text-slate-300">
              解读建议：{{ batchSafetyInterpretation }}
            </p>
          </div>
          <div class="rounded-xl border border-slate-600/60 bg-slate-900/60 p-3">
            <h3 class="mb-2 text-sm font-semibold text-arena-mint">Stability & Robustness</h3>
            <div class="space-y-2">
              <div v-for="item in batchDerivedRows" :key="item.label" class="rounded-lg border border-slate-700/60 bg-slate-900/40 p-2">
                <div class="flex items-center justify-between">
                  <span class="text-xs font-semibold text-slate-200">{{ item.label }}</span>
                  <span class="text-xs text-arena-mint">{{ item.value }}</span>
                </div>
                <p class="mt-1 text-[11px] text-slate-400">{{ item.hint }}</p>
              </div>
            </div>
          </div>
          <div class="rounded-xl border border-slate-600/60 bg-slate-900/60 p-3">
            <h3 class="mb-2 text-sm font-semibold text-arena-mint">Batch Summary</h3>
            <p class="text-xs text-slate-200">Run ID: {{ batchResult.run_id }}</p>
            <p class="text-xs text-slate-200">Cases: {{ batchResult.case_results.length }}</p>
            <button class="mt-2 rounded-lg bg-arena-amber px-3 py-1 text-xs font-semibold text-slate-900" @click="downloadCsvReport">
              Download CSV Report
            </button>
            <button class="ml-2 mt-2 rounded-lg bg-arena-amber px-3 py-1 text-xs font-semibold text-slate-900" @click="downloadMarkdownReport">
              Download .md Report
            </button>
            <button class="ml-2 mt-2 rounded-lg border border-slate-500/60 bg-slate-900/60 px-3 py-1 text-xs font-semibold text-slate-100" @click="copyBatchDiagnostic">
              复制批量诊断
            </button>
            <p v-if="batchDiagnosticCopyFeedback" class="mt-2 text-xs text-slate-300">{{ batchDiagnosticCopyFeedback }}</p>
          </div>

          <div class="rounded-xl border border-slate-600/60 bg-slate-900/60 p-3">
            <h3 class="mb-2 text-sm font-semibold text-arena-mint">Case Module IO Trace</h3>
            <div class="space-y-2">
              <details
                v-for="(c, idx) in batchResult.case_results"
                :key="`case-trace-${idx}-${c.run_id}`"
                class="rounded border border-slate-700/60 bg-slate-900/40 p-2"
              >
                <summary class="cursor-pointer text-xs text-slate-200">Case {{ idx + 1 }} | run_id={{ c.run_id }}</summary>
                <pre class="mt-2 max-h-60 overflow-auto text-xs text-slate-300">{{ asPrettyJson(c.module_trace || {}) }}</pre>
                <div class="mt-2 rounded border border-slate-700/60 bg-slate-900/30 p-2">
                  <p class="text-xs font-semibold text-arena-mint">Prompt Quality Checks</p>
                  <ul class="mt-1 space-y-1 text-[11px] text-slate-300">
                    <li
                      v-for="(row, ridx) in buildPromptQualityChecks((c.module_trace || {}) as Record<string, unknown>)"
                      :key="`case-prompt-check-${idx}-${ridx}-${row.purpose}`"
                    >
                      [{{ row.severity.toUpperCase() }}] {{ row.purpose }} | {{ row.provider }}/{{ row.model }} | {{ row.summary }}
                    </li>
                    <li v-if="buildPromptQualityChecks((c.module_trace || {}) as Record<string, unknown>).length === 0" class="text-slate-500">(none)</li>
                  </ul>
                </div>
              </details>
            </div>
          </div>

          <div class="rounded-xl border border-slate-600/60 bg-slate-900/60 p-3">
            <h3 class="mb-2 text-sm font-semibold text-arena-mint">Markdown Report Preview</h3>
            <div class="markdown-preview max-h-64 overflow-auto" v-html="renderedMarkdownReport"></div>
          </div>
        </div>

        <div v-else class="rounded-xl border border-dashed border-slate-500 p-4 text-sm text-slate-300">
          运行后将在此展示指标柱状图与 Prompt 拼装预览。
        </div>
      </section>
    </div>
  </div>
</template>
