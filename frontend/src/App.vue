<script setup lang="ts">
import MarkdownIt from 'markdown-it';
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import MetricBars from './components/MetricBars.vue';
import {
  getAsyncRunStatus,
  listDatasets,
  runBatchBenchmarkAsync,
  runBenchmarkWithTimeout,
  runDatasetBenchmarkAsync
} from './api/client';
import type {
  BatchBenchmarkRunResponse,
  BenchmarkConfig,
  BenchmarkRunResponse,
  DatasetSummary
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
const datasetCases = ref<Array<{ case_id: string; input_text: string; expected_facts: string[]; session_id: string }>>([]);
const builtinDatasets = ref<DatasetSummary[]>([]);
const selectedDatasetName = ref('');
const datasetSampleSize = ref(5);
const datasetStartIndex = ref(0);
const isolateSessions = ref(true);
const maxConcurrency = ref(3);
const batchCaseCount = ref(5);
const requestTimeoutMs = ref(120000);
const includeRawJudgeInMarkdown = ref(false);
const progressText = ref('');
const elapsedMs = ref(0);
const lastRunDurationMs = ref<number | null>(null);

let timerHandle: ReturnType<typeof setInterval> | null = null;
let runStartTs = 0;

const processors = ['RawLogger', 'Summarizer', 'EntityExtractor'] as const;
const engines = ['VectorEngine', 'GraphEngine', 'RelationalEngine'] as const;
const assemblers = ['SystemInjector', 'XMLTagging', 'TimelineRollover'] as const;
const reflectors = ['None', 'GenerativeReflection', 'ConflictResolver'] as const;
const providers = ['api', 'ollama', 'local'] as const;
const summarizerMethods = ['llm', 'kmeans'] as const;
const entityExtractorMethods = ['llm_triple', 'llm_attribute', 'spacy_llm_triple', 'spacy_llm_attribute'] as const;
const computeDevices = ['cpu', 'cuda'] as const;

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

const markdownReport = computed(() => {
  if (result.value) {
    const r = result.value;
    const memoryLines = r.search_result.hits.length
      ? r.search_result.hits
          .map((hit, idx) => `- [${idx + 1}] score=${hit.relevance.toFixed(4)}\n  - ${hit.content}`)
          .join('\n')
      : '- (none)';

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
      '## Agent Real-time Memory',
      memoryLines,
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
        const memoryLines = c.search_result.hits.length
          ? c.search_result.hits
              .map((hit, i) => `  - [${i + 1}] score=${hit.relevance.toFixed(4)}: ${hit.content}`)
              .join('\n')
          : '  - (none)';

        return [
          `### Case ${idx + 1}`,
          `- run_id: ${c.run_id}`,
          '- Agent Reply:',
          c.generated_response || '(empty)',
          '- Agent Real-time Memory:',
          memoryLines,
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

async function loadBuiltinDatasets() {
  try {
    builtinDatasets.value = await listDatasets();
    if (!selectedDatasetName.value && builtinDatasets.value.length > 0) {
      selectedDatasetName.value = builtinDatasets.value[0].name;
      datasetSampleSize.value = Math.min(5, builtinDatasets.value[0].count || 5);
    }
  } catch {
    // 后端不可用时不阻断页面
  }
}

loadBuiltinDatasets();

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
        keyword_rerank: keywordRerank.value
      }
    }, requestTimeoutMs.value);
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
        keyword_rerank: keywordRerank.value
      },
      cases: casesToRun
    }, requestTimeoutMs.value);

    progressText.value = `Batch Progress: 0/${casesToRun.length} (queued)`;

    while (true) {
      const status = await getAsyncRunStatus(startResp.run_id, requestTimeoutMs.value);
      progressText.value = `Batch Progress: ${status.completed}/${status.total} (${status.status})`;
      if (status.status === 'completed' && status.result) {
        batchResult.value = status.result;
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
        keyword_rerank: keywordRerank.value
      }
    }, requestTimeoutMs.value);

    progressText.value = `Dataset Progress: 0/${plannedTotal} (queued)`;

    while (true) {
      const status = await getAsyncRunStatus(startResp.run_id, requestTimeoutMs.value);
      progressText.value = `Dataset Progress: ${status.completed}/${status.total} (${status.status})`;
      if (status.status === 'completed' && status.result) {
        batchResult.value = status.result;
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

          <div v-if="config.engine === 'VectorEngine'" class="mt-2 rounded-lg border border-slate-600/60 bg-slate-900/40 p-3">
            <p class="mb-2 text-xs font-semibold text-arena-mint">Vector Retrieval Params (Chroma)</p>
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
          </div>
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
            <p class="mb-2 text-xs font-semibold text-arena-mint">内置数据集</p>
            <label class="field">
              <span>Dataset</span>
              <select v-model="selectedDatasetName" class="select">
                <option v-for="d in builtinDatasets" :key="d.name" :value="d.name">
                  {{ d.name }} ({{ d.count }})
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
              <span>每个测试独立会话（推荐）</span>
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
          title="LLM Judge: Precision/Faithfulness/InfoLoss。&#10;规则法: Recall@K、QA Accuracy/F1、Consistency、Rejection/Rejection@Unknown。"
        >
          评测口径：Precision/Faithfulness/InfoLoss 由 LLM Judge 给分；Recall@K、QA Accuracy/F1、Consistency、Rejection/Rejection@Unknown 为规则法自动评估。
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
