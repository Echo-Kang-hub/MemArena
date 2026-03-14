import MarkdownIt from 'markdown-it';
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import MetricBars from './components/MetricBars.vue';
import { getAuditEventsByRun, getAsyncRunStatus, listDatasets, runBatchBenchmarkAsync, runBenchmarkWithTimeout, runDatasetBenchmarkAsync } from './api/client';
const config = ref({
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
const result = ref(null);
const batchResult = ref(null);
const retrievalTopK = ref(5);
const minRelevance = ref(0);
const collectionName = ref('memarena_memory');
const similarityStrategy = ref('inverse_distance');
const keywordRerank = ref(false);
const maxContextTokens = ref(1200);
const reasoningHops = ref(1);
const shortTermMode = ref('None');
const stmWindowTurns = ref(5);
const stmTokenBudget = ref(2000);
const stmSummaryKeepRecentTurns = ref(4);
const reflectorAutoWriteback = ref(false);
const reflectorWritebackMinConfidence = ref(0.75);
const datasetCases = ref([]);
const builtinDatasets = ref([]);
const selectedDatasetName = ref('');
const datasetSampleSize = ref(5);
const datasetStartIndex = ref(0);
const isolateSessions = ref(true);
const maxConcurrency = ref(3);
const batchCaseCount = ref(5);
const requestTimeoutMs = ref(120000);
const includeRawJudgeInMarkdown = ref(false);
const progressText = ref('');
const entityHealthState = ref(null);
const entityDiagnosticCopyFeedback = ref('');
const batchDiagnosticCopyFeedback = ref('');
const elapsedMs = ref(0);
const lastRunDurationMs = ref(null);
let timerHandle = null;
let runStartTs = 0;
const processors = ['RawLogger', 'Summarizer', 'EntityExtractor'];
const engines = ['VectorEngine', 'GraphEngine', 'RelationalEngine'];
const assemblers = [
    'SystemInjector',
    'XMLTagging',
    'TimelineRollover',
    'ReverseTimeline',
    'RankedPruning',
    'ReasoningChain'
];
const reflectors = [
    'None',
    'GenerativeReflection',
    'ConflictResolver',
    'Consolidator',
    'DecayFilter',
    'InsightLinker',
    'AbstractionReflector'
];
const providers = ['api', 'ollama', 'local'];
const summarizerMethods = ['llm', 'kmeans'];
const entityExtractorMethods = ['llm_triple', 'llm_attribute', 'spacy_llm_triple', 'spacy_llm_attribute'];
const computeDevices = ['cpu', 'cuda'];
const shortTermModes = [
    'None',
    'SlidingWindow',
    'TokenBuffer',
    'RollingSummary',
    'WorkingMemoryBlackboard'
];
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
const batchInputModeLabel = computed(() => usingUploadedBatchCases.value ? 'JSON 上传测试集模式' : '单条输入自动生成模式');
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
    if (!isEntityExtractorProcessor.value)
        return;
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
        lines.push(`${idx + 1}. ${evt.ts} | ${evt.purpose} | ${evt.provider}/${evt.model} | status=${evt.status}`, `   ${evt.error}`);
    });
    const text = lines.join('\n');
    try {
        await navigator.clipboard.writeText(text);
        entityDiagnosticCopyFeedback.value = '诊断信息已复制到剪贴板';
    }
    catch {
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
            lines.push(`${idx + 1}. ${evt.ts} | ${evt.purpose} | ${evt.provider}/${evt.model} | status=${evt.status}`, `   ${evt.error}`);
        });
    }
    try {
        await navigator.clipboard.writeText(lines.join('\n'));
        batchDiagnosticCopyFeedback.value = '批量诊断信息已复制到剪贴板';
    }
    catch {
        batchDiagnosticCopyFeedback.value = '复制失败，请手动复制当前批次摘要';
    }
}
async function refreshEntityHealthWarning(runId) {
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
        const latestFailed = failed[failed.length - 1];
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
    }
    catch {
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
function formatDuration(ms) {
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
        return [];
    }
    const firstHit = result.value.search_result.hits[0];
    const meta = firstHit?.metadata;
    const chains = meta && Array.isArray(meta.reasoning_chains)
        ? meta.reasoning_chains
        : [];
    return chains.map((c) => String(c)).filter(Boolean).slice(0, 20);
});
const singleReasoningSeeds = computed(() => {
    if (!result.value) {
        return [];
    }
    const firstHit = result.value.search_result.hits[0];
    const meta = firstHit?.metadata;
    const seeds = meta && Array.isArray(meta.reasoning_seed_entities)
        ? meta.reasoning_seed_entities
        : [];
    return seeds.map((s) => String(s)).filter(Boolean).slice(0, 20);
});
const singleReasoningChainDetails = computed(() => {
    if (!result.value) {
        return [];
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
        return [];
    }
    const precision = result.value.eval_result.metrics.precision;
    const coverage = result.value.eval_result.metrics.faithfulness;
    const infoLoss = result.value.eval_result.metrics.info_loss;
    const f1 = precision + coverage > 0 ? (2 * precision * coverage) / (precision + coverage) : 0;
    const retention = 1 - infoLoss;
    const hallucinationRisk = 1 - precision;
    const hitCount = result.value.search_result.hits.length;
    const avgRelevance = hitCount > 0
        ? result.value.search_result.hits.reduce((sum, hit) => sum + hit.relevance, 0) / hitCount
        : 0;
    const extraRows = [];
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
        return [];
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
    const std = (values) => {
        const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
        const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
        return Math.sqrt(variance);
    };
    const precisionStd = std(cases.map((r) => r.eval_result.metrics.precision));
    const faithfulnessStd = std(cases.map((r) => r.eval_result.metrics.faithfulness));
    const infoLossStd = std(cases.map((r) => r.eval_result.metrics.info_loss));
    const passCount = cases.filter((r) => r.eval_result.metrics.precision >= 0.8 &&
        r.eval_result.metrics.faithfulness >= 0.8 &&
        r.eval_result.metrics.info_loss <= 0.2).length;
    const passRate = passCount / cases.length;
    const worstF1 = sortedF1[0];
    const reasoningDetails = cases
        .map((r) => r.search_result.hits?.[0]?.metadata?.reasoning_chain_details)
        .filter((x) => Array.isArray(x));
    const reasoningChainCounts = reasoningDetails.map((arr) => arr.length).filter((v) => v > 0);
    const reasoningPriorities = reasoningDetails.flatMap((arr) => arr.map((d) => Number(d.priority)).filter((v) => Number.isFinite(v)));
    const reasoningHops = reasoningDetails.flatMap((arr) => arr.map((d) => Number(d.hop)).filter((v) => Number.isFinite(v)));
    const reasoningSeedTouches = reasoningDetails.flatMap((arr) => arr.map((d) => Number(d.seed_touch ? 1 : 0)));
    const extraRows = [];
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
    const unknownCount = batchResult.value.case_results.filter((r) => r.eval_result.metrics.rejection_correctness_unknown != null).length;
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
            `- convergence_speed: ${r.eval_result.metrics.convergence_speed ?? 'N/A'}`,
            `- context_distraction: ${r.eval_result.metrics.context_distraction ?? 'N/A'}`,
            ...(singleReasoningChainDetails.value.length > 0
                ? [
                    '',
                    '## Reasoning Quality',
                    `- seed_entities: ${singleReasoningSeeds.value.join(', ') || '(none)'}`,
                    `- chain_count: ${singleReasoningChainDetails.value.length}`,
                    `- avg_priority: ${(singleReasoningChainDetails.value.reduce((s, d) => s + Number(d.priority || 0), 0) /
                        Math.max(singleReasoningChainDetails.value.length, 1)).toFixed(4)}`,
                    `- avg_hop: ${(singleReasoningChainDetails.value.reduce((s, d) => s + Number(d.hop || 0), 0) /
                        Math.max(singleReasoningChainDetails.value.length, 1)).toFixed(4)}`
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
    if (!markdownReport.value)
        return;
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
    }
    catch {
        // 后端不可用时不阻断页面
    }
}
loadBuiltinDatasets();
function handleDatasetUpload(event) {
    const file = event.target.files?.[0];
    if (!file)
        return;
    const reader = new FileReader();
    reader.onload = () => {
        datasetJson.value = String(reader.result || '');
        try {
            const parsed = JSON.parse(datasetJson.value);
            if (Array.isArray(parsed) && parsed.length > 0) {
                datasetCases.value = parsed.map((item, idx) => ({
                    case_id: String(item.case_id || `case-${idx + 1}`),
                    input_text: String(item.input_text || ''),
                    expected_facts: Array.isArray(item.expected_facts) ? item.expected_facts.map(String) : [],
                    session_id: String(item.session_id || 'batch-session')
                }));
                if (datasetCases.value[0]?.input_text) {
                    inputText.value = datasetCases.value[0].input_text;
                }
            }
        }
        catch {
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
            .map((v) => v.trim())
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
                reflector_writeback_min_confidence: reflectorWritebackMinConfidence.value
            }
        }, requestTimeoutMs.value);
        if (result.value?.run_id) {
            await refreshEntityHealthWarning(result.value.run_id);
        }
    }
    catch (e) {
        error.value = e instanceof Error ? e.message : '运行失败，请检查后端服务。';
    }
    finally {
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
            .map((v) => v.trim())
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
                reflector_writeback_min_confidence: reflectorWritebackMinConfidence.value
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
    }
    catch (e) {
        error.value = e instanceof Error ? e.message : '批量运行失败，请检查后端服务。';
    }
    finally {
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
                reflector_writeback_min_confidence: reflectorWritebackMinConfidence.value
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
    }
    catch (e) {
        error.value = e instanceof Error ? e.message : '内置数据集运行失败。';
    }
    finally {
        stopRunTimer();
        lastRunDurationMs.value = elapsedMs.value;
        loading.value = false;
    }
}
function downloadCsvReport() {
    if (!batchResult.value?.csv_report)
        return;
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
debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
const __VLS_ctx = {};
let __VLS_components;
let __VLS_directives;
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "min-h-screen bg-aurora text-slate-100" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "mx-auto grid max-w-[1320px] grid-cols-1 gap-6 p-4 md:p-8 lg:grid-cols-3" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.section, __VLS_intrinsicElements.section)({
    ...{ class: "panel lg:col-span-1" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.h2, __VLS_intrinsicElements.h2)({
    ...{ class: "panel-title" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "grid gap-3" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.select, __VLS_intrinsicElements.select)({
    value: (__VLS_ctx.config.processor),
    ...{ class: "select" },
});
for (const [x] of __VLS_getVForSourceType((__VLS_ctx.processors))) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
        key: (x),
        value: (x),
    });
    (x);
}
if (__VLS_ctx.isSummarizerProcessor) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
        ...{ class: "field" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.select, __VLS_intrinsicElements.select)({
        value: (__VLS_ctx.config.summarizer_method),
        ...{ class: "select" },
    });
    for (const [x] of __VLS_getVForSourceType((__VLS_ctx.summarizerMethods))) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
            key: (x),
            value: (x),
        });
        (x);
    }
}
if (__VLS_ctx.isEntityExtractorProcessor) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
        ...{ class: "field" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.select, __VLS_intrinsicElements.select)({
        value: (__VLS_ctx.config.entity_extractor_method),
        ...{ class: "select" },
    });
    for (const [x] of __VLS_getVForSourceType((__VLS_ctx.entityExtractorMethods))) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
            key: (x),
            value: (x),
        });
        (x);
    }
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.select, __VLS_intrinsicElements.select)({
    value: (__VLS_ctx.config.engine),
    ...{ class: "select" },
});
for (const [x] of __VLS_getVForSourceType((__VLS_ctx.engines))) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
        key: (x),
        value: (x),
    });
    (x);
}
if (__VLS_ctx.isEntityExtractorProcessor) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
        ...{ class: "rounded-lg border border-slate-700/70 bg-slate-900/50 p-2 text-xs text-slate-300" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
        ...{ class: "font-semibold text-arena-mint" },
    });
    (__VLS_ctx.isEntityTripleMode ? 'Triple -> GraphEngine' : 'Attribute -> RelationalEngine');
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.select, __VLS_intrinsicElements.select)({
    value: (__VLS_ctx.config.assembler),
    ...{ class: "select" },
});
for (const [x] of __VLS_getVForSourceType((__VLS_ctx.assemblers))) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
        key: (x),
        value: (x),
    });
    (x);
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.select, __VLS_intrinsicElements.select)({
    value: (__VLS_ctx.config.reflector),
    ...{ class: "select" },
});
for (const [x] of __VLS_getVForSourceType((__VLS_ctx.reflectors))) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
        key: (x),
        value: (x),
    });
    (x);
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.select, __VLS_intrinsicElements.select)({
    value: (__VLS_ctx.config.chat_llm_provider),
    ...{ class: "select" },
});
for (const [x] of __VLS_getVForSourceType((__VLS_ctx.providers))) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
        key: (x),
        value: (x),
    });
    (x);
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.select, __VLS_intrinsicElements.select)({
    value: (__VLS_ctx.config.judge_llm_provider),
    ...{ class: "select" },
});
for (const [x] of __VLS_getVForSourceType((__VLS_ctx.providers))) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
        key: (x),
        value: (x),
    });
    (x);
}
if (__VLS_ctx.isSummarizerProcessor) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
        ...{ class: "field" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.select, __VLS_intrinsicElements.select)({
        value: (__VLS_ctx.config.summarizer_llm_provider),
        ...{ class: "select" },
    });
    for (const [x] of __VLS_getVForSourceType((__VLS_ctx.providers))) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
            key: (x),
            value: (x),
        });
        (x);
    }
}
if (__VLS_ctx.isEntityExtractorProcessor) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
        ...{ class: "field" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.select, __VLS_intrinsicElements.select)({
        value: (__VLS_ctx.config.entity_llm_provider),
        ...{ class: "select" },
    });
    for (const [x] of __VLS_getVForSourceType((__VLS_ctx.providers))) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
            key: (x),
            value: (x),
        });
        (x);
    }
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.select, __VLS_intrinsicElements.select)({
    value: (__VLS_ctx.config.embedding_provider),
    ...{ class: "select" },
});
for (const [x] of __VLS_getVForSourceType((__VLS_ctx.providers))) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
        key: (x),
        value: (x),
    });
    (x);
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.select, __VLS_intrinsicElements.select)({
    value: (__VLS_ctx.config.compute_device),
    ...{ class: "select" },
});
for (const [x] of __VLS_getVForSourceType((__VLS_ctx.computeDevices))) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
        key: (x),
        value: (x),
    });
    (x);
}
if (__VLS_ctx.config.engine === 'VectorEngine') {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "mt-2 rounded-lg border border-slate-600/60 bg-slate-900/40 p-3" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
        ...{ class: "mb-2 text-xs font-semibold text-arena-mint" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
        ...{ class: "field" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
        type: "number",
        min: "1",
        max: "50",
        ...{ class: "select" },
    });
    (__VLS_ctx.retrievalTopK);
    __VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
        ...{ class: "field mt-2" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
        type: "number",
        min: "0",
        max: "1",
        step: "0.01",
        ...{ class: "select" },
    });
    (__VLS_ctx.minRelevance);
    __VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
        ...{ class: "field mt-2" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
        value: (__VLS_ctx.collectionName),
        type: "text",
        ...{ class: "select" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
        ...{ class: "field mt-2" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.select, __VLS_intrinsicElements.select)({
        value: (__VLS_ctx.similarityStrategy),
        ...{ class: "select" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
        value: "inverse_distance",
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
        value: "exp_decay",
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
        value: "linear",
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
        ...{ class: "mt-2 flex items-center gap-2 text-sm text-slate-200" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
        type: "checkbox",
    });
    (__VLS_ctx.keywordRerank);
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "mt-2 rounded-lg border border-slate-600/60 bg-slate-900/40 p-3" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
    ...{ class: "mb-2 text-xs font-semibold text-arena-mint" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
    type: "number",
    min: "64",
    max: "8192",
    ...{ class: "select" },
});
(__VLS_ctx.maxContextTokens);
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field mt-2" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
    type: "number",
    min: "1",
    max: "3",
    ...{ class: "select" },
});
(__VLS_ctx.reasoningHops);
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "mt-2 rounded-lg border border-slate-600/60 bg-slate-900/40 p-3" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
    ...{ class: "mb-2 text-xs font-semibold text-arena-mint" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.select, __VLS_intrinsicElements.select)({
    value: (__VLS_ctx.shortTermMode),
    ...{ class: "select" },
});
for (const [x] of __VLS_getVForSourceType((__VLS_ctx.shortTermModes))) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
        key: (x),
        value: (x),
    });
    (x);
}
if (__VLS_ctx.showStmWindow) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
        ...{ class: "field mt-2" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
        type: "number",
        min: "1",
        max: "30",
        ...{ class: "select" },
    });
    (__VLS_ctx.stmWindowTurns);
}
if (__VLS_ctx.showStmTokenBudget) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
        ...{ class: "field mt-2" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
        type: "number",
        min: "128",
        max: "16000",
        ...{ class: "select" },
    });
    (__VLS_ctx.stmTokenBudget);
}
if (__VLS_ctx.showStmRolling) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
        ...{ class: "field mt-2" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
        type: "number",
        min: "1",
        max: "12",
        ...{ class: "select" },
    });
    (__VLS_ctx.stmSummaryKeepRecentTurns);
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
    ...{ class: "mt-2 text-xs text-slate-400" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "mt-2 rounded-lg border border-slate-600/60 bg-slate-900/40 p-3" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
    ...{ class: "mb-2 text-xs font-semibold text-arena-mint" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "flex items-center gap-2 text-sm text-slate-200" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
    type: "checkbox",
});
(__VLS_ctx.reflectorAutoWriteback);
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field mt-2" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
    type: "number",
    min: "0",
    max: "1",
    step: "0.05",
    ...{ class: "select" },
});
(__VLS_ctx.reflectorWritebackMinConfidence);
__VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
    ...{ class: "mt-2 text-xs text-slate-400" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.section, __VLS_intrinsicElements.section)({
    ...{ class: "panel lg:col-span-1" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.h2, __VLS_intrinsicElements.h2)({
    ...{ class: "panel-title" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "space-y-3" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.textarea)({
    value: (__VLS_ctx.inputText),
    rows: "4",
    ...{ class: "textarea" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.textarea)({
    value: (__VLS_ctx.expectedFactsRaw),
    rows: "4",
    ...{ class: "textarea" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
    ...{ onChange: (__VLS_ctx.handleDatasetUpload) },
    type: "file",
    accept: "application/json",
    ...{ class: "input-file" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
    type: "number",
    min: "1",
    max: "200",
    ...{ class: "select" },
});
(__VLS_ctx.batchCaseCount);
__VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
    ...{ class: "text-xs text-slate-400" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "rounded-lg border border-slate-600/60 bg-slate-900/40 p-3" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
    ...{ class: "mb-2 text-xs font-semibold text-arena-mint" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.select, __VLS_intrinsicElements.select)({
    value: (__VLS_ctx.selectedDatasetName),
    ...{ class: "select" },
});
for (const [d] of __VLS_getVForSourceType((__VLS_ctx.builtinDatasets))) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.option, __VLS_intrinsicElements.option)({
        key: (d.name),
        value: (d.name),
    });
    (d.name);
    (d.count);
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "mt-2 grid grid-cols-2 gap-2" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
    type: "number",
    min: "1",
    ...{ class: "select" },
});
(__VLS_ctx.datasetSampleSize);
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
    type: "number",
    min: "0",
    ...{ class: "select" },
});
(__VLS_ctx.datasetStartIndex);
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "mt-2 flex items-center gap-2 text-sm text-slate-200" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
    type: "checkbox",
});
(__VLS_ctx.isolateSessions);
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field mt-2" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
    type: "number",
    min: "1",
    max: "32",
    ...{ class: "select" },
});
(__VLS_ctx.maxConcurrency);
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "field mt-2" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
    type: "number",
    min: "5000",
    step: "1000",
    ...{ class: "select" },
});
(__VLS_ctx.requestTimeoutMs);
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "mt-5" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
    ...{ class: "mb-2 text-sm font-semibold text-slate-300" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
    ...{ class: "mb-2 rounded-lg border border-slate-600/60 bg-slate-900/50 p-2 text-xs text-slate-300" },
});
(__VLS_ctx.batchInputModeLabel);
(__VLS_ctx.plannedBatchCaseCount);
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "grid grid-cols-1 gap-2 sm:grid-cols-3" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (__VLS_ctx.onRunBenchmark) },
    ...{ class: "run-btn w-full" },
    disabled: (__VLS_ctx.loading),
});
(__VLS_ctx.loading ? 'Running...' : 'Run Benchmark');
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (__VLS_ctx.onRunBatchBenchmark) },
    ...{ class: "run-btn w-full" },
    disabled: (__VLS_ctx.loading),
});
(__VLS_ctx.loading ? 'Running...' : 'Run Batch');
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (__VLS_ctx.onRunBuiltinDataset) },
    ...{ class: "run-btn w-full" },
    disabled: (__VLS_ctx.loading),
});
(__VLS_ctx.loading ? 'Running...' : 'Run Built-in Dataset');
if (__VLS_ctx.error) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
        ...{ class: "mt-3 rounded-lg border border-red-500/40 bg-red-500/10 p-2 text-sm text-red-200" },
    });
    (__VLS_ctx.error);
}
if (__VLS_ctx.entityHealthState) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "mt-3 rounded-lg border p-2" },
        ...{ class: (__VLS_ctx.entityHealthState.level === 'warning'
                ? 'border-arena-amber/50 bg-arena-amber/10 text-arena-amber'
                : 'border-arena-cyan/50 bg-arena-cyan/10 text-arena-mint') },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
        ...{ class: "text-sm" },
    });
    (__VLS_ctx.entityHealthState.message);
    __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
        ...{ class: "mt-1 text-xs opacity-90" },
    });
    (__VLS_ctx.entityHealthState.details);
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "mt-2 flex items-center gap-2" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
        ...{ onClick: (__VLS_ctx.copyEntityDiagnostic) },
        type: "button",
        ...{ class: "rounded border border-slate-500/60 bg-slate-900/60 px-2 py-1 text-xs text-slate-100 transition hover:border-slate-300" },
    });
    if (__VLS_ctx.entityDiagnosticCopyFeedback) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
            ...{ class: "text-xs text-slate-200/90" },
        });
        (__VLS_ctx.entityDiagnosticCopyFeedback);
    }
    __VLS_asFunctionalElement(__VLS_intrinsicElements.details, __VLS_intrinsicElements.details)({
        ...{ class: "mt-2 rounded border border-slate-600/50 bg-slate-950/40 p-2 text-xs text-slate-100" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.summary, __VLS_intrinsicElements.summary)({
        ...{ class: "cursor-pointer select-none text-slate-200" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.ul, __VLS_intrinsicElements.ul)({
        ...{ class: "mt-2 list-disc space-y-1 pl-4" },
    });
    for (const [evt, idx] of __VLS_getVForSourceType((__VLS_ctx.entityHealthState.recentFailures))) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.li, __VLS_intrinsicElements.li)({
            key: (`${evt.ts}-${idx}`),
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
            ...{ class: "text-slate-300" },
        });
        (evt.ts);
        __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
        (evt.purpose);
        (evt.provider);
        (evt.model);
        (evt.status);
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "text-slate-400" },
        });
        (evt.error);
    }
}
if (__VLS_ctx.loading && __VLS_ctx.progressText) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
        ...{ class: "mt-3 rounded-lg border border-arena-cyan/40 bg-arena-cyan/10 p-2 text-sm text-arena-mint" },
    });
    (__VLS_ctx.progressText);
}
if (__VLS_ctx.loading) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
        ...{ class: "mt-2 rounded-lg border border-arena-amber/40 bg-arena-amber/10 p-2 text-sm text-arena-amber" },
    });
    (__VLS_ctx.runningDurationLabel);
}
else if (__VLS_ctx.lastRunDurationMs !== null) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
        ...{ class: "mt-2 rounded-lg border border-slate-500/40 bg-slate-800/60 p-2 text-sm text-slate-200" },
    });
    (__VLS_ctx.finishedDurationLabel);
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.section, __VLS_intrinsicElements.section)({
    ...{ class: "panel lg:col-span-1" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.h2, __VLS_intrinsicElements.h2)({
    ...{ class: "panel-title" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
    ...{ class: "mb-3 rounded-lg border border-slate-600/60 bg-slate-900/50 p-2 text-xs text-slate-300" },
    title: "LLM Judge: Precision/Faithfulness/InfoLoss。&#10;规则法: Recall@K、QA Accuracy/F1、Consistency、Rejection/Rejection@Unknown、Convergence Speed、Context Distraction。",
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.label, __VLS_intrinsicElements.label)({
    ...{ class: "mb-3 flex items-center gap-2 rounded-lg border border-slate-600/60 bg-slate-900/40 p-2 text-xs text-slate-200" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
    type: "checkbox",
});
(__VLS_ctx.includeRawJudgeInMarkdown);
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
if (__VLS_ctx.result) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "space-y-4" },
    });
    /** @type {[typeof MetricBars, ]} */ ;
    // @ts-ignore
    const __VLS_0 = __VLS_asFunctionalComponent(MetricBars, new MetricBars({
        metrics: (__VLS_ctx.result.eval_result.metrics),
    }));
    const __VLS_1 = __VLS_0({
        metrics: (__VLS_ctx.result.eval_result.metrics),
    }, ...__VLS_functionalComponentArgsRest(__VLS_0));
    if (__VLS_ctx.singleSafetySignals) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "rounded-xl border border-slate-600/60 bg-slate-900/60 p-3" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.h3, __VLS_intrinsicElements.h3)({
            ...{ class: "mb-2 text-sm font-semibold text-arena-mint" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "grid grid-cols-1 gap-2 sm:grid-cols-2" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "rounded-lg border border-slate-700/60 bg-slate-900/40 p-2" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
            ...{ class: "text-xs font-semibold text-slate-200" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
            ...{ class: "mt-1 text-xs text-arena-mint" },
        });
        (__VLS_ctx.singleSafetySignals.rejectionRate == null ? 'N/A' : `${(__VLS_ctx.singleSafetySignals.rejectionRate * 100).toFixed(1)}%`);
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "rounded-lg border border-slate-700/60 bg-slate-900/40 p-2" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
            ...{ class: "text-xs font-semibold text-slate-200" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
            ...{ class: "mt-1 text-xs text-arena-mint" },
        });
        (__VLS_ctx.singleSafetySignals.rejectionUnknownCorrectness == null ? 'N/A（仅未知样本）' : `${(__VLS_ctx.singleSafetySignals.rejectionUnknownCorrectness * 100).toFixed(1)}%`);
    }
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "rounded-xl border border-slate-600/60 bg-slate-900/60 p-3" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.h3, __VLS_intrinsicElements.h3)({
        ...{ class: "mb-2 text-sm font-semibold text-arena-mint" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "space-y-2" },
    });
    for (const [item] of __VLS_getVForSourceType((__VLS_ctx.singleDerivedRows))) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            key: (item.label),
            ...{ class: "rounded-lg border border-slate-700/60 bg-slate-900/40 p-2" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "flex items-center justify-between" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
            ...{ class: "text-xs font-semibold text-slate-200" },
        });
        (item.label);
        __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
            ...{ class: "text-xs text-arena-mint" },
        });
        (item.value);
        __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
            ...{ class: "mt-1 text-[11px] text-slate-400" },
        });
        (item.hint);
    }
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "rounded-xl border border-slate-600/60 bg-slate-900/60 p-3" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.h3, __VLS_intrinsicElements.h3)({
        ...{ class: "mb-2 text-sm font-semibold text-arena-mint" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.pre, __VLS_intrinsicElements.pre)({
        ...{ class: "max-h-60 overflow-auto text-xs text-slate-200" },
    });
    (__VLS_ctx.result.assemble_result.prompt);
    if (__VLS_ctx.singleReasoningChains.length > 0) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "rounded-xl border border-slate-600/60 bg-slate-900/60 p-3" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.h3, __VLS_intrinsicElements.h3)({
            ...{ class: "mb-2 text-sm font-semibold text-arena-mint" },
        });
        if (__VLS_ctx.singleReasoningSeeds.length > 0) {
            __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
                ...{ class: "mb-2 text-[11px] text-slate-400" },
            });
            (__VLS_ctx.singleReasoningSeeds.join(', '));
        }
        if (__VLS_ctx.singleReasoningChainDetails.length > 0) {
            __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
                ...{ class: "mb-2 overflow-auto rounded-md border border-slate-700/60 bg-slate-900/40" },
            });
            __VLS_asFunctionalElement(__VLS_intrinsicElements.table, __VLS_intrinsicElements.table)({
                ...{ class: "min-w-full text-[11px] text-slate-200" },
            });
            __VLS_asFunctionalElement(__VLS_intrinsicElements.thead, __VLS_intrinsicElements.thead)({
                ...{ class: "bg-slate-800/70 text-slate-300" },
            });
            __VLS_asFunctionalElement(__VLS_intrinsicElements.tr, __VLS_intrinsicElements.tr)({});
            __VLS_asFunctionalElement(__VLS_intrinsicElements.th, __VLS_intrinsicElements.th)({
                ...{ class: "px-2 py-1 text-left" },
            });
            __VLS_asFunctionalElement(__VLS_intrinsicElements.th, __VLS_intrinsicElements.th)({
                ...{ class: "px-2 py-1 text-left" },
            });
            __VLS_asFunctionalElement(__VLS_intrinsicElements.th, __VLS_intrinsicElements.th)({
                ...{ class: "px-2 py-1 text-left" },
            });
            __VLS_asFunctionalElement(__VLS_intrinsicElements.th, __VLS_intrinsicElements.th)({
                ...{ class: "px-2 py-1 text-left" },
            });
            __VLS_asFunctionalElement(__VLS_intrinsicElements.tbody, __VLS_intrinsicElements.tbody)({});
            for (const [d, idx] of __VLS_getVForSourceType((__VLS_ctx.singleReasoningChainDetails))) {
                __VLS_asFunctionalElement(__VLS_intrinsicElements.tr, __VLS_intrinsicElements.tr)({
                    key: (`detail-${idx}-${d.chain}`),
                    ...{ class: "border-t border-slate-800/60" },
                });
                __VLS_asFunctionalElement(__VLS_intrinsicElements.td, __VLS_intrinsicElements.td)({
                    ...{ class: "px-2 py-1" },
                });
                (d.priority.toFixed(3));
                __VLS_asFunctionalElement(__VLS_intrinsicElements.td, __VLS_intrinsicElements.td)({
                    ...{ class: "px-2 py-1" },
                });
                (d.hop);
                __VLS_asFunctionalElement(__VLS_intrinsicElements.td, __VLS_intrinsicElements.td)({
                    ...{ class: "px-2 py-1" },
                });
                (d.seed_touch ? 'Y' : 'N');
                __VLS_asFunctionalElement(__VLS_intrinsicElements.td, __VLS_intrinsicElements.td)({
                    ...{ class: "px-2 py-1" },
                });
                ((d.lexical_overlap * 100).toFixed(1));
            }
        }
        __VLS_asFunctionalElement(__VLS_intrinsicElements.ul, __VLS_intrinsicElements.ul)({
            ...{ class: "max-h-40 list-disc space-y-1 overflow-auto pl-5 text-xs text-slate-200" },
        });
        for (const [chain, idx] of __VLS_getVForSourceType((__VLS_ctx.singleReasoningChains))) {
            __VLS_asFunctionalElement(__VLS_intrinsicElements.li, __VLS_intrinsicElements.li)({
                key: (`${idx}-${chain}`),
            });
            (chain);
        }
    }
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "rounded-xl border border-slate-600/60 bg-slate-900/60 p-3" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.h3, __VLS_intrinsicElements.h3)({
        ...{ class: "mb-2 text-sm font-semibold text-arena-mint" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
        ...{ class: "text-xs text-slate-200" },
    });
    (__VLS_ctx.result.eval_result.judge_rationale);
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "rounded-xl border border-slate-600/60 bg-slate-900/60 p-3" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.h3, __VLS_intrinsicElements.h3)({
        ...{ class: "mb-2 text-sm font-semibold text-arena-mint" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.pre, __VLS_intrinsicElements.pre)({
        ...{ class: "max-h-48 overflow-auto text-xs text-slate-200" },
    });
    (__VLS_ctx.result.eval_result.raw_judge_output || 'N/A');
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "rounded-xl border border-slate-600/60 bg-slate-900/60 p-3" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.h3, __VLS_intrinsicElements.h3)({
        ...{ class: "mb-2 text-sm font-semibold text-arena-mint" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "markdown-preview max-h-64 overflow-auto" },
    });
    __VLS_asFunctionalDirective(__VLS_directives.vHtml)(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.renderedMarkdownReport) }, null, null);
    __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
        ...{ onClick: (__VLS_ctx.downloadMarkdownReport) },
        ...{ class: "mt-2 rounded-lg bg-arena-amber px-3 py-1 text-xs font-semibold text-slate-900" },
    });
}
else if (__VLS_ctx.batchResult) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "space-y-4" },
    });
    /** @type {[typeof MetricBars, ]} */ ;
    // @ts-ignore
    const __VLS_3 = __VLS_asFunctionalComponent(MetricBars, new MetricBars({
        metrics: (__VLS_ctx.batchResult.avg_metrics),
    }));
    const __VLS_4 = __VLS_3({
        metrics: (__VLS_ctx.batchResult.avg_metrics),
    }, ...__VLS_functionalComponentArgsRest(__VLS_3));
    if (__VLS_ctx.batchSafetySignals) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "rounded-xl border border-slate-600/60 bg-slate-900/60 p-3" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.h3, __VLS_intrinsicElements.h3)({
            ...{ class: "mb-2 text-sm font-semibold text-arena-mint" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "grid grid-cols-1 gap-2 sm:grid-cols-2" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "rounded-lg border border-slate-700/60 bg-slate-900/40 p-2" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
            ...{ class: "text-xs font-semibold text-slate-200" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
            ...{ class: "mt-1 text-xs text-arena-mint" },
        });
        (__VLS_ctx.batchSafetySignals.rejectionRate == null ? 'N/A' : `${(__VLS_ctx.batchSafetySignals.rejectionRate * 100).toFixed(1)}%`);
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "rounded-lg border border-slate-700/60 bg-slate-900/40 p-2" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
            ...{ class: "text-xs font-semibold text-slate-200" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
            ...{ class: "mt-1 text-xs text-arena-mint" },
        });
        (__VLS_ctx.batchSafetySignals.rejectionUnknownCorrectness == null ? 'N/A（仅未知样本）' : `${(__VLS_ctx.batchSafetySignals.rejectionUnknownCorrectness * 100).toFixed(1)}%`);
        __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
            ...{ class: "mt-2 text-[11px] text-slate-400" },
        });
        (__VLS_ctx.batchSafetySignals.unknownCount);
        (__VLS_ctx.batchSafetySignals.knownCount);
        __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
            ...{ class: "mt-1 text-[11px] text-slate-400" },
        });
        ((__VLS_ctx.batchSafetySignals.unknownRatio * 100).toFixed(1));
        __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
            ...{ class: "mt-2 rounded-md border border-slate-700/60 bg-slate-900/40 p-2 text-[11px] text-slate-300" },
        });
        (__VLS_ctx.batchSafetyInterpretation);
    }
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "rounded-xl border border-slate-600/60 bg-slate-900/60 p-3" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.h3, __VLS_intrinsicElements.h3)({
        ...{ class: "mb-2 text-sm font-semibold text-arena-mint" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "space-y-2" },
    });
    for (const [item] of __VLS_getVForSourceType((__VLS_ctx.batchDerivedRows))) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            key: (item.label),
            ...{ class: "rounded-lg border border-slate-700/60 bg-slate-900/40 p-2" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "flex items-center justify-between" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
            ...{ class: "text-xs font-semibold text-slate-200" },
        });
        (item.label);
        __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
            ...{ class: "text-xs text-arena-mint" },
        });
        (item.value);
        __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
            ...{ class: "mt-1 text-[11px] text-slate-400" },
        });
        (item.hint);
    }
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "rounded-xl border border-slate-600/60 bg-slate-900/60 p-3" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.h3, __VLS_intrinsicElements.h3)({
        ...{ class: "mb-2 text-sm font-semibold text-arena-mint" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
        ...{ class: "text-xs text-slate-200" },
    });
    (__VLS_ctx.batchResult.run_id);
    __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
        ...{ class: "text-xs text-slate-200" },
    });
    (__VLS_ctx.batchResult.case_results.length);
    __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
        ...{ onClick: (__VLS_ctx.downloadCsvReport) },
        ...{ class: "mt-2 rounded-lg bg-arena-amber px-3 py-1 text-xs font-semibold text-slate-900" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
        ...{ onClick: (__VLS_ctx.downloadMarkdownReport) },
        ...{ class: "ml-2 mt-2 rounded-lg bg-arena-amber px-3 py-1 text-xs font-semibold text-slate-900" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
        ...{ onClick: (__VLS_ctx.copyBatchDiagnostic) },
        ...{ class: "ml-2 mt-2 rounded-lg border border-slate-500/60 bg-slate-900/60 px-3 py-1 text-xs font-semibold text-slate-100" },
    });
    if (__VLS_ctx.batchDiagnosticCopyFeedback) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
            ...{ class: "mt-2 text-xs text-slate-300" },
        });
        (__VLS_ctx.batchDiagnosticCopyFeedback);
    }
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "rounded-xl border border-slate-600/60 bg-slate-900/60 p-3" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.h3, __VLS_intrinsicElements.h3)({
        ...{ class: "mb-2 text-sm font-semibold text-arena-mint" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "markdown-preview max-h-64 overflow-auto" },
    });
    __VLS_asFunctionalDirective(__VLS_directives.vHtml)(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.renderedMarkdownReport) }, null, null);
}
else {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "rounded-xl border border-dashed border-slate-500 p-4 text-sm text-slate-300" },
    });
}
/** @type {__VLS_StyleScopedClasses['min-h-screen']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-aurora']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-100']} */ ;
/** @type {__VLS_StyleScopedClasses['mx-auto']} */ ;
/** @type {__VLS_StyleScopedClasses['grid']} */ ;
/** @type {__VLS_StyleScopedClasses['max-w-[1320px]']} */ ;
/** @type {__VLS_StyleScopedClasses['grid-cols-1']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-6']} */ ;
/** @type {__VLS_StyleScopedClasses['p-4']} */ ;
/** @type {__VLS_StyleScopedClasses['md:p-8']} */ ;
/** @type {__VLS_StyleScopedClasses['lg:grid-cols-3']} */ ;
/** @type {__VLS_StyleScopedClasses['panel']} */ ;
/** @type {__VLS_StyleScopedClasses['lg:col-span-1']} */ ;
/** @type {__VLS_StyleScopedClasses['panel-title']} */ ;
/** @type {__VLS_StyleScopedClasses['grid']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-3']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-700/70']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/50']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-300']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/40']} */ ;
/** @type {__VLS_StyleScopedClasses['p-3']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['flex']} */ ;
/** @type {__VLS_StyleScopedClasses['items-center']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/40']} */ ;
/** @type {__VLS_StyleScopedClasses['p-3']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/40']} */ ;
/** @type {__VLS_StyleScopedClasses['p-3']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-400']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/40']} */ ;
/** @type {__VLS_StyleScopedClasses['p-3']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['flex']} */ ;
/** @type {__VLS_StyleScopedClasses['items-center']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-400']} */ ;
/** @type {__VLS_StyleScopedClasses['panel']} */ ;
/** @type {__VLS_StyleScopedClasses['lg:col-span-1']} */ ;
/** @type {__VLS_StyleScopedClasses['panel-title']} */ ;
/** @type {__VLS_StyleScopedClasses['space-y-3']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['textarea']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['textarea']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['input-file']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-400']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/40']} */ ;
/** @type {__VLS_StyleScopedClasses['p-3']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['grid']} */ ;
/** @type {__VLS_StyleScopedClasses['grid-cols-2']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-2']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['flex']} */ ;
/** @type {__VLS_StyleScopedClasses['items-center']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['field']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['select']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-5']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-300']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/50']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-300']} */ ;
/** @type {__VLS_StyleScopedClasses['grid']} */ ;
/** @type {__VLS_StyleScopedClasses['grid-cols-1']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-2']} */ ;
/** @type {__VLS_StyleScopedClasses['sm:grid-cols-3']} */ ;
/** @type {__VLS_StyleScopedClasses['run-btn']} */ ;
/** @type {__VLS_StyleScopedClasses['w-full']} */ ;
/** @type {__VLS_StyleScopedClasses['run-btn']} */ ;
/** @type {__VLS_StyleScopedClasses['w-full']} */ ;
/** @type {__VLS_StyleScopedClasses['run-btn']} */ ;
/** @type {__VLS_StyleScopedClasses['w-full']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-3']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-red-500/40']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-red-500/10']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['text-red-200']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-3']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['opacity-90']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['flex']} */ ;
/** @type {__VLS_StyleScopedClasses['items-center']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-500/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/60']} */ ;
/** @type {__VLS_StyleScopedClasses['px-2']} */ ;
/** @type {__VLS_StyleScopedClasses['py-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-100']} */ ;
/** @type {__VLS_StyleScopedClasses['transition']} */ ;
/** @type {__VLS_StyleScopedClasses['hover:border-slate-300']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200/90']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/50']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-950/40']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-100']} */ ;
/** @type {__VLS_StyleScopedClasses['cursor-pointer']} */ ;
/** @type {__VLS_StyleScopedClasses['select-none']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['list-disc']} */ ;
/** @type {__VLS_StyleScopedClasses['space-y-1']} */ ;
/** @type {__VLS_StyleScopedClasses['pl-4']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-300']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-400']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-3']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-arena-cyan/40']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-arena-cyan/10']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-arena-amber/40']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-arena-amber/10']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-amber']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-500/40']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-800/60']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['panel']} */ ;
/** @type {__VLS_StyleScopedClasses['lg:col-span-1']} */ ;
/** @type {__VLS_StyleScopedClasses['panel-title']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-3']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/50']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-300']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-3']} */ ;
/** @type {__VLS_StyleScopedClasses['flex']} */ ;
/** @type {__VLS_StyleScopedClasses['items-center']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/40']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['space-y-4']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-xl']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/60']} */ ;
/** @type {__VLS_StyleScopedClasses['p-3']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['grid']} */ ;
/** @type {__VLS_StyleScopedClasses['grid-cols-1']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-2']} */ ;
/** @type {__VLS_StyleScopedClasses['sm:grid-cols-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-700/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/40']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-700/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/40']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-xl']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/60']} */ ;
/** @type {__VLS_StyleScopedClasses['p-3']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['space-y-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-700/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/40']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['flex']} */ ;
/** @type {__VLS_StyleScopedClasses['items-center']} */ ;
/** @type {__VLS_StyleScopedClasses['justify-between']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-[11px]']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-400']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-xl']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/60']} */ ;
/** @type {__VLS_StyleScopedClasses['p-3']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['max-h-60']} */ ;
/** @type {__VLS_StyleScopedClasses['overflow-auto']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-xl']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/60']} */ ;
/** @type {__VLS_StyleScopedClasses['p-3']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-[11px]']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-400']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['overflow-auto']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-md']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-700/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/40']} */ ;
/** @type {__VLS_StyleScopedClasses['min-w-full']} */ ;
/** @type {__VLS_StyleScopedClasses['text-[11px]']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-800/70']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-300']} */ ;
/** @type {__VLS_StyleScopedClasses['px-2']} */ ;
/** @type {__VLS_StyleScopedClasses['py-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-left']} */ ;
/** @type {__VLS_StyleScopedClasses['px-2']} */ ;
/** @type {__VLS_StyleScopedClasses['py-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-left']} */ ;
/** @type {__VLS_StyleScopedClasses['px-2']} */ ;
/** @type {__VLS_StyleScopedClasses['py-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-left']} */ ;
/** @type {__VLS_StyleScopedClasses['px-2']} */ ;
/** @type {__VLS_StyleScopedClasses['py-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-left']} */ ;
/** @type {__VLS_StyleScopedClasses['border-t']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-800/60']} */ ;
/** @type {__VLS_StyleScopedClasses['px-2']} */ ;
/** @type {__VLS_StyleScopedClasses['py-1']} */ ;
/** @type {__VLS_StyleScopedClasses['px-2']} */ ;
/** @type {__VLS_StyleScopedClasses['py-1']} */ ;
/** @type {__VLS_StyleScopedClasses['px-2']} */ ;
/** @type {__VLS_StyleScopedClasses['py-1']} */ ;
/** @type {__VLS_StyleScopedClasses['px-2']} */ ;
/** @type {__VLS_StyleScopedClasses['py-1']} */ ;
/** @type {__VLS_StyleScopedClasses['max-h-40']} */ ;
/** @type {__VLS_StyleScopedClasses['list-disc']} */ ;
/** @type {__VLS_StyleScopedClasses['space-y-1']} */ ;
/** @type {__VLS_StyleScopedClasses['overflow-auto']} */ ;
/** @type {__VLS_StyleScopedClasses['pl-5']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-xl']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/60']} */ ;
/** @type {__VLS_StyleScopedClasses['p-3']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-xl']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/60']} */ ;
/** @type {__VLS_StyleScopedClasses['p-3']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['max-h-48']} */ ;
/** @type {__VLS_StyleScopedClasses['overflow-auto']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-xl']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/60']} */ ;
/** @type {__VLS_StyleScopedClasses['p-3']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['markdown-preview']} */ ;
/** @type {__VLS_StyleScopedClasses['max-h-64']} */ ;
/** @type {__VLS_StyleScopedClasses['overflow-auto']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-arena-amber']} */ ;
/** @type {__VLS_StyleScopedClasses['px-3']} */ ;
/** @type {__VLS_StyleScopedClasses['py-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-900']} */ ;
/** @type {__VLS_StyleScopedClasses['space-y-4']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-xl']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/60']} */ ;
/** @type {__VLS_StyleScopedClasses['p-3']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['grid']} */ ;
/** @type {__VLS_StyleScopedClasses['grid-cols-1']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-2']} */ ;
/** @type {__VLS_StyleScopedClasses['sm:grid-cols-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-700/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/40']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-700/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/40']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-[11px]']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-400']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-[11px]']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-400']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-md']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-700/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/40']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-[11px]']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-300']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-xl']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/60']} */ ;
/** @type {__VLS_StyleScopedClasses['p-3']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['space-y-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-700/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/40']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['flex']} */ ;
/** @type {__VLS_StyleScopedClasses['items-center']} */ ;
/** @type {__VLS_StyleScopedClasses['justify-between']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-[11px]']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-400']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-xl']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/60']} */ ;
/** @type {__VLS_StyleScopedClasses['p-3']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-arena-amber']} */ ;
/** @type {__VLS_StyleScopedClasses['px-3']} */ ;
/** @type {__VLS_StyleScopedClasses['py-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-900']} */ ;
/** @type {__VLS_StyleScopedClasses['ml-2']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-arena-amber']} */ ;
/** @type {__VLS_StyleScopedClasses['px-3']} */ ;
/** @type {__VLS_StyleScopedClasses['py-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-900']} */ ;
/** @type {__VLS_StyleScopedClasses['ml-2']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-500/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/60']} */ ;
/** @type {__VLS_StyleScopedClasses['px-3']} */ ;
/** @type {__VLS_StyleScopedClasses['py-1']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-100']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-300']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-xl']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-600/60']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-900/60']} */ ;
/** @type {__VLS_StyleScopedClasses['p-3']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-arena-mint']} */ ;
/** @type {__VLS_StyleScopedClasses['markdown-preview']} */ ;
/** @type {__VLS_StyleScopedClasses['max-h-64']} */ ;
/** @type {__VLS_StyleScopedClasses['overflow-auto']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-xl']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-dashed']} */ ;
/** @type {__VLS_StyleScopedClasses['border-slate-500']} */ ;
/** @type {__VLS_StyleScopedClasses['p-4']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-300']} */ ;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            MetricBars: MetricBars,
            config: config,
            inputText: inputText,
            expectedFactsRaw: expectedFactsRaw,
            loading: loading,
            error: error,
            result: result,
            batchResult: batchResult,
            retrievalTopK: retrievalTopK,
            minRelevance: minRelevance,
            collectionName: collectionName,
            similarityStrategy: similarityStrategy,
            keywordRerank: keywordRerank,
            maxContextTokens: maxContextTokens,
            reasoningHops: reasoningHops,
            shortTermMode: shortTermMode,
            stmWindowTurns: stmWindowTurns,
            stmTokenBudget: stmTokenBudget,
            stmSummaryKeepRecentTurns: stmSummaryKeepRecentTurns,
            reflectorAutoWriteback: reflectorAutoWriteback,
            reflectorWritebackMinConfidence: reflectorWritebackMinConfidence,
            builtinDatasets: builtinDatasets,
            selectedDatasetName: selectedDatasetName,
            datasetSampleSize: datasetSampleSize,
            datasetStartIndex: datasetStartIndex,
            isolateSessions: isolateSessions,
            maxConcurrency: maxConcurrency,
            batchCaseCount: batchCaseCount,
            requestTimeoutMs: requestTimeoutMs,
            includeRawJudgeInMarkdown: includeRawJudgeInMarkdown,
            progressText: progressText,
            entityHealthState: entityHealthState,
            entityDiagnosticCopyFeedback: entityDiagnosticCopyFeedback,
            batchDiagnosticCopyFeedback: batchDiagnosticCopyFeedback,
            lastRunDurationMs: lastRunDurationMs,
            processors: processors,
            engines: engines,
            assemblers: assemblers,
            reflectors: reflectors,
            providers: providers,
            summarizerMethods: summarizerMethods,
            entityExtractorMethods: entityExtractorMethods,
            computeDevices: computeDevices,
            shortTermModes: shortTermModes,
            plannedBatchCaseCount: plannedBatchCaseCount,
            batchInputModeLabel: batchInputModeLabel,
            isSummarizerProcessor: isSummarizerProcessor,
            isEntityExtractorProcessor: isEntityExtractorProcessor,
            showStmWindow: showStmWindow,
            showStmTokenBudget: showStmTokenBudget,
            showStmRolling: showStmRolling,
            isEntityTripleMode: isEntityTripleMode,
            copyEntityDiagnostic: copyEntityDiagnostic,
            copyBatchDiagnostic: copyBatchDiagnostic,
            runningDurationLabel: runningDurationLabel,
            finishedDurationLabel: finishedDurationLabel,
            singleReasoningChains: singleReasoningChains,
            singleReasoningSeeds: singleReasoningSeeds,
            singleReasoningChainDetails: singleReasoningChainDetails,
            singleDerivedRows: singleDerivedRows,
            batchDerivedRows: batchDerivedRows,
            singleSafetySignals: singleSafetySignals,
            batchSafetySignals: batchSafetySignals,
            batchSafetyInterpretation: batchSafetyInterpretation,
            renderedMarkdownReport: renderedMarkdownReport,
            downloadMarkdownReport: downloadMarkdownReport,
            handleDatasetUpload: handleDatasetUpload,
            onRunBenchmark: onRunBenchmark,
            onRunBatchBenchmark: onRunBatchBenchmark,
            onRunBuiltinDataset: onRunBuiltinDataset,
            downloadCsvReport: downloadCsvReport,
        };
    },
});
export default (await import('vue')).defineComponent({
    setup() {
        return {};
    },
});
; /* PartiallyEnd: #4569/main.vue */
//# sourceMappingURL=App.vue.js.map