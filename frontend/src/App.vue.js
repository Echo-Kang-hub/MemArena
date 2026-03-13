import { computed, onBeforeUnmount, ref } from 'vue';
import MetricBars from './components/MetricBars.vue';
import { getAsyncRunStatus, listDatasets, runBatchBenchmarkAsync, runBenchmarkWithTimeout, runDatasetBenchmarkAsync } from './api/client';
const config = ref({
    processor: 'RawLogger',
    engine: 'VectorEngine',
    assembler: 'SystemInjector',
    reflector: 'None',
    llm_provider: 'api',
    embedding_provider: 'ollama'
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
const datasetCases = ref([]);
const builtinDatasets = ref([]);
const selectedDatasetName = ref('');
const datasetSampleSize = ref(5);
const datasetStartIndex = ref(0);
const isolateSessions = ref(true);
const requestTimeoutMs = ref(120000);
const progressText = ref('');
const elapsedMs = ref(0);
const lastRunDurationMs = ref(null);
let timerHandle = null;
let runStartTs = 0;
const processors = ['RawLogger', 'Summarizer', 'EntityExtractor'];
const engines = ['VectorEngine', 'GraphEngine', 'RelationalEngine'];
const assemblers = ['SystemInjector', 'XMLTagging', 'TimelineRollover'];
const reflectors = ['None', 'GenerativeReflection', 'ConflictResolver'];
const providers = ['api', 'ollama', 'local'];
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
    return [
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
    return [
        { label: 'Pass Rate (P>=0.8/F>=0.8/L<=0.2)', value: `${(passRate * 100).toFixed(1)}%`, hint: '达标样本比例，避免只看均值。' },
        { label: 'Mean F1', value: `${(meanF1 * 100).toFixed(1)}%`, hint: '批量整体平衡表现。' },
        { label: 'Median F1', value: `${(medianF1 * 100).toFixed(1)}%`, hint: '中位水平，降低极端值影响。' },
        { label: 'Worst-case F1', value: `${(worstF1 * 100).toFixed(1)}%`, hint: '最差样本性能，反映可靠性下限。' },
        { label: 'Std(P/F/L)', value: `${precisionStd.toFixed(3)} / ${faithfulnessStd.toFixed(3)} / ${infoLossStd.toFixed(3)}`, hint: '波动越小说明系统越稳定。' }
    ];
});
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
    progressText.value = '';
    lastRunDurationMs.value = null;
    startRunTimer();
    try {
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
                keyword_rerank: keywordRerank.value
            }
        }, requestTimeoutMs.value);
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
    progressText.value = '';
    lastRunDurationMs.value = null;
    startRunTimer();
    try {
        if (datasetCases.value.length === 0) {
            throw new Error('请先上传 JSON 数组测试集。');
        }
        progressText.value = `Batch Progress: 0/${datasetCases.value.length} (submitting)`;
        const startResp = await runBatchBenchmarkAsync({
            config: config.value,
            user_id: 'ui-batch-user',
            isolate_sessions: isolateSessions.value,
            retrieval: {
                top_k: retrievalTopK.value,
                min_relevance: minRelevance.value,
                collection_name: collectionName.value,
                similarity_strategy: similarityStrategy.value,
                keyword_rerank: keywordRerank.value
            },
            cases: datasetCases.value
        }, requestTimeoutMs.value);
        progressText.value = `Batch Progress: 0/${datasetCases.value.length} (queued)`;
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
    progressText.value = '';
    lastRunDurationMs.value = null;
    startRunTimer();
    try {
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
    value: (__VLS_ctx.config.llm_provider),
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
/** @type {__VLS_StyleScopedClasses['mt-5']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
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
            builtinDatasets: builtinDatasets,
            selectedDatasetName: selectedDatasetName,
            datasetSampleSize: datasetSampleSize,
            datasetStartIndex: datasetStartIndex,
            isolateSessions: isolateSessions,
            requestTimeoutMs: requestTimeoutMs,
            progressText: progressText,
            lastRunDurationMs: lastRunDurationMs,
            processors: processors,
            engines: engines,
            assemblers: assemblers,
            reflectors: reflectors,
            providers: providers,
            runningDurationLabel: runningDurationLabel,
            finishedDurationLabel: finishedDurationLabel,
            singleDerivedRows: singleDerivedRows,
            batchDerivedRows: batchDerivedRows,
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