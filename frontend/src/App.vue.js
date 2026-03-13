import { ref } from 'vue';
import MetricBars from './components/MetricBars.vue';
import { runBatchBenchmark, runBenchmark } from './api/client';
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
const processors = ['RawLogger', 'Summarizer', 'EntityExtractor'];
const engines = ['VectorEngine', 'GraphEngine', 'RelationalEngine'];
const assemblers = ['SystemInjector', 'XMLTagging', 'TimelineRollover'];
const reflectors = ['None', 'GenerativeReflection', 'ConflictResolver'];
const providers = ['api', 'ollama', 'local'];
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
    try {
        const expectedFacts = expectedFactsRaw.value
            .split('\n')
            .map((v) => v.trim())
            .filter(Boolean);
        result.value = await runBenchmark({
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
        });
    }
    catch (e) {
        error.value = e instanceof Error ? e.message : '运行失败，请检查后端服务。';
    }
    finally {
        loading.value = false;
    }
}
async function onRunBatchBenchmark() {
    loading.value = true;
    error.value = '';
    result.value = null;
    try {
        if (datasetCases.value.length === 0) {
            throw new Error('请先上传 JSON 数组测试集。');
        }
        batchResult.value = await runBatchBenchmark({
            config: config.value,
            user_id: 'ui-batch-user',
            retrieval: {
                top_k: retrievalTopK.value,
                min_relevance: minRelevance.value,
                collection_name: collectionName.value,
                similarity_strategy: similarityStrategy.value,
                keyword_rerank: keywordRerank.value
            },
            cases: datasetCases.value
        });
    }
    catch (e) {
        error.value = e instanceof Error ? e.message : '批量运行失败，请检查后端服务。';
    }
    finally {
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
    ...{ class: "mt-5 flex items-center gap-3" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (__VLS_ctx.onRunBenchmark) },
    ...{ class: "run-btn" },
    disabled: (__VLS_ctx.loading),
});
(__VLS_ctx.loading ? 'Running...' : 'Run Benchmark');
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (__VLS_ctx.onRunBatchBenchmark) },
    ...{ class: "run-btn" },
    disabled: (__VLS_ctx.loading),
});
(__VLS_ctx.loading ? 'Running...' : 'Run Batch');
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
    ...{ class: "text-sm text-slate-300" },
});
if (__VLS_ctx.error) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
        ...{ class: "mt-3 rounded-lg border border-red-500/40 bg-red-500/10 p-2 text-sm text-red-200" },
    });
    (__VLS_ctx.error);
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
/** @type {__VLS_StyleScopedClasses['mt-5']} */ ;
/** @type {__VLS_StyleScopedClasses['flex']} */ ;
/** @type {__VLS_StyleScopedClasses['items-center']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-3']} */ ;
/** @type {__VLS_StyleScopedClasses['run-btn']} */ ;
/** @type {__VLS_StyleScopedClasses['run-btn']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-300']} */ ;
/** @type {__VLS_StyleScopedClasses['mt-3']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-red-500/40']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-red-500/10']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['text-red-200']} */ ;
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
            processors: processors,
            engines: engines,
            assemblers: assemblers,
            reflectors: reflectors,
            providers: providers,
            handleDatasetUpload: handleDatasetUpload,
            onRunBenchmark: onRunBenchmark,
            onRunBatchBenchmark: onRunBatchBenchmark,
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