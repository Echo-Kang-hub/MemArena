import { computed } from 'vue';
const props = defineProps();
// 将指标映射为可渲染的数组，便于模板中做统一循环渲染
const rows = computed(() => {
    const base = [
        { label: 'Precision', value: props.metrics.precision, color: 'bg-arena-cyan' },
        { label: 'Faithfulness', value: props.metrics.faithfulness, color: 'bg-arena-amber' },
        { label: 'InfoLoss', value: props.metrics.info_loss, color: 'bg-arena-coral' }
    ];
    if (props.metrics.recall_at_k != null) {
        base.push({ label: 'Recall@K', value: props.metrics.recall_at_k, color: 'bg-emerald-400' });
    }
    if (props.metrics.qa_accuracy != null) {
        base.push({ label: 'QA Accuracy', value: props.metrics.qa_accuracy, color: 'bg-blue-400' });
    }
    if (props.metrics.qa_f1 != null) {
        base.push({ label: 'QA F1', value: props.metrics.qa_f1, color: 'bg-violet-400' });
    }
    if (props.metrics.consistency_score != null) {
        base.push({ label: 'Consistency', value: props.metrics.consistency_score, color: 'bg-teal-400' });
    }
    if (props.metrics.rejection_rate != null) {
        base.push({ label: 'Rejection', value: props.metrics.rejection_rate, color: 'bg-rose-400' });
    }
    if (props.metrics.rejection_correctness_unknown != null) {
        base.push({ label: 'Rejection@Unknown', value: props.metrics.rejection_correctness_unknown, color: 'bg-lime-400' });
    }
    if (props.metrics.context_distraction != null) {
        base.push({ label: 'Context Distraction', value: props.metrics.context_distraction, color: 'bg-orange-400' });
    }
    return base;
});
debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
const __VLS_ctx = {};
let __VLS_components;
let __VLS_directives;
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "space-y-3" },
});
for (const [row] of __VLS_getVForSourceType((__VLS_ctx.rows))) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        key: (row.label),
        ...{ class: "space-y-1" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "flex justify-between text-sm text-slate-200" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
    (row.label);
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
    ((row.value * 100).toFixed(1));
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "h-2 rounded-full bg-slate-700" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div)({
        ...{ class: "h-2 rounded-full transition-all duration-500" },
        ...{ class: (row.color) },
        ...{ style: ({ width: `${Math.max(0, Math.min(100, row.value * 100))}%` }) },
    });
}
/** @type {__VLS_StyleScopedClasses['space-y-3']} */ ;
/** @type {__VLS_StyleScopedClasses['space-y-1']} */ ;
/** @type {__VLS_StyleScopedClasses['flex']} */ ;
/** @type {__VLS_StyleScopedClasses['justify-between']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['text-slate-200']} */ ;
/** @type {__VLS_StyleScopedClasses['h-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-full']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-slate-700']} */ ;
/** @type {__VLS_StyleScopedClasses['h-2']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-full']} */ ;
/** @type {__VLS_StyleScopedClasses['transition-all']} */ ;
/** @type {__VLS_StyleScopedClasses['duration-500']} */ ;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            rows: rows,
        };
    },
    __typeProps: {},
});
export default (await import('vue')).defineComponent({
    setup() {
        return {};
    },
    __typeProps: {},
});
; /* PartiallyEnd: #4569/main.vue */
//# sourceMappingURL=MetricBars.vue.js.map