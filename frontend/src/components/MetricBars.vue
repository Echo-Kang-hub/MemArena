<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{
  metrics: {
    precision: number;
    faithfulness: number;
    info_loss: number;
    recall_at_k?: number | null;
    qa_accuracy?: number | null;
    qa_f1?: number | null;
    consistency_score?: number | null;
    rejection_rate?: number | null;
    rejection_correctness_unknown?: number | null;
    context_distraction?: number | null;
  };
}>();

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
</script>

<template>
  <div class="space-y-3">
    <div v-for="row in rows" :key="row.label" class="space-y-1">
      <div class="flex justify-between text-sm text-slate-200">
        <span>{{ row.label }}</span>
        <span>{{ (row.value * 100).toFixed(1) }}%</span>
      </div>
      <div class="h-2 rounded-full bg-slate-700">
        <div
          class="h-2 rounded-full transition-all duration-500"
          :class="row.color"
          :style="{ width: `${Math.max(0, Math.min(100, row.value * 100))}%` }"
        />
      </div>
    </div>
  </div>
</template>
