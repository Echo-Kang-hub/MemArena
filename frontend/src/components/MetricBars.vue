<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{
  metrics: {
    precision: number;
    faithfulness: number;
    info_loss: number;
  };
}>();

// 将指标映射为可渲染的数组，便于模板中做统一循环渲染
const rows = computed(() => [
  { label: 'Precision', value: props.metrics.precision, color: 'bg-arena-cyan' },
  { label: 'Faithfulness', value: props.metrics.faithfulness, color: 'bg-arena-amber' },
  { label: 'InfoLoss', value: props.metrics.info_loss, color: 'bg-arena-coral' }
]);
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
