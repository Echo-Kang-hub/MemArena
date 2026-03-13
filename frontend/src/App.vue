<script setup lang="ts">
import { ref } from 'vue';
import MetricBars from './components/MetricBars.vue';
import { runBatchBenchmark, runBenchmark } from './api/client';
import type { BatchBenchmarkRunResponse, BenchmarkConfig, BenchmarkRunResponse } from './types';

const config = ref<BenchmarkConfig>({
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
const result = ref<BenchmarkRunResponse | null>(null);
const batchResult = ref<BatchBenchmarkRunResponse | null>(null);
const retrievalTopK = ref(5);
const minRelevance = ref(0);
const collectionName = ref('memarena_memory');
const similarityStrategy = ref<'inverse_distance' | 'exp_decay' | 'linear'>('inverse_distance');
const keywordRerank = ref(false);
const datasetCases = ref<Array<{ case_id: string; input_text: string; expected_facts: string[]; session_id: string }>>([]);

const processors = ['RawLogger', 'Summarizer', 'EntityExtractor'] as const;
const engines = ['VectorEngine', 'GraphEngine', 'RelationalEngine'] as const;
const assemblers = ['SystemInjector', 'XMLTagging', 'TimelineRollover'] as const;
const reflectors = ['None', 'GenerativeReflection', 'ConflictResolver'] as const;
const providers = ['api', 'ollama', 'local'] as const;

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

  try {
    const expectedFacts = expectedFactsRaw.value
      .split('\n')
      .map((v: string) => v.trim())
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
  } catch (e) {
    error.value = e instanceof Error ? e.message : '运行失败，请检查后端服务。';
  } finally {
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
  } catch (e) {
    error.value = e instanceof Error ? e.message : '批量运行失败，请检查后端服务。';
  } finally {
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
          <label class="field">
            <span>Engine</span>
            <select v-model="config.engine" class="select">
              <option v-for="x in engines" :key="x" :value="x">{{ x }}</option>
            </select>
          </label>
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
            <span>LLM Provider</span>
            <select v-model="config.llm_provider" class="select">
              <option v-for="x in providers" :key="x" :value="x">{{ x }}</option>
            </select>
          </label>
          <label class="field">
            <span>Embedding Provider</span>
            <select v-model="config.embedding_provider" class="select">
              <option v-for="x in providers" :key="x" :value="x">{{ x }}</option>
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
        </div>

        <div class="mt-5 flex items-center gap-3">
          <button class="run-btn" :disabled="loading" @click="onRunBenchmark">
            {{ loading ? 'Running...' : 'Run Benchmark' }}
          </button>
          <button class="run-btn" :disabled="loading" @click="onRunBatchBenchmark">
            {{ loading ? 'Running...' : 'Run Batch' }}
          </button>
          <span class="text-sm text-slate-300">Execution</span>
        </div>

        <p v-if="error" class="mt-3 rounded-lg border border-red-500/40 bg-red-500/10 p-2 text-sm text-red-200">
          {{ error }}
        </p>
      </section>

      <section class="panel lg:col-span-1">
        <h2 class="panel-title">Results Dashboard</h2>

        <div v-if="result" class="space-y-4">
          <MetricBars :metrics="result.eval_result.metrics" />

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
        </div>

        <div v-else-if="batchResult" class="space-y-4">
          <MetricBars :metrics="batchResult.avg_metrics" />
          <div class="rounded-xl border border-slate-600/60 bg-slate-900/60 p-3">
            <h3 class="mb-2 text-sm font-semibold text-arena-mint">Batch Summary</h3>
            <p class="text-xs text-slate-200">Run ID: {{ batchResult.run_id }}</p>
            <p class="text-xs text-slate-200">Cases: {{ batchResult.case_results.length }}</p>
            <button class="mt-2 rounded-lg bg-arena-amber px-3 py-1 text-xs font-semibold text-slate-900" @click="downloadCsvReport">
              Download CSV Report
            </button>
          </div>
        </div>

        <div v-else class="rounded-xl border border-dashed border-slate-500 p-4 text-sm text-slate-300">
          运行后将在此展示指标柱状图与 Prompt 拼装预览。
        </div>
      </section>
    </div>
  </div>
</template>
