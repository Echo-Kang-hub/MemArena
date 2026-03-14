import axios from 'axios';
import type {
  AsyncRunStartResponse,
  AsyncRunStatusResponse,
  BatchBenchmarkRunRequest,
  BatchBenchmarkRunResponse,
  DatasetRunRequest,
  DatasetSummary,
  BenchmarkRunRequest,
  BenchmarkRunResponse
} from '../types';

const client = axios.create({
  baseURL: import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000',
  timeout: 30000
});

export async function runBenchmark(payload: BenchmarkRunRequest): Promise<BenchmarkRunResponse> {
  const { data } = await client.post<BenchmarkRunResponse>('/api/benchmark/run', payload);
  return data;
}

export async function runBenchmarkWithTimeout(
  payload: BenchmarkRunRequest,
  timeoutMs: number
): Promise<BenchmarkRunResponse> {
  const { data } = await client.post<BenchmarkRunResponse>('/api/benchmark/run', payload, {
    timeout: timeoutMs
  });
  return data;
}

export async function runBatchBenchmark(payload: BatchBenchmarkRunRequest): Promise<BatchBenchmarkRunResponse> {
  const { data } = await client.post<BatchBenchmarkRunResponse>('/api/benchmark/run-batch', payload, {
    timeout: 180000
  });
  return data;
}

export async function runBatchBenchmarkAsync(
  payload: BatchBenchmarkRunRequest,
  timeoutMs: number
): Promise<AsyncRunStartResponse> {
  const { data } = await client.post<AsyncRunStartResponse>('/api/benchmark/run-batch-async', payload, {
    timeout: timeoutMs
  });
  return data;
}

export async function listDatasets(): Promise<DatasetSummary[]> {
  const { data } = await client.get<{ datasets: DatasetSummary[] }>('/api/datasets');
  return data.datasets;
}

export async function runDatasetBenchmark(payload: DatasetRunRequest): Promise<BatchBenchmarkRunResponse> {
  const { data } = await client.post<BatchBenchmarkRunResponse>('/api/benchmark/run-dataset', payload, {
    timeout: 180000
  });
  return data;
}

export async function runDatasetBenchmarkAsync(
  payload: DatasetRunRequest,
  timeoutMs: number
): Promise<AsyncRunStartResponse> {
  const { data } = await client.post<AsyncRunStartResponse>('/api/benchmark/run-dataset-async', payload, {
    timeout: timeoutMs
  });
  return data;
}

export async function getAsyncRunStatus(runId: string, timeoutMs?: number): Promise<AsyncRunStatusResponse> {
  const { data } = await client.get<AsyncRunStatusResponse>(`/api/benchmark/runs/${runId}`, {
    timeout: timeoutMs ?? 30000
  });
  return data;
}

export async function getAuditEventsByRun(
  runId: string,
  limit = 300,
  timeoutMs?: number
): Promise<{ run_id: string; count: number; events: Array<Record<string, unknown>> }> {
  const { data } = await client.get<{ run_id: string; count: number; events: Array<Record<string, unknown>> }>(
    `/api/audit/runs/${runId}`,
    {
      params: { limit },
      timeout: timeoutMs ?? 30000
    }
  );
  return data;
}
