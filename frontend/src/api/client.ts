import axios from 'axios';
import type {
  BatchBenchmarkRunRequest,
  BatchBenchmarkRunResponse,
  BenchmarkRunRequest,
  BenchmarkRunResponse
} from '../types';

const client = axios.create({
  baseURL: import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000',
  timeout: 15000
});

export async function runBenchmark(payload: BenchmarkRunRequest): Promise<BenchmarkRunResponse> {
  const { data } = await client.post<BenchmarkRunResponse>('/api/benchmark/run', payload);
  return data;
}

export async function runBatchBenchmark(payload: BatchBenchmarkRunRequest): Promise<BatchBenchmarkRunResponse> {
  const { data } = await client.post<BatchBenchmarkRunResponse>('/api/benchmark/run-batch', payload);
  return data;
}
