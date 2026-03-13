import axios from 'axios';
const client = axios.create({
    baseURL: import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000',
    timeout: 30000
});
export async function runBenchmark(payload) {
    const { data } = await client.post('/api/benchmark/run', payload);
    return data;
}
export async function runBenchmarkWithTimeout(payload, timeoutMs) {
    const { data } = await client.post('/api/benchmark/run', payload, {
        timeout: timeoutMs
    });
    return data;
}
export async function runBatchBenchmark(payload) {
    const { data } = await client.post('/api/benchmark/run-batch', payload, {
        timeout: 180000
    });
    return data;
}
export async function runBatchBenchmarkAsync(payload, timeoutMs) {
    const { data } = await client.post('/api/benchmark/run-batch-async', payload, {
        timeout: timeoutMs
    });
    return data;
}
export async function listDatasets() {
    const { data } = await client.get('/api/datasets');
    return data.datasets;
}
export async function runDatasetBenchmark(payload) {
    const { data } = await client.post('/api/benchmark/run-dataset', payload, {
        timeout: 180000
    });
    return data;
}
export async function runDatasetBenchmarkAsync(payload, timeoutMs) {
    const { data } = await client.post('/api/benchmark/run-dataset-async', payload, {
        timeout: timeoutMs
    });
    return data;
}
export async function getAsyncRunStatus(runId) {
    const { data } = await client.get(`/api/benchmark/runs/${runId}`);
    return data;
}
//# sourceMappingURL=client.js.map