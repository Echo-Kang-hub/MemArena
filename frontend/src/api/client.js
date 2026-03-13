import axios from 'axios';
const client = axios.create({
    baseURL: import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000',
    timeout: 15000
});
export async function runBenchmark(payload) {
    const { data } = await client.post('/api/benchmark/run', payload);
    return data;
}
export async function runBatchBenchmark(payload) {
    const { data } = await client.post('/api/benchmark/run-batch', payload);
    return data;
}
export async function listDatasets() {
    const { data } = await client.get('/api/datasets');
    return data.datasets;
}
export async function runDatasetBenchmark(payload) {
    const { data } = await client.post('/api/benchmark/run-dataset', payload);
    return data;
}
//# sourceMappingURL=client.js.map