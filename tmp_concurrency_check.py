import json
import time
import urllib.request

BASE = {
    "dataset_name": "builtin_memory_smoke",
    "config": {
        "processor": "RawLogger",
        "engine": "VectorEngine",
        "assembler": "SystemInjector",
        "reflector": "None",
        "llm_provider": "api",
        "chat_llm_provider": "api",
        "judge_llm_provider": "api",
        "embedding_provider": "ollama",
        "compute_device": "cpu",
    },
    "retrieval": {
        "top_k": 5,
        "min_relevance": 0.0,
        "collection_name": "memarena_memory",
        "similarity_strategy": "inverse_distance",
        "keyword_rerank": False,
    },
    "user_id": "perf-user",
    "sample_size": 5,
    "start_index": 0,
    "isolate_sessions": True,
}


def run(concurrency: int):
    payload = dict(BASE)
    payload["max_concurrency"] = concurrency
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/benchmark/run-dataset",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    duration = time.perf_counter() - start
    print(f"c={concurrency} seconds={duration:.2f} run_id={body.get('run_id')}")


if __name__ == "__main__":
    for c in (1, 3, 5):
        run(c)
