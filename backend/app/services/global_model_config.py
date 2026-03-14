from __future__ import annotations

import re
from pathlib import Path

from app.config import PROJECT_ROOT, settings
from app.models.contracts import GlobalModelConfig


ENV_KEYS: dict[str, str] = {
    "default_llm_provider": "DEFAULT_LLM_PROVIDER",
    "default_embedding_provider": "DEFAULT_EMBEDDING_PROVIDER",
    "chat_llm_provider": "CHAT_LLM_PROVIDER",
    "chat_api_base_url": "CHAT_API_BASE_URL",
    "chat_api_key": "CHAT_API_KEY",
    "chat_api_model": "CHAT_API_MODEL",
    "chat_ollama_base_url": "CHAT_OLLAMA_BASE_URL",
    "chat_ollama_model": "CHAT_OLLAMA_MODEL",
    "chat_local_model_path": "CHAT_LOCAL_MODEL_PATH",
    "judge_llm_provider": "JUDGE_LLM_PROVIDER",
    "judge_api_base_url": "JUDGE_API_BASE_URL",
    "judge_api_key": "JUDGE_API_KEY",
    "judge_api_model": "JUDGE_API_MODEL",
    "judge_ollama_base_url": "JUDGE_OLLAMA_BASE_URL",
    "judge_ollama_model": "JUDGE_OLLAMA_MODEL",
    "judge_local_model_path": "JUDGE_LOCAL_MODEL_PATH",
    "summarizer_llm_provider": "SUMMARIZER_LLM_PROVIDER",
    "summarizer_api_base_url": "SUMMARIZER_API_BASE_URL",
    "summarizer_api_key": "SUMMARIZER_API_KEY",
    "summarizer_api_model": "SUMMARIZER_API_MODEL",
    "summarizer_ollama_base_url": "SUMMARIZER_OLLAMA_BASE_URL",
    "summarizer_ollama_model": "SUMMARIZER_OLLAMA_MODEL",
    "summarizer_local_model_path": "SUMMARIZER_LOCAL_MODEL_PATH",
    "entity_llm_provider": "ENTITY_LLM_PROVIDER",
    "entity_api_base_url": "ENTITY_API_BASE_URL",
    "entity_api_key": "ENTITY_API_KEY",
    "entity_api_model": "ENTITY_API_MODEL",
    "entity_ollama_base_url": "ENTITY_OLLAMA_BASE_URL",
    "entity_ollama_model": "ENTITY_OLLAMA_MODEL",
    "entity_local_model_path": "ENTITY_LOCAL_MODEL_PATH",
    "reflector_llm_provider": "REFLECTOR_LLM_PROVIDER",
    "reflector_api_base_url": "REFLECTOR_API_BASE_URL",
    "reflector_api_key": "REFLECTOR_API_KEY",
    "reflector_api_model": "REFLECTOR_API_MODEL",
    "reflector_ollama_base_url": "REFLECTOR_OLLAMA_BASE_URL",
    "reflector_ollama_model": "REFLECTOR_OLLAMA_MODEL",
    "reflector_local_model_path": "REFLECTOR_LOCAL_MODEL_PATH",
    "embedding_provider": "EMBEDDING_PROVIDER",
    "embedding_api_base_url": "EMBEDDING_API_BASE_URL",
    "embedding_api_key": "EMBEDDING_API_KEY",
    "embedding_api_model": "EMBEDDING_API_MODEL",
    "embedding_ollama_base_url": "EMBEDDING_OLLAMA_BASE_URL",
    "embedding_ollama_model": "EMBEDDING_OLLAMA_MODEL",
    "embedding_local_model_path": "EMBEDDING_LOCAL_MODEL_PATH",
    "local_infer_device": "LOCAL_INFER_DEVICE",
}


def env_file_path() -> Path:
    return (PROJECT_ROOT / ".env").resolve()


def _read_env_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def read_global_model_config() -> GlobalModelConfig:
    env_map = _read_env_map(env_file_path())
    payload: dict[str, str] = {}
    for field_name, env_key in ENV_KEYS.items():
        payload[field_name] = str(env_map.get(env_key, getattr(settings, field_name, "") or ""))
    return GlobalModelConfig(**payload)


def _normalize_value(value: str) -> str:
    text = str(value or "")
    text = text.replace("\r", "").replace("\n", " ")
    return text


def save_global_model_config(config: GlobalModelConfig) -> None:
    path = env_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    updates = {ENV_KEYS[k]: _normalize_value(str(v)) for k, v in config.model_dump().items() if k in ENV_KEYS}
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    key_pattern = re.compile(r"^\s*([A-Z0-9_]+)\s*=.*$")
    seen: set[str] = set()
    output_lines: list[str] = []

    for line in existing_lines:
        m = key_pattern.match(line)
        if not m:
            output_lines.append(line)
            continue
        key = m.group(1)
        if key in updates:
            output_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output_lines.append(line)

    missing = [k for k in updates.keys() if k not in seen]
    if missing and output_lines and output_lines[-1].strip() != "":
        output_lines.append("")
    for key in missing:
        output_lines.append(f"{key}={updates[key]}")

    path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    # 运行时热更新，下一次请求立即生效。
    for field_name, value in config.model_dump().items():
        if hasattr(settings, field_name):
            setattr(settings, field_name, str(value or ""))
