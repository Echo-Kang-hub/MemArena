from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def _resolve_from_project_root(path_value: str) -> str:
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


# 统一读取 .env 配置，并给出合理默认值，便于快速本地启动
class Settings(BaseSettings):
    app_env: str = "dev"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_origin: str = "http://localhost:5173"

    default_llm_provider: str = "api"
    default_embedding_provider: str = "ollama"

    chat_llm_provider: str = ""
    chat_api_base_url: str = ""
    chat_api_key: str = ""
    chat_api_model: str = ""
    chat_ollama_base_url: str = ""
    chat_ollama_model: str = ""
    chat_local_model_path: str = ""

    judge_llm_provider: str = ""
    judge_api_base_url: str = ""
    judge_api_key: str = ""
    judge_api_model: str = ""
    judge_ollama_base_url: str = ""
    judge_ollama_model: str = ""
    judge_local_model_path: str = ""

    summarizer_llm_provider: str = ""
    summarizer_api_base_url: str = ""
    summarizer_api_key: str = ""
    summarizer_api_model: str = ""
    summarizer_ollama_base_url: str = ""
    summarizer_ollama_model: str = ""
    summarizer_local_model_path: str = ""

    entity_llm_provider: str = ""
    entity_api_base_url: str = ""
    entity_api_key: str = ""
    entity_api_model: str = ""
    entity_ollama_base_url: str = ""
    entity_ollama_model: str = ""
    entity_local_model_path: str = ""

    embedding_provider: str = ""
    embedding_api_base_url: str = ""
    embedding_api_key: str = ""
    embedding_api_model: str = ""
    embedding_ollama_base_url: str = ""
    embedding_ollama_model: str = ""
    embedding_local_model_path: str = ""
    local_infer_device: str = "cpu"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_chat_model: str = "claude-3-5-sonnet-latest"

    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "Qwen/Qwen3-Embedding-0.6B"

    local_llm_model_path: str = ""
    local_embed_model_path: str = ""

    graph_relevance_lexical_weight: float = 0.45
    graph_relevance_entity_weight: float = 0.35
    graph_relevance_completeness_weight: float = 0.20
    graph_relevance_hint_weight: float = 0.10
    graph_relevance_fallback_lexical_weight: float = 0.90
    graph_relevance_fallback_hint_weight: float = 0.10

    relational_relevance_lexical_weight: float = 0.45
    relational_relevance_entity_weight: float = 0.35
    relational_relevance_completeness_weight: float = 0.20
    relational_relevance_hint_weight: float = 0.10
    relational_relevance_fallback_lexical_weight: float = 0.90
    relational_relevance_fallback_hint_weight: float = 0.10

    sqlite_path: str = str(DEFAULT_DATA_DIR / "memarena.db")
    chroma_persist_dir: str = str(DEFAULT_DATA_DIR / "chroma")
    chroma_collection_name: str = "memarena_memory"
    request_log_path: str = str(DEFAULT_DATA_DIR / "logs" / "request_audit.jsonl")
    milvus_uri: str = "http://localhost:19530"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "memarena"

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_post_init(self, __context: object) -> None:
        self.sqlite_path = _resolve_from_project_root(self.sqlite_path)
        self.chroma_persist_dir = _resolve_from_project_root(self.chroma_persist_dir)
        self.request_log_path = _resolve_from_project_root(self.request_log_path)


settings = Settings()
