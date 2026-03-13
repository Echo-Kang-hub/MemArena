from pydantic_settings import BaseSettings, SettingsConfigDict


# 统一读取 .env 配置，并给出合理默认值，便于快速本地启动
class Settings(BaseSettings):
    app_env: str = "dev"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_origin: str = "http://localhost:5173"

    default_llm_provider: str = "api"
    default_embedding_provider: str = "ollama"

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

    sqlite_path: str = "./backend/data/memarena.db"
    chroma_persist_dir: str = "./backend/data/chroma"
    chroma_collection_name: str = "memarena_memory"
    milvus_uri: str = "http://localhost:19530"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "memarena"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
