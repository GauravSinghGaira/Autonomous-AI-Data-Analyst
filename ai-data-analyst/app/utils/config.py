"""
Central application configuration.

All environment-driven settings live here so the rest of the codebase
never reads os.environ directly. Uses pydantic-settings so values are
validated and typed at startup.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider ---
    llm_provider: str = Field(default="ollama", description="'ollama', 'openai', or 'groq'")
    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")
    groq_api_key: str | None = Field(default=None)
    groq_model: str = Field(default="openai/gpt-oss-120b")
    ollama_model: str = Field(default="llama3.2:3b")
    ollama_base_url: str = Field(default="http://localhost:11434")
    llm_temperature: float = Field(default=0.1)

    # --- Embeddings / RAG ---
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    chroma_persist_dir: str = Field(default="./storage/chroma_db")
    rag_top_k: int = Field(default=4)
    docs_dir: str = Field(default="./sample_data/docs")

    # --- Data / uploads ---
    upload_dir: str = Field(default="./storage/uploads")
    reports_dir: str = Field(default="./storage/reports")
    max_upload_mb: int = Field(default=50)

    # --- ML ---
    anomaly_contamination: float = Field(default=0.05)

    # --- API ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # --- Logging ---
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="./storage/logs/app.log")


settings = Settings()
