from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    ollama_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "qwen2.5:3b"
    max_upload_mb: int = Field(default=10, ge=1, le=100)
    max_document_chars: int = Field(default=500_000, ge=1000)
    chunk_size: int = Field(default=1200, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    top_k: int = Field(default=5, ge=1, le=10)
    min_similarity: float = Field(default=0.45, ge=-1, le=1)
    ollama_timeout: float = Field(default=120, gt=0)
