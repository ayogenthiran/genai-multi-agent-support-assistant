"""Application configuration loaded from environment variables.

Uses pydantic-settings to validate paths, model names, and RAG parameters.
Centralizes all `.env` values so agents, tools, and the UI share one source
of truth.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """Validated application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="OPENAI_EMBEDDING_MODEL",
    )
    sqlite_db_path: Path = Field(
        default=Path("data/customers.db"),
        validation_alias="SQLITE_DB_PATH",
    )
    chroma_persist_dir: Path = Field(
        default=Path("data/chroma"),
        validation_alias="CHROMA_PERSIST_DIR",
    )
    policies_dir: Path = Field(
        default=Path("data/policies"),
        validation_alias="POLICIES_DIR",
    )
    rag_chunk_size: int = Field(default=800, validation_alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=120, validation_alias="RAG_CHUNK_OVERLAP")
    rag_top_k: int = Field(default=4, validation_alias="RAG_TOP_K")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    @field_validator("sqlite_db_path", "chroma_persist_dir", "policies_dir", mode="after")
    @classmethod
    def resolve_relative_paths(cls, value: Path) -> Path:
        if value.is_absolute():
            return value
        return PROJECT_ROOT / value


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
