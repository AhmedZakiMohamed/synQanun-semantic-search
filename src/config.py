
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "


class Settings(BaseSettings):
    """Typed, validated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #  Required by the specification 
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-base"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "legal_documents"
    RAW_DATA_DIR: str = "./data"
    TOP_K: int = Field(default=5, ge=1)
    MAX_CHUNK_CHARS: int = Field(default=1200, ge=100)
    CHUNK_OVERLAP_CHARS: int = Field(default=150, ge=0)

    #  Operational tuning  
    EMBEDDING_DEVICE: str = "auto"  
    UPSERT_BATCH_SIZE: int = Field(default=64, ge=1)
    MAX_TOP_K: int = Field(default=50, ge=1)
    CANDIDATE_MULTIPLIER: int = Field(default=3, ge=1)
    MAX_QUERY_CHARS: int = Field(default=1000, ge=1)

    @model_validator(mode="after")
    def _validate_consistency(self) -> "Settings":
        if self.CHUNK_OVERLAP_CHARS >= self.MAX_CHUNK_CHARS:
            raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than MAX_CHUNK_CHARS.")
        if self.TOP_K > self.MAX_TOP_K:
            raise ValueError("TOP_K must not exceed MAX_TOP_K.")
        return self

    @property
    def is_e5_model(self) -> bool:
        """Whether the configured model expects E5-style ``query:``/``passage:`` prefixes."""
        return "e5" in self.EMBEDDING_MODEL_NAME.lower()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance."""
    return Settings()
