"""Configuration management for the study assistant."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class Config(BaseModel):
    """Application configuration."""

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    vector_store_path: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "faiss_index"
    )

    # LLM Configuration
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7"))
    )

    # RAG Configuration
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))
    top_k_retrieval: int = Field(
        default_factory=lambda: int(os.getenv("TOP_K_RETRIEVAL", "5"))
    )

    # MongoDB Configuration
    mongodb_url: str = Field(
        default_factory=lambda: os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    )
    mongodb_db_name: str = Field(
        default_factory=lambda: os.getenv("MONGODB_DB_NAME", "study_assist")
    )

    # JWT Configuration
    jwt_secret_key: str = Field(
        default_factory=lambda: os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    )
    jwt_algorithm: str = Field(
        default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256")
    )
    jwt_access_token_expire_minutes: int = Field(
        default_factory=lambda: int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    )

    def ensure_directories(self):
        """Ensure all required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def validate_api_key(self) -> bool:
        """Validate that OpenAI API key is set."""
        return bool(self.openai_api_key and self.openai_api_key.strip())


# Global configuration instance
config = Config()
config.ensure_directories()




